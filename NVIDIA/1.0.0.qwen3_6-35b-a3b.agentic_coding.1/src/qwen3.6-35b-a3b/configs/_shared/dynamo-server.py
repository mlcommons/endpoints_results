#!/usr/bin/env python3
"""Four TP1 vLLM workers behind Dynamo.

One owned allocation, file discovery, session-affine routing (round-robin by default), and
NUMA-local worker CPU sets. Preserve the image's vLLM/torch implementations.
"""
import argparse
import contextlib
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

import yaml
import dynamo_cpu_affinity as affinity

# Log writing allowed after SIGTERM/SIGINT, before Slurm or sflow escalates to SIGKILL.
SIGNAL_DRAIN_SECONDS = 20
# Exit status of the launcher when any child's log sink failed, whatever else ended it (see finish_logs).
LOG_SINK_FAILED_EXIT = 3


def say(message, file=None):
    """print() to file (default: the current sys.stderr) that never raises: stdout and stderr are pipes into
    server.sh's tee, which may be gone. A broken stdout must not stop a healthy server."""
    try:
        print(message, file=file or sys.stderr, flush=True)
    except Exception:
        pass


def engine_arguments(config):
    # HTTP-server settings belong to the Dynamo frontend, not AsyncEngineArgs.
    frontend = {'host', 'port', 'enable-auto-tool-choice', 'tool-call-parser',
                'disable-uvicorn-access-log', 'enable-prompt-tokens-details'}
    args = []
    for key, value in config.items():
        if key in frontend:
            continue
        if isinstance(value, bool):
            args.append('--' + ('' if value else 'no-') + key)
        else:
            args += ['--' + key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return args


def pump(source, log_path, prefix, out):
    """Copy a child's output lines to log_path and to out, prefixed, without letting the sinks block the child.

    The frontend writes its logs synchronously to stderr: if the pipe is not drained, it fills and the frontend
    stalls (CI C512 froze it for up to 3.65 s while Lustre writes stalled, failing requests on the 5 s TCP request
    timeout). The reader only moves lines from the pipe to an unbounded queue; the writer drains the queue in
    batches to the slow sinks (the Lustre log, and stdout, which server.sh tees to Lustre). Order is kept and no
    line is dropped while a sink works.

    A sink that raises (ENOSPC or EIO on the log, BrokenPipeError on stdout) is reported at once on stderr,
    recorded in progress['failed'] and not written again; the other sink still gets every line. With both
    failed, the writer keeps emptying the queue and counts the lines in progress['discarded'], so the queue
    stays short and the child is never blocked. progress also counts the lines 'read' from the pipe and
    'handled' by the writer. Returns (reader, writer, progress); the writer ends after it has handled
    everything read before end of file.
    """
    lines = queue.SimpleQueue()
    progress = {'read': 0, 'handled': 0, 'discarded': 0, 'failed': {}}

    def read():
        try:
            for line in source:
                progress['read'] += 1
                lines.put(line)
        finally:
            lines.put(None)

    def fail(sink, error, action):
        progress['failed'][sink] = repr(error)
        say(f'{prefix}{sink} sink failed {action}: {error!r}')

    def span(batch):
        return f'writing lines {progress["handled"] + 1}-{progress["handled"] + len(batch)}'

    def write():
        try:
            log = open(log_path, 'w')
        except OSError as error:
            log = None
            fail('log', error, 'opening')
        stdout = out
        done = False
        while not done:
            batch = [lines.get()]
            while batch[-1] is not None:
                try:
                    batch.append(lines.get_nowait())
                except queue.Empty:
                    break
            if batch[-1] is None:
                batch.pop()
                done = True
            if not batch:
                continue
            if log is not None:
                try:
                    log.writelines(batch)
                    log.flush()
                except Exception as error:
                    fail('log', error, span(batch))
                    with contextlib.suppress(Exception):
                        log.close()
                    log = None
            if stdout is not None:
                try:
                    stdout.write(''.join(prefix + line for line in batch))
                    stdout.flush()
                except Exception as error:
                    fail('stdout', error, span(batch))
                    stdout = None
            if log is None and stdout is None:
                progress['discarded'] += len(batch)
            progress['handled'] += len(batch)
        if log is not None:
            try:
                log.close()
            except Exception as error:
                fail('log', error, 'closing')

    reader = threading.Thread(target=read, daemon=True)
    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    reader.start()
    return reader, writer, progress


def pipe_holders(inodes):
    """PIDs of other processes that still have one of the given pipes (by inode) open."""
    links = {f'pipe:[{inode}]' for inode in inodes}
    holders = set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            descriptors = list((entry / 'fd').iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) in links:
                    holders.add(int(entry.name))
                    break
            except OSError:
                continue
    return holders


def release(inodes):
    """SIGKILL every process still holding a child's output pipe once the children are gone.

    A reader only reaches end of file when every writer of the pipe has exited. Grandchildren inherit the pipe
    (vLLM's engine core, nsys's target): if a child exits or crashes first, or a grandchild ignores the
    process-group SIGTERM or has left the group, it keeps the pipe open and the reader would never finish.
    """
    if not inodes:
        return
    for pid in pipe_holders(inodes):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def sink_failures(pumps):
    """One line per failed log sink: '[name] sink: error'."""
    return [f'[{name}] {sink}: {error}' for name, _, _, progress in pumps for sink, error in progress['failed'].items()]


def drain(pumps, deadline=None):
    """Wait until each pump has read its pipe to end of file and handled every line, then report sink failures.

    deadline None (normal exit, the children are gone and their pipes released) waits as long as the sinks
    need. A time.monotonic() deadline (signal path) stops there and reports the lines left behind. Returns
    sink_failures(pumps); a writer that ended with lines unhandled counts as a failed 'writer' sink.
    """
    for name, reader, writer, progress in pumps:
        for thread in (reader, writer):
            while thread.is_alive():
                remaining = 1 if deadline is None else min(1, deadline - time.monotonic())
                if remaining <= 0:
                    break
                thread.join(remaining)  # In slices, so that a signal is handled promptly.
        unhandled = progress['read'] - progress['handled']
        if writer.is_alive():
            say(f'[{name}] log writer behind at exit: {unhandled} of {progress["read"]} lines read not written'
                + (', pipe still open' if reader.is_alive() else ''))
        elif unhandled:
            progress['failed'].setdefault('writer', f'ended with {unhandled} lines unwritten')
        if progress['failed']:
            say(f'[{name}] log sinks failed: {progress["failed"]}; {progress["discarded"]} of '
                f'{progress["read"]} lines reached no sink')
    return sink_failures(pumps)


def finish_logs(pumps, signalled):
    """Write the children's remaining output at shutdown; exit LOG_SINK_FAILED_EXIT if any log sink failed.

    Normal exit waits for the whole backlog; after SIGTERM/SIGINT, or a signal during that wait, it writes for
    at most SIGNAL_DRAIN_SECONDS. Without a sink failure it returns and the launcher's own exit stands (1 for an
    error such as a child exit, 128 + signal number for SIGTERM/SIGINT). With one it replaces that status by
    LOG_SINK_FAILED_EXIT, after printing the exception it replaces.
    """
    try:
        failed = drain(pumps, time.monotonic() + SIGNAL_DRAIN_SECONDS if signalled else None)
    except SystemExit:
        failed = drain(pumps, time.monotonic() + SIGNAL_DRAIN_SECONDS)
        if not failed:
            raise
    if failed:
        error = sys.exc_info()[1]
        if error is not None and not isinstance(error, SystemExit):
            say(''.join(traceback.format_exception(error)).rstrip())
        say(f'Dynamo log sinks failed, exiting {LOG_SINK_FAILED_EXIT}: ' + '; '.join(failed))
        raise SystemExit(LOG_SINK_FAILED_EXIT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--ttl', type=int, default=3600)
    parser.add_argument('--frontend-cores', type=int, default=8)
    parser.add_argument('--client-cores', type=int, default=8)
    # Explicit point arguments override SERVER_EXTRA_ENV; session affinity applies in every mode.
    parser.add_argument('--router-mode', choices=('round-robin', 'least-loaded'),
                        default=os.environ.get('DYNAMO_ROUTER_MODE', 'round-robin'))
    args = parser.parse_args()
    nsys_enabled = os.environ.get('NSYS') == '1'
    if nsys_enabled:
        profile_delay = int(os.environ.get('NSYS_DELAY', '1200'))
        profile_duration = int(os.environ.get('NSYS_DURATION', '60'))
        if profile_delay < 0 or profile_duration <= 0:
            raise ValueError('NSYS_DELAY must be nonnegative and NSYS_DURATION must be positive')
        # process-tree (default) or system-wide; system-wide needs one perf descriptor per CPU instead of one per CPU
        # per profiled thread, so it still samples where RLIMIT_NOFILE is too low for process-tree (for example 131072).
        # An explicit NSYS_ARGS replaces the whole option list, so the setting is then unused and not checked.
        nsys_cpu_sampling = os.environ.get('NSYS_CPU_SAMPLING') or 'process-tree'
        if not os.environ.get('NSYS_ARGS') and nsys_cpu_sampling not in ('process-tree', 'system-wide'):
            raise ValueError(f'NSYS_CPU_SAMPLING must be process-tree or system-wide, got {nsys_cpu_sampling!r}')
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    # The server image must already contain the pinned Dynamo integration.
    if importlib.metadata.version('ai-dynamo') != args.version:
        raise RuntimeError(f'The server image must contain ai-dynamo=={args.version}')
    protected = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers')}
    devices = affinity.selected_devices(affinity.gpu_devices(), '')
    allowed = os.sched_getaffinity(0)
    plan, counts = affinity.allocation(devices, affinity.cores_by_numa(affinity.physical_cores(allowed), allowed),
                                        args.frontend_cores, args.client_cores)
    inventory = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                                                         '--format=csv,noheader'], text=True).splitlines()))
    uuids = {affinity.bdf(pci): uuid.strip() for uuid, pci in inventory}
    report = {'active_gpus': 4, 'workers': [{**gpu, 'uuid': uuids[gpu['pci']],
               'cpus': affinity.rendered(plan['worker'][gpu['pci']])} for gpu in devices],
              'frontend_cpus': affinity.rendered(plan['frontend']), 'client_cpus': affinity.rendered(plan['client']),
              'physical_worker_cores': counts, 'protected_packages': protected}
    (args.output / 'dp4-topology.json').write_text(json.dumps(report, indent=2)+'\n')
    runtime = Path(os.environ['TMPDIR']) / 'qwen-dynamo'
    runtime.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'DYN_DISCOVERY_BACKEND': 'file', 'DYN_FILE_KV': str(runtime),
           'DYN_REQUEST_PLANE': 'tcp', 'DYN_EVENT_PLANE': 'zmq',
           'DYN_TOKENIZER': 'fastokens', 'DYN_TOKENIZER_CACHE_BYTES': '8000000000'}
    # Direct Enroot omits HOME. Dynamo 1.4.2 then uses relative MDC blob
    # targets, producing broken tokenizer symlinks. Its supported fallback
    # must be absolute; keep HOME unchanged and caches on job scratch.
    if 'HOME' not in env:
        env['USERPROFILE'] = str(runtime.resolve())
    children, pumps, pipes = [], [], []
    signalled = False

    def start(name, command, mask, overrides):
        if name == 'vllm-worker-0' and nsys_enabled:
            command += ['--enable-layerwise-nvtx-tracing', '--profiler-config', '{"profiler":"cuda"}']
            nsys_options = os.environ.get('NSYS_ARGS') or f'''
                --force-overwrite=true
                --capture-range=cudaProfilerApi
                --capture-range-end=repeat:1:async
                --kill=none
                --wait=primary
                --trace=cuda,nvtx
                --gpu-metrics-devices=all
                --trace-fork-before-exec=true
                --cuda-graph-trace=node
                --sample={nsys_cpu_sampling}
                --cpuctxsw={nsys_cpu_sampling}
            '''
            command = [
                str(Path(os.environ['NSYS_DIR']) / 'nsys'),
                'profile',
                *shlex.split(nsys_options),
                '--output=' + str(args.output / os.environ['NSYS_NAME']),
                *command,
            ]
        process = subprocess.Popen(['taskset', '-c', affinity.rendered(mask), *command],
                env={**env, **overrides}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
        children.append(process)
        pipes.append(os.fstat(process.stdout.fileno()).st_ino)
        pumps.append((name, *pump(process.stdout, args.output / f'{name}.log', f'[{name}] ', sys.stdout)))

    reported = set()

    def alive():
        # A failed log sink is reported but does not stop the server; finish_logs turns it into the exit status.
        for failure in sink_failures(pumps):
            if failure not in reported:
                reported.add(failure)
                say(f'Dynamo log sink failed, server continues, launcher will exit {LOG_SINK_FAILED_EXIT}: {failure}')
        for process in children:
            if process.poll() is not None:
                raise RuntimeError(f'Dynamo process {process.pid} exited {process.returncode}')

    def terminate(signum, frame):
        nonlocal signalled
        signalled = True
        raise SystemExit(128 + signum)

    def profile(action):
        request = urllib.request.Request(f'http://127.0.0.1:18081/engine/control/{action}',
            data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=float(os.environ.get('NSYS_CONTROL_TIMEOUT', 30))) as response:
            result = json.load(response)
        if result.get('status') != 'ok':
            raise RuntimeError(f'Nsys {action} failed: {result}')
        say(f'Nsys {action}: {result}', sys.stdout)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    transport = ['--discovery-backend', 'file', '--request-plane', 'tcp', '--event-plane', 'zmq']
    profile_action = 'start_profile' if nsys_enabled else None
    profile_at = None
    try:
        start('dynamo-frontend', [sys.executable, '-m', 'dynamo.frontend', *transport,
              '--router-mode', args.router_mode, '--router-session-affinity-ttl-secs', str(args.ttl),
              '--http-host', '0.0.0.0', '--http-port', str(config['port']),
              '--router-min-initial-workers', '4'], plan['frontend'], {'CUDA_VISIBLE_DEVICES': ''})
        for index, gpu in enumerate(devices):
            cache = runtime / f'worker-{index}'
            cache.mkdir()
            start(f'vllm-worker-{index}', [sys.executable, '-m', 'dynamo.vllm', *transport,
                  '--dyn-tool-call-parser', 'qwen3_coder', '--dyn-reasoning-parser', 'qwen3',
                  *engine_arguments(config)], plan['worker'][gpu['pci']],
                  {'CUDA_VISIBLE_DEVICES': uuids[gpu['pci']], 'DYN_SYSTEM_PORT': str(18081+index),
                   'VLLM_CACHE_ROOT': str(cache/'vllm'), 'TRITON_CACHE_DIR': str(cache/'triton'),
                   'TORCHINDUCTOR_CACHE_DIR': str(cache/'inductor'), 'FLASHINFER_WORKSPACE_BASE': str(cache/'flashinfer')})
        deadline = time.monotonic() + 5100
        while True:
            alive()
            try:
                for port in range(18081, 18085):
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                        if response.status != 200:
                            raise RuntimeError('Worker health is not ready')
                request = urllib.request.Request(f'http://127.0.0.1:{config["port"]}/v1/chat/completions',
                    data=json.dumps({'model': config['served-model-name'], 'messages': [{'role': 'user', 'content': 'ping'}],
                                     'max_tokens': 1, 'chat_template_kwargs': {'preserve_thinking': True}}).encode(),
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if not json.load(response).get('choices'):
                        raise RuntimeError('Readiness response has no choices')
                break
            except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
                detail = error.read().decode(errors='replace')[:2000] if isinstance(error, urllib.error.HTTPError) else str(error)
                say(f'Dynamo readiness pending: {detail}', sys.stdout)
                if time.monotonic() >= deadline:
                    raise RuntimeError('Dynamo readiness timed out') from error
                time.sleep(10)
        say('DYNAMO_DP4_READY', sys.stdout)
        while True:
            alive()
            if profile_action is not None:
                if profile_at is None and Path(os.environ['NSYS_START_FILE']).exists():
                    profile_at = time.monotonic() + profile_delay
                if profile_at is not None and time.monotonic() >= profile_at:
                    profile(profile_action)
                    if profile_action == 'start_profile':
                        profile_action = 'stop_profile'
                        profile_at = time.monotonic() + profile_duration
                    else:
                        profile_action = None
            time.sleep(2)
    finally:
        if profile_action == 'stop_profile':
            try:
                profile('stop_profile')
            except Exception as error:
                say(f'Nsys cleanup: {error}', sys.stdout)
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # Grandchildren may still be writing their shutdown logs; after a grace period, release what is still held.
        grace = time.monotonic() + 5
        for _, reader, _, _ in pumps:
            reader.join(max(0, grace - time.monotonic()))
        release([pipe for pipe, (_, reader, _, _) in zip(pipes, pumps) if reader.is_alive()])
        finish_logs(pumps, signalled)


if __name__ == '__main__':
    main()
