"""Launch the pinned SWE service, publish readiness, and clean up on shutdown."""
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
import urllib.request


def serve(out, env, python, project, registry):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(48)
    connection = Path(os.environ['SWE_CONNECTION'])
    service = None
    log = (out / 'service.log').open('w')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        service = subprocess.Popen([python, '-m', 'swebench_service', '--host', '127.0.0.1',
            '--port', str(port), '--auth-token=' + token, '--artifact-root', str(out / 'artifacts'),
            '--runtime', 'pyxis', '--image-registry', registry],
            cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        url = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 120
        while True:
            if service.poll() is not None:
                raise RuntimeError('SWE Pyxis service exited; inspect service.log')
            try:
                with urllib.request.urlopen(url + '/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                # Connection refusal is expected until aiohttp finishes startup.
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('SWE Pyxis service readiness timed out')
            time.sleep(1)
        connection.chmod(0o600)
        connection.write_text(json.dumps({'url': url, 'token': token}))
        print('SWE_READY (ARM Pyxis, current allocation)', flush=True)
        returncode = service.wait()
        raise RuntimeError(f'SWE service exited unexpectedly: {returncode}')
    finally:
        if service is not None and service.poll() is None:
            # Let aiohttp cancel in-flight runs and reap their Pyxis steps first.
            service.terminate()
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid, signal.SIGKILL)
                service.wait()
        log.close()
        connection.write_text('{}')
        roots = [Path(os.environ['SFLOW_WORKFLOW_OUTPUT_DIR'])]
        if os.environ.get('RUN_DIR'):
            roots.append(Path(os.environ['RUN_DIR']))
        for root in roots:
            for path in root.rglob('*.yaml'):
                content = path.read_text()
                if token in content:
                    path.write_text(content.replace(token, '<redacted>'))


def main():
    mode = os.environ['MODE']
    if mode not in ('perf', 'acc', 'both'):
        raise ValueError('MODE must be perf, acc, or both')
    if mode == 'perf':
        print('SWE_READY (perf: service disabled)', flush=True)
        return
    registry = os.environ['SWE_IMAGE_REGISTRY'].strip().rstrip('/')
    if not registry or '<' in registry or '/' not in registry:
        raise ValueError('Set SWE_IMAGE_REGISTRY to the registry/namespace containing ARM64 SWE images')

    out = Path(os.environ['SFLOW_TASK_OUTPUT_DIR']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    env = dict(os.environ)
    # The service creates its own Slurm steps inside this allocation.
    for key in list(env):
        if key.startswith('SLURM_STEP') or key in (
            'SLURM_NTASKS', 'SLURM_NPROCS', 'SLURM_NTASKS_PER_NODE', 'SLURM_TASKS_PER_NODE',
            'SLURM_CPUS_PER_TASK', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID',
            'SLURM_CPU_BIND', 'SLURM_CPU_BIND_LIST', 'SLURM_CPU_BIND_TYPE',
        ):
            env.pop(key)
    serve(out, env, os.environ.get('SWE_SERVICE_PYTHON', 'python3'),
          os.environ.get('SWE_SERVICE_DIRECTORY'), registry)


if __name__ == '__main__':
    main()
