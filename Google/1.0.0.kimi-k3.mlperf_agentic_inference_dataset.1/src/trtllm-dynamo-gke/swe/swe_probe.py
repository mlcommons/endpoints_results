#!/usr/bin/env python3
"""Helpers for the kimi-swe-host pod (Kimi-K3 SWE-bench on GKE, 1-node pilot).

conn, snapshot, leakscan and postcheck-report use only the standard library; the entry script
runs them as root with `python3 -I` from /, so nothing on sys.path is writable by the service user.
smoke and prewarm run as the service user with the service venv's python, so they load the
same drop-patched pinned runtime the service uses (swebench_service.pyxis_environment with
NVIDIA's swe-enroot-transport.py appended) and go through the srun shim like the service.
Nothing here prints a secret: checks report booleans, counts, timings and file names.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import mmap
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TRANSPORT_MARKER = "# ARM adapter for the pinned SWE Pyxis runtime's command transport."
SKIP_DIRS = {"enroot-cache"}      # imported image layers and .sqsh files only
MAX_SCAN_BYTES = 1 << 30
# swe-service.py always writes url = f'http://127.0.0.1:{port}'. Accept nothing else.
SERVICE_URL_RE = re.compile(r"http://127\.0\.0\.1:[0-9]{1,5}")
# smoke --metadata-block (CS-NET-001): from inside a sandbox, a plain TCP connect to the GCE
# and GKE metadata servers must be refused at once by the host's block. Nothing is sent, so
# no metadata is read even if the block were missing.
MDB_TARGETS = ("<ip>/80", "<ip>/988")
MDB_TARGET_RE = re.compile(r"[0-9.]{7,15}/[0-9]{1,5}")


def _terminate(signum, frame):
    # SIGTERM from the entry script: unwind so the finally blocks run the drop's teardown.
    raise SystemExit(128 + signum)


def exc_chain(exc: BaseException | None, depth: int = 4) -> str:
    """An exception and its causes. The pinned runtime wraps the captured srun output in
    __cause__ ("failed to start Pyxis container" <- "Pyxis infrastructure failure ...")."""
    parts: list[str] = []
    while exc is not None and len(parts) < depth:
        parts.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return " <- caused by ".join(parts)


# ----------------------------------------------------------------------- root helpers
def read_small(path: str, limit: int = 1 << 20) -> bytes:
    """Read a small regular file from a directory the service user can write.

    Runs as root: never follow a swapped-in symlink, never block on a FIFO, and re-check
    the type and size on the open descriptor.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise OSError(f"{path}: not a regular file of at most {limit} bytes")
        data = f.read(limit + 1)
    if len(data) > limit:
        raise OSError(f"{path}: larger than {limit} bytes")
    return data


def read_connection(path: str) -> dict | None:
    """The connection file as a dict, or None if it is missing, empty, unsafe or not an object."""
    try:
        data = json.loads(read_small(path))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def cmd_conn(args) -> int:
    """Check the connection file: 0 if ready (or reset, with --expect-reset), else 1.

    --print-url prints the URL only after it matches the drop's loopback form.
    """
    data = read_connection(args.connection)
    if args.expect_reset:
        return 0 if data == {} else 1
    url = (data or {}).get("url")
    if not (isinstance(url, str) and SERVICE_URL_RE.fullmatch(url)):
        return 1
    if args.print_url:
        print(url)
    return 0


def cmd_snapshot(args) -> int:
    """Copy the service token from the connection file to a 0600 file for the leak scan."""
    token = (read_connection(args.connection) or {}).get("token", "")
    if not (isinstance(token, str) and token):
        print("snapshot: the connection file holds no service token", file=sys.stderr)
        return 1
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return 0


def _netrc_passwords(path: str | None) -> list[str]:
    # Pass the root-owned Secret mount (its files are kubelet symlinks into ..data), not the
    # service user's 0600 copy, so the service user cannot change what the scan looks for.
    if not path:
        return []
    try:
        words = Path(path).read_text().split()
    except OSError:
        return []
    return sorted({words[i + 1] for i, w in enumerate(words[:-1]) if w == "password"})


def _count(buf, needle: bytes) -> int:
    n, i = 0, buf.find(needle)
    while i != -1:
        n += 1
        i = buf.find(needle, i + 1)
    return n


def write_report(path: str, text: str) -> None:
    """Write a report as a new root-owned file in a directory the service user can write.

    Unlink first and create with O_EXCL|O_NOFOLLOW, so a symlink or file planted at that
    name cannot redirect the write or keep the report writable by the service user.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def cmd_leakscan(args) -> int:
    """Search every regular file under the roots for the secrets; report counts only."""
    needles: dict[str, bytes] = {}
    if args.secret_file and Path(args.secret_file).is_file():
        token = Path(args.secret_file).read_text().strip()
        if token:
            needles["service_token"] = token.encode()
    for i, password in enumerate(p for p in _netrc_passwords(args.netrc) if len(p) >= 8):
        needles[f"registry_token_{i + 1}"] = password.encode()
    print("leakscan: searching for " + (", ".join(needles) or "nothing (no secrets available)"),
          flush=True)
    hits = {label: 0 for label in needles}
    files_hit, skipped, n_files, n_bytes = [], [], 0, 0
    for root in args.root:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode) or st.st_size == 0:
                    continue
                if st.st_size > MAX_SCAN_BYTES:
                    skipped.append(path)
                    continue
                found = {}
                try:
                    # Runs as root over directories the service user can write: never follow a
                    # swapped-in symlink, never block on a FIFO, re-check the type after open.
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as f:
                        fst = os.fstat(f.fileno())
                        if not stat.S_ISREG(fst.st_mode) or not 0 < fst.st_size <= MAX_SCAN_BYTES:
                            continue
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                            for label, needle in needles.items():
                                c = _count(buf, needle)
                                if c:
                                    found[label] = c
                                    hits[label] += c
                except (OSError, ValueError):
                    continue
                n_files += 1
                n_bytes += st.st_size
                if found:
                    files_hit.append({"path": path, **found})
    report = {"searched_for": list(needles), "files_scanned": n_files, "bytes_scanned": n_bytes,
              "hits": hits, "files_with_hits": files_hit, "skipped_over_1GiB": skipped}
    write_report(args.report, json.dumps(report, indent=2) + "\n")
    print(f"leakscan: {n_files} files, {n_bytes / 1e9:.2f} GB, hits {json.dumps(hits)}", flush=True)
    for entry in files_hit:
        print(f"leakscan: HIT {entry}", flush=True)
    for path in skipped:
        print(f"leakscan: skipped (>1 GiB) {path}", flush=True)
    if not needles:
        return 2
    return 1 if files_hit else 0


def cmd_postcheck_report(args) -> int:
    """Write postcheck.json with write_report (root, in the run directory the service user owns)."""
    if not re.fullmatch(r"none|[0-9]{1,6}", args.leftover):
        raise SystemExit("postcheck-report: --leftover must be none or a count")
    report = {"service_rc": args.service_rc, "ok": args.fail == "0",
              "scratch_retained": [r for r in args.retained.split("\n") if r],
              # Warning only: processes of the service user still alive after the drop's
              # teardown. The entry script kills them, as Slurm does at the end of a step.
              "leftover_processes": None if args.leftover == "none" else int(args.leftover)}
    write_report(args.out, json.dumps(report, indent=2) + "\n")
    return 0


# ----------------------------------------------------------------------- runtime helpers
def _load_runtime(drop: Path):
    """Import the drop-patched pinned runtime and the drop's teardown module."""
    sys.path.insert(0, str(drop))
    from swebench_service import pyxis_environment as pe   # editable install in the service venv
    import swe_cleanup
    if TRANSPORT_MARKER not in Path(pe.__file__).read_text() or "SWE_STEP_NAME" not in pe._SAFE_SRUN_ENV:
        raise RuntimeError(f"{pe.__file__} is not the drop-patched copy")
    return pe, swe_cleanup


def _scratch_env(out: Path, cache: Path) -> Path:
    """Create a scratch and environment like swe-service.py does for the service."""
    scratch = Path(tempfile.mkdtemp(prefix="kimi-swe-", dir="/tmp"))
    env = {"SWE_STEP_NAME": scratch.name}
    for key, directory in {
        "TMPDIR": scratch / "tmp", "XDG_CONFIG_HOME": out / "xdg-config",
        "XDG_DATA_HOME": out / "xdg-data", "XDG_CACHE_HOME": out / "xdg-cache",
        "XDG_RUNTIME_DIR": scratch / "runtime", "HF_HOME": out / "hf-cache",
        "ENROOT_DATA_PATH": scratch / "enroot-data", "ENROOT_RUNTIME_PATH": scratch / "enroot-runtime",
        "ENROOT_CACHE_PATH": cache, "ENROOT_TEMP_PATH": scratch / "enroot-tmp",
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    env.update(TMP=env["TMPDIR"], TEMP=env["TMPDIR"],
               ENROOT_CONFIG_PATH=os.environ["SWE_ENROOT_CONFIG_PATH"],
               ENROOT_MOUNT_HOME="n", ENROOT_LOGIN_SHELL="n", ENROOT_MAX_PROCESSORS="4",
               GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    os.environ.update(env)
    tempfile.tempdir = None      # re-read TMPDIR: the runtime's per-sandbox dirs go under the scratch
    return scratch


def metadata_probe_command(targets=MDB_TARGETS) -> str:
    """Shell command for a sandbox: one TCP connect per HOST/PORT with bash's /dev/tcp.

    Prints `MDB HOST/PORT rc=EXIT MESSAGE` per target and sends nothing over a connection.
    """
    if not targets or not all(MDB_TARGET_RE.fullmatch(t) for t in targets):
        raise ValueError(f"bad metadata probe targets: {targets!r}")
    return ("for t in " + " ".join(targets) + "; do "
            "e=$(timeout 5 bash -c \": </dev/tcp/$t\" 2>&1); rc=$?; "
            "echo \"MDB $t rc=$rc $(printf %s \"$e\" | tr '\\n' ' ')\"; done")


def metadata_probe_ok(output: str, targets=MDB_TARGETS) -> bool:
    """True if every target was refused (exit 1, "Connection refused"). A connect (rc=0), a
    timeout (rc=124) or a bash without /dev/tcp ("No such file or directory") fails."""
    lines = output.splitlines()
    return bool(targets) and all(
        any(re.match(rf"MDB {re.escape(t)} rc=1 .*Connection refused", line) for line in lines)
        for t in targets)


def cmd_smoke(args) -> int:
    """Sandbox round trip through the drop transport and the shim, on a public image."""
    signal.signal(signal.SIGTERM, _terminate)
    os.umask(0o077)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pe, swe_cleanup = _load_runtime(Path(args.drop))
    scratch = _scratch_env(out, out / "enroot-cache")
    checks: dict[str, bool] = {}
    timings: dict[str, float] = {}

    def check(name, ok, detail="", limit=500):
        checks[name] = bool(ok)
        tail = f": {detail[:limit]}" if detail and not ok else ""
        print(f"probe {'ok  ' if ok else 'FAIL'} {name}{tail}", flush=True)

    try:
        cmd = pe.build_srun_command(argv=["true"], image="registry.invalid#x/y:z", name="probe")
        check("drop transport: bare srun + enroot, no --container-* flags",
              cmd[0] == "srun" and not any(a.startswith("--container") for a in cmd))

        t0 = time.monotonic()
        env1 = pe.PyxisEnvironment(image=args.image, run_id="smoke", cwd="/", timeout=args.timeout)
        timings["init_with_import_s"] = round(time.monotonic() - t0, 1)
        check("agent-style init: import + create + start", True)
        r = env1.execute({"command": "echo PROBE_OUT; id -u; cat /proc/1/comm; "
                                     "ls /dev | tr '\\n' ' '; echo; exit 7"})
        lines = r["output"].splitlines()
        check("exit status passes through (7)", r["returncode"] == 7, str(r["returncode"]))
        check("stdout passes through", lines[:1] == ["PROBE_OUT"], r["output"])
        check("root inside the sandbox (remapped)", len(lines) > 1 and lines[1].strip() == "0")
        check("private PID namespace and /proc (pid 1 is timeout)",
              len(lines) > 2 and lines[2].strip() == "timeout", r["output"])
        devs = lines[3].split() if len(lines) > 3 else []
        check("restricted /dev: no GPUs or disks",
              bool(devs) and not any(d.startswith(("nvidia", "nvme", "sd", "dm-", "loop")) for d in devs),
              " ".join(devs))
        if args.metadata_block:
            r = env1.execute({"command": metadata_probe_command()})
            check("metadata block: TCP from the sandbox to "
                  + ", ".join(t.replace("/", ":") for t in MDB_TARGETS) + " refused",
                  metadata_probe_ok(r["output"]), r["output"])
        else:
            print("probe skip metadata block check (no --metadata-block)", flush=True)
        r = env1.execute({"command": "sleep 30"}, timeout=3)
        check("command timeout path (-1, timed out)",
              r["returncode"] == -1 and "timed out" in r.get("exception_info", ""), str(r["returncode"]))
        try:
            env1.execute({"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nPROBE_PATCH\\n'"})
            check("submission raises Submitted", False, "no exception")
        except Exception as exc:  # noqa: BLE001 - the expected outcome is an exception
            check("submission raises Submitted", type(exc).__name__ == "Submitted", type(exc).__name__)
        env1.cleanup()

        t0 = time.monotonic()
        env2 = pe.PyxisEnvironment(image=args.image, run_id="smoke2", cwd="/", timeout=args.timeout)
        timings["init_cached_image_s"] = round(time.monotonic() - t0, 1)
        env2.cleanup()

        log_dir = out / "eval"
        log_dir.mkdir(exist_ok=True)
        patch = log_dir / "patch.diff"
        patch.write_text("PROBE_EVAL\n")
        status = log_dir / ".mlperf_srun_status"
        status.write_text("")
        t0 = time.monotonic()
        res = pe.run_srun_step(
            image=args.image, workdir="/", status_path=status, timeout_s=120,
            mounts=[(patch, "/tmp/swebench_patch.diff"), (status, "/tmp/.mlperf_srun_status")],
            stderr=subprocess.PIPE, argv=["bash", "-c", "cat /tmp/swebench_patch.diff"])
        timings["grader_style_step_s"] = round(time.monotonic() - t0, 1)
        check("grader-style ephemeral step with file mounts",
              res.returncode == 0 and "PROBE_EVAL" in res.stdout, res.stdout + res.stderr)

        listed = subprocess.run(pe.build_srun_command(argv=["enroot", "list", "-f"]),
                                capture_output=True, text=True, timeout=60, env=pe.safe_srun_env())
        names = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
        check("no sandbox containers left", listed.returncode == 0
              and not any(n.startswith("pyxis_") for n in names), " ".join(names) + listed.stderr)

        bad = subprocess.run(
            ["srun", "--overlap", f"--jobid={os.environ['SLURM_JOB_ID']}", "-N1", "-n1",
             f"--nodelist={os.environ['SLURMD_NODENAME']}", f"--chdir={os.environ['TMPDIR']}",
             f"--job-name={scratch.name}", "--container-image=x", "true"],
            capture_output=True, text=True, timeout=30, env=pe.safe_srun_env())
        check("shim rejects Pyxis flags (exit 2)", bad.returncode == 2, str(bad.returncode))
    except Exception as exc:  # noqa: BLE001 - report and fail the probe
        check("probe ran to the end", False, exc_chain(exc), limit=4000)
    finally:
        removed = swe_cleanup.cleanup_scratch(scratch, dict(os.environ))
        check("drop teardown (swe_cleanup) removed the probe scratch", removed and not scratch.exists())
        (out / "probe.json").write_text(json.dumps(
            {"image": args.image, "checks": checks, "timings": timings}, indent=2) + "\n")
        print("probe timings: " + json.dumps(timings), flush=True)
    return 0 if checks and all(checks.values()) else 1


def _first_instance_ids(n: int, out: Path) -> list[str]:
    # Same source and order as the client's SWEBench loader (HF dataset order, first n).
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test",
                      cache_dir=str(out / "hf-cache"))
    return [str(x) for x in ds["instance_id"][:n]]


def cmd_prewarm(args) -> int:
    """Import SWE images into the persistent cache through the drop transport.

    Each image goes through run_srun_step(image=..., name=None) exactly like a grading step:
    import under flock into $ENROOT_CACHE_PATH/swe-images/<sha256(docker://ref)>.sqsh, then an
    ephemeral create/start/remove. Same refs and keys as the service; only earlier, and with a
    longer timeout than the agent's 300 s init limit.
    """
    signal.signal(signal.SIGTERM, _terminate)
    os.umask(0o077)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pe, swe_cleanup = _load_runtime(Path(args.drop))
    scratch = _scratch_env(out, Path(args.cache))
    if args.ids.startswith("first:"):
        ids = _first_instance_ids(int(args.ids.split(":", 1)[1]), out)
    else:
        ids = [i for i in args.ids.split(",") if i]
    results = []

    def one(iid: str) -> dict:
        t0 = time.monotonic()
        try:
            pe.run_srun_step(image=pe.resolve_image(args.registry, iid), argv=["true"], workdir="/",
                             status_path=out / f".status-{iid}", timeout_s=args.timeout)
            res = {"instance_id": iid, "ok": True}
        except Exception as exc:  # noqa: BLE001 - recorded per image
            res = {"instance_id": iid, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
        res["seconds"] = round(time.monotonic() - t0, 1)
        print(f"prewarm {json.dumps(res)}", flush=True)
        return res

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(one, ids))
    finally:
        swe_cleanup.cleanup_scratch(scratch, dict(os.environ))
        (out / "prewarm.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if results and all(r["ok"] for r in results) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("conn")
    s.add_argument("--connection", required=True)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--print-url", action="store_true")
    g.add_argument("--expect-reset", action="store_true")
    s = sub.add_parser("snapshot")
    s.add_argument("--connection", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("leakscan")
    s.add_argument("--root", action="append", required=True)
    s.add_argument("--secret-file")
    s.add_argument("--netrc")
    s.add_argument("--report", required=True)
    s = sub.add_parser("postcheck-report")
    s.add_argument("--out", required=True)
    s.add_argument("--service-rc", required=True)
    s.add_argument("--fail", required=True, choices=("0", "1"))
    s.add_argument("--retained", default="")
    s.add_argument("--leftover", default="none")
    s = sub.add_parser("smoke")
    s.add_argument("--image", required=True)
    s.add_argument("--drop", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--timeout", type=int, default=300)   # the template's environment.timeout
    s.add_argument("--metadata-block", action="store_true",
                   help="also check from a sandbox that TCP to the metadata servers is refused")
    s = sub.add_parser("prewarm")
    s.add_argument("--ids", required=True)
    s.add_argument("--registry", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--drop", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args(argv)
    return {"conn": cmd_conn, "snapshot": cmd_snapshot, "leakscan": cmd_leakscan,
            "postcheck-report": cmd_postcheck_report,
            "smoke": cmd_smoke, "prewarm": cmd_prewarm}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
