"""Unit tests for the v1.0.2 production SWE-bench port (Design P): swe_pilot.py serve/host/collect
pieces and the SWE blocks that generate_point_recipes.py puts into the Kimi v1.0.2 SWE recipes.

No cluster access: kubectl is never called. The recipe blocks run under local bash/python3
against temporary directories and a local 127.0.0.1 HTTP server; every token is fake.
"""
import contextlib
import http.server
import importlib.util
import io
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

SWE_DIR = Path(__file__).resolve().parents[1]
KIMI_DIR = SWE_DIR.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_generator_path() -> Path:
    raw = os.environ.get("KIMI_SWE_GENERATOR") or ""
    if raw:
        return Path(raw).expanduser().resolve()
    cand = KIMI_DIR / "generate_point_recipes.py"
    if cand.is_file() and "SWE_CONTRACT" in cand.read_text(errors="replace"):
        return cand
    spec = importlib.util.find_spec("generate_point_recipes")
    if spec and spec.origin:
        return Path(spec.origin).resolve()
    dev = (Path.home() / "<workdir>"
           / "scratch/a2_kimi_gen/swe_dev/generate_point_recipes.py")
    if dev.is_file():
        return dev
    return cand


sp = _load("swe_pilot", SWE_DIR / "pilot" / "swe_pilot.py")
GEN_PATH = _find_generator_path()
gen = _load("generate_point_recipes", GEN_PATH)

FAKE_TOKEN = "FAKE-service-token_" + "x" * 40
HAVE_DROP = (Path(gen.DEFAULT_SLURM_ROOT).is_dir()
             and Path(gen.DEFAULT_SWE_DROP_ROOT).is_dir())


def py_snippet(block: str) -> str:
    """The python3 -c '...' program inside a recipe block (the blocks use no other quotes)."""
    m = re.search(r"python3 -c '(.*?)'", block, re.S)
    assert m, "no python3 -c '...' in the block"
    return m.group(1)


def sflow_subst(block: str, **values) -> str:
    """Resolve the ${{ ... }} expressions a block uses, as sflow would."""
    out = block
    for expr, val in values.items():
        out, n = re.subn(r"\$\{\{\s*" + re.escape(expr) + r"\s*\}\}", str(val), out)
        assert n, f"{expr} not used in the block"
    assert "${{" not in out, out
    return out


class HealthServer:
    """127.0.0.1 HTTP server: GET /health -> `status` (200 by default)."""

    def __init__(self, status=200):
        outer = self
        self.status = status

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):                       # noqa: N802
                self.send_response(outer.status if self.path == "/health" else 404)
                self.end_headers()

            def log_message(self, *_):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_bash(script: str, env: dict, timeout=60) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(["bash", "-c", "set -euo pipefail\n" + script], env={**base, **env},
                          capture_output=True, text=True, timeout=timeout, check=False)


# ----------------------------------------------------------------------------- constants
class SharedPaths(unittest.TestCase):
    def test_handoff_report_and_contract_agree(self):
        self.assertEqual(sp.HANDOFF_DIR, gen.SWE_HANDOFF_DIR)
        self.assertEqual(sp.HANDOFF_DIR, f"{sp.STATE}/handoff")
        self.assertEqual(sp.REPORTS_DIR, gen.REPORTS_DIR)
        self.assertEqual(sp.CLIENT_IMAGE, gen.CLIENT_IMAGE)
        self.assertEqual(sp.SWE_MODES, gen.SWE_MODES)
        self.assertEqual(sp.MODE_MARKER, gen.SWE_MODE_MARKER)
        self.assertEqual(sp.CONNECTION_FILE, gen.SWE_CONNECTION_FILE)
        self.assertEqual(sp.URL_PREFIX, gen.SWE_URL_PREFIX)
        self.assertEqual(sp.SWE_WAIT_S, gen.SWE_WAIT_S)
        self.assertEqual(sp.PULL_SECRET_SUBDIR, gen.PULL_SECRET_SUBDIR)
        c = gen.SWE_CONTRACT
        self.assertEqual(c["handoff_dir"], sp.HANDOFF_DIR)
        self.assertEqual(c["connection_file"], sp.CONNECTION_FILE)
        self.assertEqual(c["wait_s"], sp.SWE_WAIT_S)
        self.assertEqual(c["reports_dir"], sp.REPORTS_DIR)
        self.assertEqual(c["client_task"], sp.CLIENT_TASK)
        self.assertEqual(c["frontend_task"], sp.FRONTEND_TASK)
        self.assertEqual(tuple(c["modes"]), sp.SWE_MODES)
        self.assertEqual(c["mode_marker"], sp.MODE_MARKER)
        self.assertEqual(c["url_prefix"], sp.URL_PREFIX)
        self.assertEqual(c["pull_secret_subdir"], sp.PULL_SECRET_SUBDIR)
        self.assertEqual(c["client_node"],
                         {pt: (2 if pt <= sp.CLIENT_NODE_SPLIT else 4) for pt in sp.KNOWN_POINTS})
        self.assertEqual(c["frontend_node"],
                         {pt: (2 if pt <= sp.CLIENT_NODE_SPLIT else 4) for pt in sp.KNOWN_POINTS})
        entry = (SWE_DIR / "swe-host-entry.sh").read_text()
        self.assertIn(f"STATE={sp.STATE}", entry)
        self.assertIn('CONN="$STATE/handoff/$RUN_ID/swe-connection.json"', entry)
        # The benchmark derives the same <run id>/swe-connection.json from its output dir name.
        self.assertIn('/$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")/swe-connection.json"',
                      gen.SWE_WAIT_BLOCK)


# ----------------------------------------------------------------------------- host pod
class HostPodProduction(unittest.TestCase):
    node = "gke-node-a"

    @staticmethod
    def env(pod):
        return {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}

    def test_pilot_pod_is_unchanged_by_default(self):
        pod = sp.host_pod(sp.POD["host"], "host", self.node, "pilot-3", mode="service")
        self.assertNotIn("SWE_HANDOFF", self.env(pod))
        self.assertNotIn("SWE_SCAN_REPORTS", self.env(pod))
        self.assertNotIn(sp.ALLOC_LABEL, pod["metadata"]["labels"])
        self.assertNotIn("reports", [v["name"] for v in pod["spec"]["volumes"]])

    def test_production_pod(self):
        run_id = "26-10-08-00-00-00_abcdef_kimi_k3_gke_disagg"
        alloc = "abc12345"
        hname = sp.run_pod_name("host", alloc)
        pod = sp.host_pod(hname, "host", self.node, run_id, mode="service",
                          handoff=True, reports_dir=sp.REPORTS_DIR, alloc=alloc)
        self.assertEqual(pod["metadata"]["name"], "kimi-swe-host-abc12345")
        self.assertEqual(pod["metadata"]["labels"][sp.ALLOC_LABEL], alloc)
        env = self.env(pod)
        self.assertEqual((env["SWE_HANDOFF"], env["SWE_SCAN_REPORTS"], env["SWE_RUN_ID"]),
                         ("1", sp.REPORTS_MOUNT, run_id))
        c = pod["spec"]["containers"][0]
        mount = [m for m in c["volumeMounts"] if m["name"] == "reports"][0]
        self.assertEqual((mount["mountPath"], mount.get("readOnly")), (sp.REPORTS_MOUNT, True))
        vol = [v for v in pod["spec"]["volumes"] if v["name"] == "reports"][0]
        self.assertEqual(vol["hostPath"], {"path": sp.REPORTS_DIR, "type": "Directory"})
        # The handoff directory lives under the state hostPath the pod already mounts.
        self.assertIn({"name": "state", "mountPath": sp.STATE},
                      [{k: m[k] for k in ("name", "mountPath")} for m in c["volumeMounts"]])

    def test_production_options_need_service_mode(self):
        for kw in ({"handoff": True}, {"reports_dir": sp.REPORTS_DIR}, {"alloc": "abc12345"}):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.host_pod(sp.POD["smoke"], "smoke", self.node, "smoke-9", mode="smoke", **kw)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sp.host_pod(sp.POD["host"], "host", self.node, "r1", mode="service", alloc="BAD_ALLOC")


# ----------------------------------------------------------------------------- run folders
TASK_MANIFEST = """apiVersion: v1
kind: List
items:
- apiVersion: v1
  kind: ConfigMap
  metadata: {{name: {task}-{alloc}-cfg, namespace: {ns}, labels: {{sflow.ai/allocation: {alloc}}}}}
- apiVersion: v1
  kind: Pod
  metadata: {{name: {name}, namespace: {ns}, labels: {{sflow.ai/allocation: {alloc}}}}}
  spec: {{nodeSelector: {{kubernetes.io/hostname: {node}}}}}
"""
CLIENT_ACC = "datasets:\n  - name: swe_bench\n    accuracy_config:\n      eval_method: swe_bench_scorer\n"
CLIENT_PERF = "datasets:\n  - name: agentic_combined_performance\n"


def make_run(point_dir: Path, run_id: str, *, alloc="abc12345", frontend_alloc=None,
             node="gke-test-node-3", frontend_node="gke-test-node-2",
             bench_name=None, frontend_name=None, ns="default", client=CLIENT_ACC) -> Path:
    falloc = alloc if frontend_alloc is None else frontend_alloc
    d = point_dir / "sflow_output" / run_id
    (d / "benchmark").mkdir(parents=True)
    (d / "frontend").mkdir(parents=True)
    files = (
        ("benchmark/benchmark.k8s.yaml", TASK_MANIFEST.format(
            task="benchmark", alloc=alloc, node=node, ns=ns,
            name=bench_name or f"benchmark-{alloc}")),
        ("frontend/frontend.k8s.yaml", TASK_MANIFEST.format(
            task="frontend", alloc=falloc, node=frontend_node, ns=ns,
            name=frontend_name or f"frontend-{falloc}")),
        ("client.yaml", client),
    )
    for rel, text in files:
        tmp = d / (rel + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, d / rel)          # serve never sees a half-written file
    return d


class RunFolders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.point24 = self.root / "swe_acc" / "8nodes" / "point_24"
        self.point1024 = self.root / "swe_both" / "16nodes" / "point_1024"

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_8node_and_16node(self):
        d24 = make_run(self.point24, "26-10-08-00-00-00_abcdef_kimi_k3_gke_disagg",
                       node="gke-test-node-3", frontend_node="gke-test-node-2")
        run24 = sp.parse_run_dir(d24)
        self.assertEqual((run24.point, run24.run_id, run24.node, run24.alloc),
                         ("24", d24.name, "gke-test-node-3", "abc12345"))
        self.assertTrue(sp.run_ready(d24))
        self.assertTrue(sp.needs_swe(d24))

        d1024 = make_run(self.point1024, "26-10-08-00-00-01_123456_kimi_k3_gke_disagg",
                         alloc="f00baa99", node="gke-test-node-0", frontend_node="gke-test-node-0")
        run1024 = sp.parse_run_dir(d1024)
        self.assertEqual((run1024.point, run1024.run_id, run1024.node, run1024.alloc),
                         ("1024", d1024.name, "gke-test-node-0", "f00baa99"))

    def test_perf_run_needs_no_service(self):
        d = make_run(self.point24, "26-10-08-00-00-00_perf00_kimi_k3_gke_disagg", client=CLIENT_PERF)
        self.assertFalse(sp.needs_swe(d))

    def test_refusals(self):
        bad = [
            dict(node=""),
            dict(node="Bad_Node"),
            dict(bench_name="benchmark-other"),
            dict(frontend_name="frontend-other"),
            dict(alloc="UPPER"),
            dict(alloc="abc12345", frontend_alloc="def67890"),
            dict(ns="kube-system"),
        ]
        for i, kw in enumerate(bad):
            d = make_run(self.point24, f"run-{i}", **kw)
            with self.assertRaises(ValueError, msg=kw):
                sp.parse_run_dir(d)
        # Perf folder layout (missing swe_acc / swe_both parent) is refused.
        perf_point = self.root / "8nodes" / "point_24"
        d_perf = make_run(perf_point, "run-perf-tree")
        with self.assertRaises(ValueError):
            sp.parse_run_dir(d_perf)
        # Two Pod objects in benchmark.k8s.yaml is refused.
        d = make_run(self.point24, "run-two-pods")
        doc = yaml.safe_load((d / "benchmark" / "benchmark.k8s.yaml").read_text())
        doc["items"].append(doc["items"][1])
        (d / "benchmark" / "benchmark.k8s.yaml").write_text(yaml.safe_dump(doc))
        with self.assertRaises(ValueError):
            sp.parse_run_dir(d)
        # Symlinked manifest or client.yaml is refused.
        d_sym = make_run(self.point24, "run-symlink")
        (d_sym / "client.yaml").unlink()
        (d_sym / "client.yaml").symlink_to(d / "client.yaml")
        self.assertFalse(sp.run_ready(d_sym))
        with self.assertRaises(ValueError):
            sp.needs_swe(d_sym)

    def test_a_real_sflow_run_folder_if_present(self):
        runs = sorted(p.parent.parent for p in
                      KIMI_DIR.glob("*nodes/point_*/sflow_output/*/benchmark/benchmark.k8s.yaml")
                      if (p.parent.parent / "frontend" / "frontend.k8s.yaml").is_file())
        if not runs:
            self.skipTest("no complete v1.0.2 sflow run folder on this workstation")
        d = runs[-1]
        bnode, balloc = sp.task_pod(d, "benchmark")
        _, falloc = sp.task_pod(d, "frontend")
        self.assertEqual(balloc, falloc)
        self.assertRegex(bnode, sp.NODE_RE)


class HitsInside(unittest.TestCase):
    def test_only_hits_in_the_copied_directory_block_the_copy(self):
        root = f"{sp.STATE}/runs/r1"
        hits = [{"path": f"{root}/swe_service/service.log", "service_token": 1},
                {"path": f"{sp.REPORTS_MOUNT}/r1/config.yaml", "service_token": 1},
                {"path": f"{sp.STATE}/runs/r10/x"},       # a sibling, not inside r1
                {"service_token": 1}, "garbage"]          # unexpected formats count as inside
        self.assertEqual(sp.hits_inside(hits, root), [hits[0], hits[3], hits[4]])


class WaitRunEnd(unittest.TestCase):
    sflow_run = sp.SflowRun("24", Path("/nonexistent"), "r1", "gke-test-node-3", "abc12345")
    host = "kimi-swe-host-abc12345"

    def test_run_pod_phase_verifies_allocation_label(self):
        def make_kube(rc, out):
            return SimpleNamespace(
                run=lambda *a, **k: subprocess.CompletedProcess(a, rc, out.encode(), b""))

        self.assertEqual(sp.run_pod_phase(make_kube(0, "Running|abc12345"), "benchmark", self.sflow_run),
                         "Running")
        self.assertEqual(sp.run_pod_phase(make_kube(0, ""), "benchmark", self.sflow_run),
                         "Absent")
        self.assertEqual(sp.run_pod_phase(make_kube(1, ""), "benchmark", self.sflow_run),
                         "Unknown")
        self.assertEqual(sp.run_pod_phase(make_kube(0, "Running|other999"), "benchmark", self.sflow_run),
                         "Foreign")
        self.assertEqual(sp.run_pod_phase(make_kube(0, "Running|"), "benchmark", self.sflow_run),
                         "Foreign")

    def wait(self, script, timeout=3600.0, stop=False):
        """script: list of (benchmark, frontend, host) phases, one per poll; the last repeats."""
        calls = {"r": 0, "h": 0}

        def rphase(_kube, task, _run):
            step = script[min(calls["r"] // 2, len(script) - 1)]
            calls["r"] += 1
            return step[0] if task == "benchmark" else step[1]

        def hphase(_kube, _name, _ns):
            step = script[min(calls["h"], len(script) - 1)]
            calls["h"] += 1
            return step[2]

        event = threading.Event()
        if stop:
            event.set()
        orig_r, orig_h = sp.run_pod_phase, sp.pod_phase
        sp.run_pod_phase, sp.pod_phase = rphase, hphase
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return sp.wait_run_end(None, self.sflow_run, self.host, timeout, 0, event)
        finally:
            sp.run_pod_phase, sp.pod_phase = orig_r, orig_h

    def test_reasons(self):
        R = "Running"
        self.assertEqual(self.wait([("Pending", R, R), (R, R, R), ("Succeeded", R, R)]),
                         "benchmark pod Succeeded")
        self.assertEqual(self.wait([(R, R, R), ("Failed", R, R)]), "benchmark pod Failed")
        self.assertEqual(self.wait([(R, R, R), ("Absent", R, R)]), "benchmark pod removed")
        self.assertIn("ended before its benchmark started (frontend pod removed)",
                      self.wait([("Absent", R, R), ("Absent", "Absent", R)]))
        self.assertIn("pod benchmark-abc12345 does not carry allocation abc12345",
                      self.wait([("Foreign", R, R)]))
        self.assertIn("pod frontend-abc12345 does not carry allocation abc12345",
                      self.wait([(R, "Foreign", R)]))
        self.assertIn("SWE service ended before the benchmark", self.wait([(R, R, "Failed")]))
        self.assertEqual(self.wait([(R, R, R)], stop=True), "serve was asked to stop")
        self.assertIn("timed out", self.wait([(R, R, R)], timeout=0))
        # API errors only delay the decision.
        self.assertEqual(self.wait([("Unknown", "Unknown", "Unknown"), ("Succeeded", R, R)]),
                         "benchmark pod Succeeded")


class ServeLoop(unittest.TestCase):
    """cmd_serve: refuses non-SWE groups, ignores existing runs, skips perf runs, serves new acc/both runs."""

    def serve(self, later):
        """Run cmd_serve --once on a temp swe_acc group; `later(point_dir)` creates run folders."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        group = Path(tmp.name) / "swe_acc" / "8nodes"
        point = group / "point_24"
        make_run(point, "26-10-08-00-00-00_old000_kimi_k3_gke_disagg")       # existing: ignored
        served = []
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        orig = (sp.require_configmaps, sp.serve_one, sp.UNREADABLE_RETRY_S)
        sp.require_configmaps = lambda *a, **k: None
        sp.serve_one = lambda kube, args, run, stopping: served.append(run) or True
        sp.UNREADABLE_RETRY_S = 0.2
        done = threading.Event()

        def watchdog():
            if not done.wait(30):
                os.kill(os.getpid(), signal.SIGTERM)      # handled by serve: stop

        threading.Thread(target=later, args=(point,), daemon=True).start()
        threading.Thread(target=watchdog, daemon=True).start()
        kube = SimpleNamespace(dry_run=False, exists=lambda kind, name: True)
        args = SimpleNamespace(group=str(group), points=["24"], once=True, poll=0.05)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rc = sp.cmd_serve(args, kube)
        finally:
            done.set()
            sp.require_configmaps, sp.serve_one, sp.UNREADABLE_RETRY_S = orig
            for s, h in saved.items():
                signal.signal(s, h)
        return rc, served, out.getvalue()

    def test_refuses_non_swe_node_group(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        perf_group = Path(tmp.name) / "8nodes"
        (perf_group / "point_24").mkdir(parents=True)
        kube = SimpleNamespace(dry_run=False, exists=lambda kind, name: True)
        args = SimpleNamespace(group=str(perf_group), points=["24"], once=True, poll=0.05)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            sp.cmd_serve(args, kube)
        self.assertIn("is not a SWE node group", err.getvalue())

    def test_snapshot_skip_and_serve_once(self):
        def later(point):
            time.sleep(0.3)
            make_run(point, "26-10-08-00-00-01_perf00_kimi_k3_gke_disagg", client=CLIENT_PERF)
            time.sleep(0.3)
            make_run(point, "26-10-08-00-00-02_acc000_kimi_k3_gke_disagg", alloc="def67890")

        rc, served, out = self.serve(later)
        self.assertEqual(rc, 1, out)     # the unexpected perf run in swe_acc is counted as a problem
        self.assertEqual([r.run_id for r in served], ["26-10-08-00-00-02_acc000_kimi_k3_gke_disagg"])
        self.assertEqual(served[0].alloc, "def67890")
        self.assertEqual(served[0].node, "gke-test-node-3")
        self.assertIn("1 existing run(s) ignored", out)
        self.assertIn("not an acc/both run; no SWE service", out)

    def test_an_unreadable_run_folder_is_retried_then_reported(self):
        def later(point):
            time.sleep(0.2)
            make_run(point, "26-10-08-00-00-01_bad000_kimi_k3_gke_disagg", bench_name="benchmark-other")
            time.sleep(0.6)
            make_run(point, "26-10-08-00-00-02_acc000_kimi_k3_gke_disagg")

        rc, served, out = self.serve(later)
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("ERROR: run 26-10-08-00-00-01_bad000_kimi_k3_gke_disagg"), 1)
        self.assertEqual([r.run_id for r in served], ["26-10-08-00-00-02_acc000_kimi_k3_gke_disagg"])


# ----------------------------------------------------------------------------- recipe blocks
class WaitBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_id = "26-10-08-00-00-00_abcdef_kimi_k3_gke_disagg"
        self.handoff = self.root / "handoff"
        self.conn = self.handoff / self.run_id / "swe-connection.json"
        self.conn.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def probe(self, content):
        if content is None:
            self.conn.unlink(missing_ok=True)
        else:
            self.conn.write_text(content)
        return subprocess.run(["python3", "-c", py_snippet(gen.SWE_WAIT_BLOCK)],
                              env={"PATH": os.environ["PATH"], "SWE_CONNECTION": str(self.conn)},
                              capture_output=True, text=True, timeout=30, check=False).returncode

    def test_probe(self):
        with HealthServer() as srv:
            port = srv.url.rsplit(":", 1)[1]
            self.assertEqual(self.probe(f'{{"url": "{srv.url}", "token": "{FAKE_TOKEN}"}}'), 0)
            self.assertEqual(self.probe("{}"), 1)                              # not written yet
            self.assertEqual(self.probe(""), 1)                                # being written
            self.assertEqual(self.probe(None), 1)                              # no file
            self.assertEqual(self.probe(f'{{"url": "http://localhost:{port}"}}'), 1)
            srv.status = 503
            self.assertEqual(self.probe(f'{{"url": "{srv.url}"}}'), 1)

    def block(self, wait_s=2):
        text = gen.SWE_WAIT_BLOCK
        self.assertEqual(text.count("sleep 10"), 1)
        text = text.replace("sleep 10", "sleep 0.2")                          # test speed only
        return sflow_subst(text, **{"artifacts.SWE_HANDOFF_DIR.path": self.handoff,
                                    "variables.SWE_WAIT_S": wait_s})

    def env(self, mode="acc"):
        return {"BENCHMARK_MODE": mode,
                "SFLOW_WORKFLOW_OUTPUT_DIR": str(self.root / "out" / self.run_id)}

    def test_waits_for_this_runs_service(self):
        with HealthServer() as srv:
            self.conn.write_text(f'{{"url": "{srv.url}", "token": "{FAKE_TOKEN}"}}')
            p = run_bash(self.block() + 'echo "CONN=${SWE_CONNECTION}"\n', self.env())
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("SWE service is ready", p.stdout)
        self.assertIn(f"CONN={self.conn}", p.stdout)
        self.assertNotIn(FAKE_TOKEN, p.stdout + p.stderr)

    def test_fails_at_the_deadline_without_a_service(self):
        self.conn.write_text("{}")
        t0 = time.monotonic()
        p = run_bash(self.block(wait_s=1), self.env())
        self.assertEqual(p.returncode, 1)
        self.assertIn("FATAL: no SWE service for this run", p.stderr)
        self.assertIn("run_swe_sweep.sh", p.stderr)
        self.assertLess(time.monotonic() - t0, 20)

    def test_perf_does_not_wait(self):
        p = run_bash(self.block(wait_s=1) + 'echo "CONN=${SWE_CONNECTION:-unset}"\n',
                     self.env("perf"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("CONN=unset", p.stdout)

    @unittest.skipUnless(HAVE_DROP, "NVIDIA drops not on this workstation")
    def test_the_drops_swe_block_reads_the_handoff_file(self):
        swe_drop = gen.SweDrop(gen._resolve_swe_drop_root(Path(gen.DEFAULT_SLURM_ROOT), None))
        drop_swe = swe_drop.bench_swe
        self.assertTrue(drop_swe.startswith(gen.DROP_SWE_PREFIX))
        check = ('echo "URL=${SWE_SERVICE_URL} HF=${HF_HUB_OFFLINE}"\n'
                 '[ "${SWE_SERVICE_TOKEN}" = "${EXPECT}" ] && echo TOKEN_OK\n')
        with HealthServer() as srv:
            self.conn.write_text(f'{{"url": "{srv.url}", "token": "{FAKE_TOKEN}"}}')
            p = run_bash(self.block() + drop_swe + check, {**self.env("both"), "EXPECT": FAKE_TOKEN})
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn(f"URL={srv.url} HF=0", p.stdout)
        self.assertIn("TOKEN_OK", p.stdout)
        self.assertNotIn(FAKE_TOKEN, p.stdout + p.stderr)
        # perf: the drop's defaults, no file read.
        p = run_bash(self.block() + drop_swe + 'echo "URL=${SWE_SERVICE_URL} T=${SWE_SERVICE_TOKEN}"\n',
                     self.env("perf"))
        self.assertIn("URL=http://127.0.0.1:1 T=disabled", p.stdout)


class RedactBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.out, self.rep = root / "out" / "run-1", root / "reports" / "run-1"
        (self.out / "benchmark").mkdir(parents=True)
        (self.rep / "accuracy").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_block(self, token, mode="acc"):
        return run_bash(gen.SWE_REDACT_BLOCK, {"BENCHMARK_MODE": mode, "SWE_SERVICE_TOKEN": token,
                                               "SFLOW_WORKFLOW_OUTPUT_DIR": str(self.out),
                                               "REPORT_DIR": str(self.rep)})

    def test_the_drops_rule(self):
        cfg = self.rep / "config.yaml"
        cfg.write_text(f"swebench_service_auth_token: {FAKE_TOKEN}\nother: 1\n")
        nested = self.out / "benchmark" / "x.yaml"
        nested.write_text(f"a: {FAKE_TOKEN} b: {FAKE_TOKEN}\n")
        clean = self.rep / "accuracy" / "clean.yaml"
        clean.write_text("score: 0.5\n")
        events = self.rep / "events.jsonl"                    # not *.yaml: the drop leaves it
        events.write_text(f'{{"t": "{FAKE_TOKEN}"}}\n')
        p = self.run_block(FAKE_TOKEN)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(cfg.read_text(), "swebench_service_auth_token: <redacted>\nother: 1\n")
        self.assertEqual(nested.read_text(), "a: <redacted> b: <redacted>\n")
        self.assertEqual(clean.read_text(), "score: 0.5\n")
        self.assertIn(FAKE_TOKEN, events.read_text())
        self.assertIn("SWE token redacted in 2 file(s)", p.stdout)
        self.assertNotIn(FAKE_TOKEN, p.stdout + p.stderr)

    def test_refuses_without_a_token(self):
        (self.rep / "config.yaml").write_text("swebench_service_auth_token: disabled\n")
        for token in ("", "disabled"):
            p = self.run_block(token)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("FATAL: SWE_SERVICE_TOKEN is not set", p.stderr)

    def test_perf_is_a_no_op(self):
        cfg = self.rep / "config.yaml"
        cfg.write_text(f"x: {FAKE_TOKEN}\n")
        p = self.run_block("disabled", mode="perf")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertIn(FAKE_TOKEN, cfg.read_text())


@unittest.skipUnless(HAVE_DROP, "NVIDIA drops not on this workstation")
class RecipeCheck(unittest.TestCase):
    """Both generate_point_recipes.py verify_swe and swe_pilot.py check_swe_recipe_text catch a broken SWE port."""

    @classmethod
    def setUpClass(cls):
        slurm_root = Path(gen.DEFAULT_SLURM_ROOT)
        cls.drop = gen.Drop(slurm_root)
        cls.swe_drop = gen.SweDrop(gen._resolve_swe_drop_root(slurm_root, None))

    def problems(self, c=24, mode="acc", pull_secret=False, mutation=None):
        text = gen.render_recipe_swe(self.drop, self.swe_drop, c, mode, pull_secret=pull_secret)
        launch = gen.render_launch_swe(self.drop, c, mode, pull_secret=pull_secret)
        if mutation:
            doc = yaml.safe_load(text)
            mutation(doc, [t for t in doc["workflow"]["tasks"] if t["name"] == "benchmark"][0]["script"])
            text = yaml.safe_dump(doc, sort_keys=False, width=10000)
        n_nodes = gen.EXPECTED[c][6]
        p1 = gen.verify_swe(self.drop, self.swe_drop, c, mode, text, launch, f"C{c}",
                            pull_secret=pull_secret)
        p2 = sp.check_swe_recipe_text(c, mode, n_nodes, text, launch, f"C{c}")
        return "\n".join(p1 + p2)

    def test_fresh_renders_pass_8node_and_16node_all_variants(self):
        for c in (24, 192):
            for mode in ("acc", "both"):
                for ps in (False, True):
                    with self.subTest(c=c, mode=mode, pull_secret=ps):
                        self.assertEqual(self.problems(c=c, mode=mode, pull_secret=ps), "")

    def test_mutations_are_caught(self):
        def swap(doc, s):
            i = s.index(gen.SWE_REDACT_BLOCK)
            s[i], s[i + 1] = s[i + 1], s[i]

        def handoff_elsewhere(doc, s):
            for a in doc["artifacts"]:
                if a["name"] == "SWE_HANDOFF_DIR":
                    a["uri"] = "fs:///tmp/elsewhere"

        def wrong_client_node(doc, s):
            doc["variables"]["CLIENT_NODE"]["value"] = 0

        cases = [
            (swap, "SWE_REDACT_BLOCK"),
            (lambda doc, s: s.remove(gen.SWE_WAIT_BLOCK), "SWE_WAIT_BLOCK"),
            (lambda doc, s: s.insert(1, "set -x"), "xtrace"),
            (lambda doc, s: doc["variables"].update(SWE_SERVICE_TOKEN={"value": "disabled"}),
             "SWE_SERVICE_TOKEN"),
            (handoff_elsewhere, "SWE_HANDOFF_DIR"),
            (wrong_client_node, "CLIENT_NODE"),
        ]
        for mutation, want in cases:
            with self.subTest(want=want):
                self.assertIn(want, self.problems(mutation=mutation))


if __name__ == "__main__":
    unittest.main()
