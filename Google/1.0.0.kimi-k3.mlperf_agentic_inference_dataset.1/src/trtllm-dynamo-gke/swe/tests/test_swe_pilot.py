"""Unit tests for pilot/swe_pilot.py: pins, ConfigMaps, pod specs, redaction, credentials.

No cluster access: kubectl is never called (a fake Kube records what would be sent).
"""
import base64
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

SWE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swe_pilot", SWE_DIR / "pilot" / "swe_pilot.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

TOKEN = "<gitlab-pat-prefix>-EXAMPLEexample0123456789"
HAVE_INPUTS = sp.DEFAULT_DROP.is_dir() and all((SWE_DIR / "artifacts" / n).is_file() for n in sp.ARTIFACTS)


class FakeKube:
    def __init__(self):
        self.redact = sp.Redactor()
        self.dry_run = False
        self.applied = []

    def apply(self, obj):
        self.applied.append(obj)


def args(**kw):
    return SimpleNamespace(drop_dir=str(sp.DEFAULT_DROP), artifacts=str(SWE_DIR / "artifacts"), **kw)


class PinsAndConfigMaps(unittest.TestCase):
    def test_pins_match_the_entry_script(self):
        self.assertEqual(sp.entry_pins((SWE_DIR / "swe-host-entry.sh").read_text()), sp.PINS)

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_configmaps(self):
        cms = {cm["metadata"]["name"]: cm for cm in sp.desired_configmaps(args())}
        self.assertEqual(set(cms), {sp.CM_DROP, sp.CM_HOST, sp.CM_CLIENT, sp.CM_MODEL})
        for cm in cms.values():
            size = sum(len(v.encode()) for v in cm["data"].values()) + sum(
                len(base64.b64decode(v)) for v in cm.get("binaryData", {}).values())
            self.assertLess(size, 900 * 1024, cm["metadata"]["name"])
        host = cms[sp.CM_HOST]
        for name in sp.ARTIFACTS:
            self.assertEqual(sp.sha256(base64.b64decode(host["binaryData"][name])), sp.PINS[name])
        for name in sp.DROP_FILES:
            self.assertEqual(sp.sha256(cms[sp.CM_DROP]["data"][name].encode()), sp.PINS[name])
        lines = host["data"]["MANIFEST.sha256"].splitlines()
        self.assertEqual([l.split("  ")[1] for l in lines], list(sp.HOST_FILES + sp.ARTIFACTS))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}  \S+", l) for l in lines))
        self.assertEqual(cms[sp.CM_CLIENT]["data"]["perf-dataset-unused.jsonl"], "")
        again = {cm["metadata"]["name"]: cm for cm in sp.desired_configmaps(args())}
        self.assertEqual(again[sp.CM_HOST]["metadata"]["annotations"],
                         host["metadata"]["annotations"])

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_a_changed_pin_is_refused(self):
        saved = dict(sp.PINS)
        try:
            sp.PINS["swe_cleanup.py"] = "0" * 64
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.desired_configmaps(args())
        finally:
            sp.PINS.clear()
            sp.PINS.update(saved)


class ClientConfigFaithfulness(unittest.TestCase):
    def drop_config(self) -> dict:
        """The drop's CLIENT_CONFIG (templates/ifb.yaml) for BENCHMARK_MODE=acc and point_12."""
        template = (sp.DEFAULT_DROP.parent / "templates" / "ifb.yaml").read_text()
        block = template.split("    content: |\n", 1)[1].split("\n  - name: DRAFT", 1)[0]
        rendered = []
        for line in block.splitlines():
            if line.strip().startswith("{%"):
                continue                                  # BENCHMARK_MODE=acc keeps both blocks
            line = line.replace("${{ variables.CONCURRENCY }}", "12")
            line = line.replace("${{ variables.NUM_TRAJECTORIES }}", "613")
            rendered.append(line[6:])
        drop = yaml.safe_load("\n".join(rendered))
        self.assertEqual(drop["name"], "kimi-k3-c12")
        drop["name"] = "kimi-k3-swe-pilot"                                    # PILOT 1
        drop.pop("submission_ref")                                            # PILOT 2
        drop["model_params"]["name"] = "${PILOT_MODEL_NAME}"                  # PILOT 3
        drop["endpoint_config"]["endpoints"] = ["http://127.0.0.1:18000"]       # PILOT 5
        return drop

    @unittest.skipUnless(sp.DEFAULT_DROP.is_dir(), "drop not present")
    def test_pilot_config_differs_from_the_drop_only_in_the_marked_edits(self):
        drop = self.drop_config()
        pilot = yaml.safe_load((SWE_DIR / "pilot" / "client-pilot.yaml").read_text())
        drop["datasets"][1]["accuracy_config"]["extras"]["num_instances"] = 10   # PILOT 4
        self.assertEqual(pilot, drop)

    @unittest.skipUnless(sp.DEFAULT_DROP.is_dir(), "drop not present")
    def test_num_instances_200_is_the_drops_task_list(self):
        """Test only: with --num-instances 200, PILOT 4 is gone: the drop's 200 tasks, 16/16 workers."""
        drop = self.drop_config()
        self.assertEqual(drop["datasets"][1]["accuracy_config"]["extras"]["num_instances"], 200)
        self.assertEqual(yaml.safe_load(sp.client_config(200)), drop)


class Pods(unittest.TestCase):
    node, run_id = "gke-node-a", "pilot-1"

    def all_pods(self):
        return {
            "smoke": sp.host_pod(sp.POD["smoke"], "smoke", self.node, "smoke-1", mode="smoke"),
            "host": sp.host_pod(sp.POD["host"], "host", self.node, self.run_id, mode="service"),
            "model": sp.model_pod(self.node),
            "client": sp.client_pod(self.node, self.run_id, sp.SERVICE_UID, sp.SERVICE_GID),
            "collect": sp.collect_pod(self.node, self.run_id, sp.SERVICE_UID, sp.SERVICE_GID),
        }

    def test_only_the_host_pods_are_privileged(self):
        for role, pod in self.all_pods().items():
            spec = pod["spec"]
            self.assertFalse(spec["automountServiceAccountToken"])
            self.assertEqual(spec["nodeSelector"], {"kubernetes.io/hostname": self.node})
            sc = spec["containers"][0]["securityContext"]
            if role in ("smoke", "host"):
                self.assertTrue(sc["privileged"])
            else:
                self.assertNotIn("privileged", sc)
                self.assertFalse(sc["allowPrivilegeEscalation"])
                self.assertEqual(sc["capabilities"], {"drop": ["ALL"]})
            if role in ("client", "collect"):
                self.assertTrue(sc["runAsNonRoot"])
                self.assertEqual((sc["runAsUser"], sc["runAsGroup"]), (61000, 61000))

    def test_secret_only_in_the_service_pod(self):
        pods = self.all_pods()
        for role, pod in pods.items():
            secrets = [v for v in pod["spec"]["volumes"] if "secret" in v]
            if role == "host":
                self.assertEqual(secrets[0]["secret"]["secretName"], sp.SECRET)
                self.assertEqual(secrets[0]["secret"]["defaultMode"], 0o400)
            else:
                self.assertEqual(secrets, [])
            self.assertNotIn("password", json.dumps(pod))

    def test_listeners_and_mounts(self):
        pods = self.all_pods()
        cmd = pods["model"]["spec"]["containers"][0]["command"]
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--port") + 1], "18000")
        self.assertIn("127.0.0.1:18000", (SWE_DIR / "pilot" / "client-pilot.yaml").read_text())
        client = pods["client"]["spec"]
        rw = [m["mountPath"] for m in client["containers"][0]["volumeMounts"] if not m.get("readOnly")]
        self.assertEqual(rw, [f"{sp.STATE}/runs/{self.run_id}"])
        collect = pods["collect"]["spec"]
        self.assertTrue(all(m.get("readOnly") for m in collect["containers"][0]["volumeMounts"]))
        self.assertNotIn("hostNetwork", collect)
        self.assertEqual(pods["host"]["spec"]["terminationGracePeriodSeconds"], 900)

    def test_client_script_is_valid_bash(self):
        p = subprocess.run(["bash", "-n"], input=sp.CLIENT_SCRIPT, text=True, capture_output=True,
                           check=False)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_model_starts_under_a_login_shell_like_sflow(self):
        # pilot-1: the image sets the TensorRT library path only in its shell init files.
        cmd = sp.model_pod(self.node)["spec"]["containers"][0]["command"]
        self.assertEqual(cmd[:5], ["bash", "-l", "-c", 'exec "$0" "$@"', "trtllm-serve"])
        self.assertEqual(cmd[5], sp.MODEL_DIR)
        self.assertFalse(any("$" in a for a in cmd[5:]))

    def test_run_ids(self):
        for ok in ("pilot-1", "smoke-20261006T013000Z", "a", "A.b_c-9"):
            self.assertTrue(sp.RUN_ID_RE.fullmatch(ok), ok)
        for bad in ("", "-x", "x-", "x/y", "a" * 64, "x y", "x;y"):
            self.assertFalse(sp.RUN_ID_RE.fullmatch(bad), bad)


class Redaction(unittest.TestCase):
    def test_masks_tokens_and_keeps_digests(self):
        r = sp.Redactor()
        r.add("s3cret-literal-value")
        service = "Ab_-" * 16
        digest = "a" * 64
        text = (f"{TOKEN} {service} Bearer abc.def --auth-token xyz password hunter2 "
                f"s3cret-literal-value {base64.b64encode(b's3cret-literal-value').decode()} "
                f"sha256:{digest}")
        out = r(text)
        for secret in (TOKEN, service, "abc.def", "xyz", "hunter2", "s3cret-literal-value"):
            self.assertNotIn(secret, out)
        self.assertIn(digest, out)


class Credentials(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "gitlab.netrc"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, mode=0o600):
        self.path.write_text(text)
        os.chmod(self.path, mode)

    def test_builds_both_machine_lines(self):
        self.write(f"machine registry.gitlab.com login someone password {TOKEN}\n")
        login, token, text = sp.build_credentials(self.path)
        self.assertEqual((login, token), ("someone", TOKEN))
        self.assertEqual(text.splitlines(), [
            f"machine registry.gitlab.com login someone password {TOKEN}",
            f"machine gitlab.com login someone password {TOKEN}"])

    def test_refusals_never_show_the_token(self):
        cases = [(f"machine registry.gitlab.com login someone password {TOKEN}\n", 0o644),
                 (f"machine example.com login someone password {TOKEN}\n", 0o600),
                 (f"machine gitlab.com login someone password short\n", 0o600),
                 (f"machine gitlab.com login some/one password {TOKEN}\n", 0o600),
                 (f"machine gitlab.com login someone password {TOKEN} bogus {TOKEN}\n", 0o600)]
        for text, mode in cases:
            self.write(text, mode)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                sp.build_credentials(self.path)
            self.assertNotIn(TOKEN, err.getvalue())

    def test_secret_command_sends_but_never_prints(self):
        self.write(f"machine gitlab.com login someone password {TOKEN}\n")
        kube, out = FakeKube(), io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(sp.cmd_secret(SimpleNamespace(netrc=str(self.path)), kube), 0)
        self.assertNotIn(TOKEN, out.getvalue())
        secret = [o for o in kube.applied if o["kind"] == "Secret"][0]
        creds = base64.b64decode(secret["data"][".credentials"]).decode()
        self.assertIn(f"machine registry.gitlab.com login someone password {TOKEN}", creds)
        self.assertEqual(kube.redact(json.dumps(secret)).count("<redacted>"), 1)


class Teardown(unittest.TestCase):
    def kube(self, client_phase, client_state):
        pods = {sp.POD["host"]: {"metadata": {"labels": {"kimi-swe/run-id": "pilot-1"}},
                                 "status": {"phase": "Running"}}}
        if client_phase:
            pods[sp.POD["client"]] = {"status": {"phase": client_phase,
                                                 "containerStatuses": [{"state": client_state}]}}
        return SimpleNamespace(get=lambda kind, name: pods.get(name),
                               delete_pod=lambda *a, **k: self.fail("host pod deleted"))

    def test_down_host_refuses_while_the_client_can_write_the_run_dir(self):
        for phase, state in (("Running", {"running": {}}), ("Pending", {"waiting": {}})):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                sp.down_host(self.kube(phase, state))
            self.assertIn("down client", err.getvalue())


class NumInstancesOverride(unittest.TestCase):
    """Test only: --num-instances N rewrites the PILOT 4 number of the client config, nothing else."""
    text = (SWE_DIR / "pilot" / "client-pilot.yaml").read_text()

    def test_default_and_the_files_own_number_leave_it_as_is(self):
        self.assertEqual(sp.client_config(), self.text)
        self.assertEqual(sp.client_config(10), self.text)

    def test_only_the_pilot_4_line_changes(self):
        new = sp.client_config(200)
        changed = [(a, b) for a, b in zip(self.text.splitlines(), new.splitlines(), strict=True) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0][1].lstrip().startswith("num_instances: 200 "), changed)
        self.assertTrue(changed[0][1].endswith("# PILOT 4, set by swe_pilot.py --num-instances (file: 10)"))
        want = yaml.safe_load(self.text)
        want["datasets"][1]["accuracy_config"]["extras"]["num_instances"] = 200
        self.assertEqual(yaml.safe_load(new), want)

    def test_refusals_fail_closed(self):
        two = self.text + "        num_instances: 5   # PILOT 4\n"
        for text, n, fragment in ((self.text, 0, "1 to 500"), (self.text, 501, "1 to 500"),
                                  (self.text, True, "1 to 500"), (self.text, "200", "1 to 500"),
                                  (self.text.replace("# PILOT 4", "# PILOT 4b"), 200, "found 0"),
                                  (self.text.replace("num_instances: 10", "num_instances: ten"), 200,
                                   "found 0"),
                                  (two, 200, "found 2")):
            with self.subTest(n=n, fragment=fragment):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    sp.set_num_instances(text, n)
                self.assertIn(fragment, err.getvalue())

    def test_parser(self):
        self.assertIsNone(sp.parser().parse_args(["configmaps"]).num_instances)
        self.assertEqual(sp.parser().parse_args(["--num-instances", "200", "configmaps"]).num_instances, 200)
        for bad in ("0", "501", "-1", "abc", "1e2", "200 ", "2000", "\uff12\uff10\uff10", ""):
            with self.subTest(bad=bad):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    sp.parser().parse_args(["--num-instances", bad, "configmaps"])
                self.assertEqual(cm.exception.code, 2)
                self.assertIn("1 to 500", err.getvalue())

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_only_the_client_configmap_changes_and_client_checks_it(self):
        base = {cm["metadata"]["name"]: cm for cm in sp.desired_configmaps(args())}
        big = {cm["metadata"]["name"]: cm for cm in sp.desired_configmaps(args(num_instances=200))}
        self.assertEqual(set(base), set(big))
        for name in base:
            same = base[name]["metadata"]["annotations"] == big[name]["metadata"]["annotations"]
            self.assertEqual(same, name != sp.CM_CLIENT, name)
        self.assertEqual(big[sp.CM_CLIENT]["data"]["client-pilot.yaml"], sp.client_config(200))
        live = {("configmap", sp.CM_CLIENT): big[sp.CM_CLIENT]}
        kube = SimpleNamespace(get=lambda kind, name: live.get((kind, name)))
        sp.require_configmaps(kube, args(num_instances=200), (sp.CM_CLIENT,))      # same N: ok
        for other in (args(), args(num_instances=10), args(num_instances=199)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                sp.require_configmaps(kube, other, (sp.CM_CLIENT,))
            self.assertIn("same --num-instances", err.getvalue())

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_configmaps_command_applies_and_reports_it(self):
        kube, out = FakeKube(), io.StringIO()
        argv = ["--oracle-dir", "", "--num-instances", "200", "configmaps"]
        with contextlib.redirect_stdout(out):
            self.assertEqual(sp.cmd_configmaps(sp.parser().parse_args(argv), kube), 0)
        (client,) = [o for o in kube.applied if o["metadata"]["name"] == sp.CM_CLIENT]
        self.assertEqual(client["data"]["client-pilot.yaml"], sp.client_config(200))
        self.assertIn("swe_bench num_instances 200 from --num-instances", out.getvalue())


class V102ClientImageAndPerRunPods(unittest.TestCase):
    node, run_id, alloc = "gke-node-a", "run-1", "a1b2c3"

    def test_client_image_is_v103_d3c57f2(self):
        want = ("ghcr.io/mlcommons/endpoints@sha256:"
                "1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d")
        self.assertEqual(sp.CLIENT_IMAGE, want)
        for pod in (
            sp.host_pod(sp.POD["smoke"], "smoke", self.node, "smoke-1", mode="smoke"),
            sp.host_pod(sp.POD["host"], "host", self.node, self.run_id, mode="service"),
            sp.client_pod(self.node, self.run_id, sp.SERVICE_UID, sp.SERVICE_GID),
            sp.collect_pod(self.node, self.run_id, sp.SERVICE_UID, sp.SERVICE_GID),
        ):
            self.assertEqual(pod["spec"]["containers"][0]["image"], want)
            self.assertNotIn("6cf516d0", json.dumps(pod))

    def test_per_allocation_host_and_collect_pod_names(self):
        hname = sp.run_pod_name("host", self.alloc)
        cname = sp.run_pod_name("collect", self.alloc)
        self.assertEqual(hname, "kimi-swe-host-a1b2c3")
        self.assertEqual(cname, "kimi-swe-collect-a1b2c3")
        self.assertEqual(sp.down_hint(hname), "swe_pilot.py down host --alloc a1b2c3")
        self.assertEqual(sp.down_hint(sp.POD["host"]), "swe_pilot.py down host")
        hpod = sp.host_pod(hname, "host", self.node, self.run_id, mode="service",
                           handoff=True, reports_dir=sp.REPORTS_DIR, alloc=self.alloc)
        self.assertEqual(hpod["metadata"]["name"], hname)
        self.assertEqual(hpod["metadata"]["labels"][sp.ALLOC_LABEL], self.alloc)
        cpod = sp.collect_pod(self.node, self.run_id, sp.SERVICE_UID, sp.SERVICE_GID, name=cname)
        self.assertEqual(cpod["metadata"]["name"], cname)
        for bad in ("", "-bad", "BAD", "a/b", "a" * 41):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.run_pod_name("host", bad)

    def test_down_with_alloc_and_check_node_with_alloc(self):
        deleted = []
        pods = {
            "kimi-swe-host-a1b2c3": {
                "metadata": {"name": "kimi-swe-host-a1b2c3",
                             "labels": {"kimi-swe/run-id": self.run_id, sp.ALLOC_LABEL: self.alloc}},
                "status": {"phase": "Succeeded", "containerStatuses": [{"state": {"terminated": {}}}]},
            }
        }
        kube = SimpleNamespace(
            dry_run=False,
            get=lambda kind, name, namespaced=True: (
                {"metadata": {"name": self.node},
                 "status": {"conditions": [{"type": "Ready", "status": "True"}],
                            "allocatable": {"nvidia.com/gpu": "4"}}}
                if kind == "node" else pods.get(name)
            ),
            logs=lambda name, tail=None: "service run finished: ok\n",
            delete_pod=lambda name, timeout=180: deleted.append(name),
            run=lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""),
        )
        orig_follow = sp.follow_logs
        sp.follow_logs = lambda *a, **k: (["service run finished: ok\n"], "exited")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    sp.cmd_down(SimpleNamespace(target="host", alloc=self.alloc), kube), 0)
        finally:
            sp.follow_logs = orig_follow
        self.assertEqual(deleted, ["kimi-swe-host-a1b2c3"])
        for bad_target in ("model", "smoke", "client", "secret", "all"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.cmd_down(SimpleNamespace(target=bad_target, alloc=self.alloc), kube)


class OfflineCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gen = self.root / "generate_point_recipes.py"
        self.write_generator()

    def tearDown(self):
        self.tmp.cleanup()

    def write_generator(self, extra: str = ""):
        expected = {
            12: (613, 1, 8, 5, 8, 48, 12, 2, 2),
            13: (613, 1, 8, 3, 8, 32, 8, 2, 2),
            16: (613, 1, 8, 3, 8, 32, 8, 2, 2),
            24: (613, 1, 8, 3, 8, 32, 8, 2, 2),
            32: (613, 1, 8, 3, 8, 32, 8, 2, 2),
            48: (1226, 1, 8, 3, 8, 32, 8, 2, 2),
            64: (1839, 1, 8, 3, 8, 32, 8, 2, 2),
            80: (1839, 1, 8, 3, 8, 32, 8, 2, 2),
            192: (3065, 16, 1, 2, 16, 48, 12, 4, 4),
            288: (4904, 16, 1, 2, 16, 48, 12, 4, 4),
            384: (4291, 16, 1, 2, 16, 48, 12, 4, 4),
            448: (4291, 16, 1, 1, 16, 32, 8, 4, 4),
            512: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
            576: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
            640: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
            1024: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
        }
        contract = {
            "modes": ("acc", "both"),
            "handoff_dir": sp.HANDOFF_DIR,
            "connection_file": sp.CONNECTION_FILE,
            "wait_s": sp.SWE_WAIT_S,
            "reports_dir": sp.REPORTS_DIR,
            "mode_marker": sp.MODE_MARKER,
            "url_prefix": sp.URL_PREFIX,
            "client_image": sp.CLIENT_IMAGE,
            "pull_secret_subdir": sp.PULL_SECRET_SUBDIR,
            "client_task": sp.CLIENT_TASK,
            "frontend_task": sp.FRONTEND_TASK,
            "k8s_namespace": sp.SFLOW_NS,
            "client_node": {c: tup[8] for c, tup in expected.items()},
            "frontend_node": {c: tup[7] for c, tup in expected.items()},
        }
        self.gen.write_text(
            f"CLIENT_IMAGE = {sp.CLIENT_IMAGE!r}\n"
            f"REPORTS_DIR = {sp.REPORTS_DIR!r}\n"
            f"PULL_SECRET_SUBDIR = {sp.PULL_SECRET_SUBDIR!r}\n"
            f"EXPECTED = {expected!r}\n"
            f"SWE_CONTRACT = {contract!r}\n"
            + extra
        )

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_check_never_invokes_kubectl_and_validates_contract(self):
        shim_dir = self.root / "bin"
        shim_dir.mkdir()
        marker = self.root / "kubectl_called"
        kubectl = shim_dir / "kubectl"
        kubectl.write_text(f"#!/bin/sh\necho called > {marker}\nexit 99\n")
        kubectl.chmod(0o755)
        old_path = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = f"{shim_dir}:{old_path}"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = sp.main(["--check", "--generator", str(self.gen),
                              "--recipes-root", ""])
            self.assertEqual(rc, 0)
            self.assertIn("CHECK OK", out.getvalue())
            self.assertFalse(marker.exists(), "kubectl was invoked during --check")
        finally:
            os.environ["PATH"] = old_path

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_check_fails_closed_on_contract_drift(self):
        for snippet, needle in (
            ("del SWE_CONTRACT\n", "SWE_CONTRACT missing"),
            ("SWE_CONTRACT['handoff_dir'] = '/tmp/wrong'\n", "SWE_CONTRACT['handoff_dir']"),
            ("SWE_CONTRACT['wait_s'] = 60\n", "SWE_CONTRACT['wait_s']"),
            ("SWE_CONTRACT['client_node'][24] = 0\n", "SWE_CONTRACT client_node"),
            ("CLIENT_IMAGE = 'ghcr.io/mlcommons/endpoints@sha256:' + '0'*64\n", "generator CLIENT_IMAGE"),
        ):
            with self.subTest(needle=needle):
                self.write_generator(snippet)
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    sp.main(["--check", "--generator", str(self.gen)])
                self.assertIn(needle, err.getvalue())


if __name__ == "__main__":
    unittest.main()
