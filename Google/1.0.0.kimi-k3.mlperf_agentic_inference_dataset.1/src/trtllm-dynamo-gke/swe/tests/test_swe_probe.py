"""Unit tests for swe_probe.py root-side helpers (conn, snapshot, leakscan, postcheck-report). Standard library only."""
import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SWE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swe_probe", SWE_DIR / "swe_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

TOKEN = "Zq3_service-token-" + "x" * 46                      # 64 chars, like token_urlsafe(48)
REGISTRY = "<gitlab-pat-prefix>-EXAMPLEexample0123456789"


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = probe.main(list(argv))
        return rc, out.getvalue()

    def test_snapshot_writes_a_private_copy(self):
        conn, dst = self.root / "conn.json", self.root / "token"
        conn.write_text(json.dumps({"url": "http://127.0.0.1:1", "token": TOKEN}))
        rc, out = self.run_main("snapshot", "--connection", str(conn), "--out", str(dst))
        self.assertEqual(rc, 0)
        self.assertEqual(dst.read_text(), TOKEN)
        self.assertEqual(stat.S_IMODE(dst.stat().st_mode), 0o600)
        self.assertNotIn(TOKEN, out)
        conn.write_text("{}")
        self.assertEqual(self.run_main("snapshot", "--connection", str(conn), "--out", str(dst))[0], 1)

    def make_tree(self, leak: bool):
        data = self.root / "run"
        (data / "report").mkdir(parents=True)
        (data / "swe_service" / "enroot-cache").mkdir(parents=True)
        (data / "report" / "config.yaml").write_text("auth_token: <redacted>\n")
        (data / "swe_service" / "enroot-cache" / "layer").write_text(TOKEN)   # skipped dir
        if leak:
            (data / "report" / "log.txt").write_text(f"a {TOKEN} b {TOKEN}\n")
            (data / "swe_service" / "service.log").write_text(f"pw {REGISTRY}\n")
        secret = self.root / "secret"
        secret.write_text(TOKEN)
        netrc = self.root / "netrc"
        netrc.write_text(f"machine registry.gitlab.com login u password {REGISTRY}\n"
                         f"machine gitlab.com login u password {REGISTRY}\n")
        return data, secret, netrc

    def test_leakscan_counts_hits_without_printing_secrets(self):
        data, secret, netrc = self.make_tree(leak=True)
        report = self.root / "leaks.json"
        rc, out = self.run_main("leakscan", "--root", str(data), "--secret-file", str(secret),
                                "--netrc", str(netrc), "--report", str(report))
        self.assertEqual(rc, 1)
        result = json.loads(report.read_text())
        self.assertEqual(result["hits"], {"service_token": 2, "registry_token_1": 1})
        self.assertEqual(len(result["files_with_hits"]), 2)
        for text in (out, report.read_text()):
            self.assertNotIn(TOKEN, text)
            self.assertNotIn(REGISTRY, text)

    def test_leakscan_clean_tree_and_no_needles(self):
        data, secret, netrc = self.make_tree(leak=False)
        report = self.root / "leaks.json"
        rc, _ = self.run_main("leakscan", "--root", str(data), "--secret-file", str(secret),
                              "--netrc", str(netrc), "--report", str(report))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(report.read_text())["files_with_hits"], [])
        rc, _ = self.run_main("leakscan", "--root", str(data), "--report", str(report))
        self.assertEqual(rc, 2)

    def test_netrc_passwords_parse(self):
        _, _, netrc = self.make_tree(leak=False)
        self.assertEqual(probe._netrc_passwords(str(netrc)), [REGISTRY])
        self.assertEqual(probe._netrc_passwords(str(self.root / "missing")), [])

    # ---- root reads of files the service user can replace
    def test_read_small_refuses_symlink_fifo_and_oversize(self):
        target = self.root / "target"
        target.write_text("{}")
        link = self.root / "link"
        link.symlink_to(target)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)                 # no writer: a blocking open would hang the test
        big = self.root / "big"
        big.write_bytes(b"x" * 2048)
        self.assertEqual(probe.read_small(str(target)), b"{}")
        for path in (link, fifo):
            with self.assertRaises(OSError):
                probe.read_small(str(path))
        with self.assertRaises(OSError):
            probe.read_small(str(big), limit=1024)

    def test_conn_accepts_only_the_drop_loopback_url(self):
        conn = self.root / "conn.json"
        conn.write_text(json.dumps({"url": "http://127.0.0.1:43211", "token": TOKEN}))
        rc, out = self.run_main("conn", "--connection", str(conn), "--print-url")
        self.assertEqual((rc, out), (0, "http://127.0.0.1:43211\n"))
        self.assertEqual(self.run_main("conn", "--connection", str(conn))[0], 0)
        for url in ("http://<ip>:80", "http://127.0.0.1:1/x", "<workdir>/etc/passwd",
                    "http://127.0.0.1:1 -o /etc/x", 5):
            conn.write_text(json.dumps({"url": url, "token": TOKEN}))
            rc, out = self.run_main("conn", "--connection", str(conn), "--print-url")
            self.assertEqual((rc, out), (1, ""), url)
        for text in ("", "[]", "not json", "{}"):
            conn.write_text(text)
            self.assertEqual(self.run_main("conn", "--connection", str(conn))[0], 1, text)

    def test_conn_expect_reset(self):
        conn = self.root / "conn.json"
        conn.write_text("{}")
        self.assertEqual(self.run_main("conn", "--connection", str(conn), "--expect-reset")[0], 0)
        conn.write_text(json.dumps({"url": "http://127.0.0.1:1", "token": TOKEN}))
        self.assertEqual(self.run_main("conn", "--connection", str(conn), "--expect-reset")[0], 1)
        conn.write_text("")
        self.assertEqual(self.run_main("conn", "--connection", str(conn), "--expect-reset")[0], 1)
        reset = self.root / "reset.json"
        reset.write_text("{}")
        conn.unlink()
        conn.symlink_to(reset)          # a swapped-in symlink is not a reset connection file
        self.assertEqual(self.run_main("conn", "--connection", str(conn), "--expect-reset")[0], 1)

    def test_snapshot_ignores_a_symlinked_connection_file(self):
        real, conn, dst = self.root / "real.json", self.root / "conn.json", self.root / "token"
        real.write_text(json.dumps({"url": "http://127.0.0.1:1", "token": TOKEN}))
        conn.symlink_to(real)
        rc, _ = self.run_main("snapshot", "--connection", str(conn), "--out", str(dst))
        self.assertEqual(rc, 1)
        self.assertFalse(dst.exists())

    def test_write_report_replaces_a_planted_symlink(self):
        victim = self.root / "victim"
        victim.write_text("keep")
        report = self.root / "postcheck-leaks.json"
        report.symlink_to(victim)
        probe.write_report(str(report), "new\n")
        self.assertEqual(victim.read_text(), "keep")
        self.assertFalse(report.is_symlink())
        self.assertEqual(report.read_text(), "new\n")
        self.assertEqual(stat.S_IMODE(report.stat().st_mode), 0o644)

    def test_postcheck_report_fields(self):
        out = self.root / "postcheck.json"
        rc, _ = self.run_main("postcheck-report", "--out", str(out), "--service-rc", "143",
                              "--fail", "0", "--retained", "", "--leftover", "3")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.read_text()), {
            "service_rc": "143", "ok": True, "scratch_retained": [], "leftover_processes": 3})
        rc, _ = self.run_main("postcheck-report", "--out", str(out), "--service-rc", "none",
                              "--fail", "1", "--retained", "/tmp/kimi-swe-a\n/tmp/kimi-swe-b")
        result = json.loads(out.read_text())
        self.assertEqual((result["ok"], result["leftover_processes"]), (False, None))
        self.assertEqual(result["scratch_retained"], ["/tmp/kimi-swe-a", "/tmp/kimi-swe-b"])
        with self.assertRaises(SystemExit):
            self.run_main("postcheck-report", "--out", str(out), "--service-rc", "0",
                          "--fail", "0", "--leftover", "3; rm -rf /")

    def test_leakscan_skips_symlinks_and_fifos(self):
        data, secret, netrc = self.make_tree(leak=False)
        outside = self.root / "outside.txt"
        outside.write_text(TOKEN)
        (data / "report" / "link.txt").symlink_to(outside)
        os.mkfifo(data / "report" / "pipe")
        report = self.root / "leaks.json"
        rc, _ = self.run_main("leakscan", "--root", str(data), "--secret-file", str(secret),
                              "--netrc", str(netrc), "--report", str(report))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(report.read_text())["files_with_hits"], [])

    def test_exc_chain_shows_the_wrapped_srun_output(self):
        def start():
            try:
                raise RuntimeError("Pyxis infrastructure failure (srun return code 1)\nenroot: boom")
            except RuntimeError as inner:
                raise ValueError("failed to start Pyxis container for img") from inner
        with self.assertRaises(ValueError) as ctx:
            start()
        text = probe.exc_chain(ctx.exception)
        self.assertTrue(text.startswith("ValueError: failed to start Pyxis container"))
        self.assertIn("<- caused by RuntimeError: Pyxis infrastructure failure", text)
        self.assertIn("enroot: boom", text)
        self.assertEqual(probe.exc_chain(None), "")


if __name__ == "__main__":
    unittest.main()
