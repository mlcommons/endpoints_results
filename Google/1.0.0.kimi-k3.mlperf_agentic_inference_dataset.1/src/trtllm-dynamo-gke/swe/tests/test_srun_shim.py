"""Unit tests for the srun shim (swe/srun). Standard library only.

Run: python3 -m unittest discover -s examples/self_contained/kubernetes/v1.0.2/kimi-k3/swe/tests
"""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SWE_DIR = Path(__file__).resolve().parents[1]
SHIM = SWE_DIR / "srun"


def load_shim():
    loader = importlib.machinery.SourceFileLoader("srun_shim", str(SHIM))
    spec = importlib.util.spec_from_loader("srun_shim", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class SrunShimTest(unittest.TestCase):
    def setUp(self):
        self.shim = load_shim()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.conf = root / "shim.json"
        self.conf.write_text(json.dumps({"jobid": "1", "node": "node-a"}))
        self.scratch_root = root / "tmp"
        self.step = "kimi-swe-abc123"
        self.chdir = self.scratch_root / self.step / "tmp"
        self.chdir.mkdir(parents=True)
        (self.scratch_root / "outside").mkdir()
        self.shim.CONF_PATH = str(self.conf)
        self.shim.SCRATCH_ROOT = str(self.scratch_root)
        self.env = mock.patch.dict(os.environ, {"SWE_STEP_NAME": self.step})
        self.env.start()
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        self.env.stop()
        self.tmp.cleanup()

    def argv(self, *cmd, jobid="1", node="node-a", chdir=None, name=None, extra=()):
        return ["srun", "--overlap", f"--jobid={jobid}", "-N1", "-n1", f"--nodelist={node}",
                f"--chdir={chdir or self.chdir}", f"--job-name={name or self.step}", *extra, *cmd]

    def run_shim(self, argv):
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.shim.os, "execvp") as execvp, \
                mock.patch("sys.stderr"):
            try:
                self.shim.main()
            except SystemExit as exc:
                return exc.code, execvp
        return 0, execvp

    def assert_exec(self, *cmd):
        code, execvp = self.run_shim(self.argv(*cmd))
        self.assertEqual(code, 0)
        execvp.assert_called_once_with(cmd[0], list(cmd))
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.chdir))

    def assert_rejected(self, argv):
        code, execvp = self.run_shim(argv)
        self.assertEqual(code, 2)
        execvp.assert_not_called()

    def test_accepts_the_three_command_forms(self):
        self.assert_exec("bash", "--noprofile", "--norc", "-c", "set -e\necho hi")
        os.chdir(self.cwd)
        self.assert_exec("enroot", "list", "-f")
        os.chdir(self.cwd)
        self.assert_exec("enroot", "remove", "-f", "pyxis_mswe_x_1234abcd")

    def test_rejects_pyxis_flags(self):
        self.assert_rejected(self.argv("true", extra=("--container-image=x",)))
        self.assert_rejected(self.argv("true", extra=("--container-name=x",)))

    def test_rejects_other_srun_options(self):
        self.assert_rejected(self.argv("true", extra=("--exclusive",)))
        argv = self.argv("bash", "--noprofile", "--norc", "-c", "true")
        argv.insert(2, "--jobid=1")
        self.assert_rejected(argv)                                     # duplicate option
        argv = self.argv("bash", "--noprofile", "--norc", "-c", "true")
        argv.remove("--overlap")
        self.assert_rejected(argv)                                     # missing flag

    def test_rejects_wrong_allocation(self):
        cmd = ("enroot", "list", "-f")
        self.assert_rejected(self.argv(*cmd, jobid="2"))
        self.assert_rejected(self.argv(*cmd, node="node-b"))
        self.assert_rejected(self.argv(*cmd, name="kimi-swe-other"))
        with mock.patch.dict(os.environ, {"SWE_STEP_NAME": ""}):
            self.assert_rejected(self.argv(*cmd, name=""))

    def test_rejects_chdir_outside_the_scratch(self):
        cmd = ("enroot", "list", "-f")
        self.assert_rejected(self.argv(*cmd, chdir=self.scratch_root / "outside"))
        self.assert_rejected(self.argv(*cmd, chdir=self.scratch_root))
        self.assert_rejected(self.argv(*cmd, chdir=self.scratch_root / "kimi-swe-"))
        self.assert_rejected(self.argv(*cmd, chdir=f"{self.chdir}/../../outside"))
        link = self.scratch_root / self.step / "escape"
        link.symlink_to(self.scratch_root / "outside")
        self.assert_rejected(self.argv(*cmd, chdir=link))

    def test_rejects_other_commands(self):
        for cmd in (("bash", "-c", "true"), ("bash", "--noprofile", "--norc", "-c"),
                    ("enroot", "remove", "-f", "notpyxis"), ("enroot", "start", "x"),
                    ("python3", "-c", "print(1)"), ()):
            self.assert_rejected(self.argv(*cmd))

    def test_exec_failure_is_127(self):
        argv = self.argv("enroot", "list", "-f")
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.shim.os, "execvp", side_effect=FileNotFoundError(2, "x")), \
                mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                self.shim.main()
        self.assertEqual(ctx.exception.code, 127)

    def test_exit_status_and_stdout_pass_through(self):
        """Real exec in a child process: the step's exit code and stdout are unchanged."""
        driver = (
            "import importlib.machinery, importlib.util, sys\n"
            f"l = importlib.machinery.SourceFileLoader('s', {str(SHIM)!r})\n"
            "s = importlib.util.spec_from_loader('s', l); m = importlib.util.module_from_spec(s)\n"
            "l.exec_module(m)\n"
            f"m.CONF_PATH = {str(self.conf)!r}; m.SCRATCH_ROOT = {str(self.scratch_root)!r}\n"
            "sys.argv = sys.argv[1:]\n"
            "m.main()\n")
        argv = self.argv("bash", "--noprofile", "--norc", "-c", "echo STEP_OUT; exit 7")
        p = subprocess.run([sys.executable, "-c", driver, *argv], capture_output=True, text=True,
                           env={**os.environ, "SWE_STEP_NAME": self.step}, timeout=30, check=False)
        self.assertEqual(p.returncode, 7)
        self.assertEqual(p.stdout, "STEP_OUT\n")
        self.assertEqual(p.stderr, "")


if __name__ == "__main__":
    unittest.main()
