"""Tests for the test-only SWE-bench oracle (pilot/oracle/) and its `model --oracle` wiring.

No cluster and no network: every server listens on 127.0.0.1, the data fetcher gets a fake
fetch, and bash and git run in temporary directories only.
"""
import ast
import base64
import contextlib
import gzip
import hashlib
import http.client
import importlib.util
import io
import json
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.dont_write_bytecode = True                  # no __pycache__ next to the modules loaded below

SWE_DIR = Path(__file__).resolve().parents[1]
ORACLE_DIR = SWE_DIR / "pilot" / "oracle"
SERVER = ORACLE_DIR / "oracle_server.py"
REAL_DATA, REAL_MANIFEST = ORACLE_DIR / "oracle-data.json.gz", ORACLE_DIR / "oracle-manifest.json"
PILOT2 = Path.home() / ".cache" / "kimi-swe-pilot" / "collected" / "pilot-2"     # read only


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


od = load("oracle_data", ORACLE_DIR / "oracle_data.py")
srv = od.srv                                    # the oracle_server module that oracle_data loaded
sp = load("swe_pilot_oracle_wiring", SWE_DIR / "pilot" / "swe_pilot.py")

HAVE_INPUTS = sp.DEFAULT_DROP.is_dir() and all((SWE_DIR / "artifacts" / n).is_file() for n in sp.ARTIFACTS)
HAVE_DATA = REAL_DATA.is_file() and REAL_MANIFEST.is_file()
TRAJS = (sorted(PILOT2.glob("swe_service/artifacts/*/swe_bench_output/*/*.traj.json"))
         if PILOT2.is_dir() else [])

# mini-swe-agent 2.3.0 BASH_TOOL, sent with every LitellmModel query.
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "required": ["command"], "properties": {
        "command": {"type": "string", "description": "The bash command to execute"}}}}}
SYSTEM = {"role": "system", "content": "You are a helpful assistant that can interact with a "
                                       "computer shell to solve programming tasks."}
PROBLEM_1 = ("Widget.resize() loses the aspect ratio\r\n\r\nResizing a widget twice makes its aspect "
             "ratio drift:\r\n\r\n```python\r\n>>> w = Widget(4, 3)\r\n>>> w.resize(2); w.resize(0.5)\r\n"
             ">>> w.ratio\r\n1.3334\r\n```\r\n")
PROBLEM_2 = ("Gadget.parse() rejects empty input\r\n\r\nParsing an empty string raises IndexError "
             "instead of returning an empty Gadget.\r\n")
PATCH_1 = ("diff --git a/widget.py b/widget.py\n--- a/widget.py\n+++ b/widget.py\n@@ -1 +1 @@\n"
           "-ratio = 1.3334\n+ratio = 4 / 3\n")
PATCH_2 = ("diff --git a/gadget.py b/gadget.py\n--- a/gadget.py\n+++ b/gadget.py\n@@ -1 +1 @@\n"
           "-def parse(s): return s[0]\n+def parse(s): return s[:1]\n")
TASKS = [srv.Task("octo__widget-101", PROBLEM_1, PATCH_1), srv.Task("octo__gadget-202", PROBLEM_2, PATCH_2)]
ROWS = [t._asdict() for t in TASKS]
SECRETISH = "sk-oracle-test-not-a-real-key"      # must never reach a log line

LOGS: list[str] = []
_saved_log = None


def setUpModule():
    global _saved_log
    _saved_log = srv.log
    srv.log = lambda msg, *, err=False: LOGS.append(msg)        # the server's log lines, captured


def tearDownModule():
    srv.log = _saved_log


def logged(fragment: str, timeout: float = 3.0) -> bool:
    """A response can reach the client just before its log line is written: wait a little."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(fragment in line for line in list(LOGS)):
            return True
        time.sleep(0.01)
    return False


def user_message(problem: str) -> dict:
    """The first user message: mini-swe-agent's instance template around the problem statement."""
    return {"role": "user", "content": "<pr_description>\nConsider the following PR description:\n"
            f"{problem}\n</pr_description>\n\n<instructions>\n...\nSubmit: echo "
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n</instructions>"}


def chat(messages, **extra) -> dict:
    return {"model": "openai/swe-oracle-gold-patch", "messages": messages, "tools": [BASH_TOOL], **extra}


def command_of(payload: dict) -> str:
    call = payload["choices"][0]["message"]["tool_calls"][0]
    return json.loads(call["function"]["arguments"])["command"]


def next_turn(messages: list, payload: dict) -> list:
    """The agent's history after one step: the assistant message, then its tool observation."""
    message = payload["choices"][0]["message"]
    return [*messages, message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"],
                                 "content": "<returncode>0</returncode>\n<output>\n</output>"}]


# ----------------------------------------------------------------------------- bash and git
GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}


def bash(command: str, cwd: Path, env=None) -> subprocess.CompletedProcess:
    """One agent step as swebench_service PyxisEnvironment.execute sees it: `bash -c` with the
    command as one argument, stderr into stdout, read in text mode (CR becomes LF)."""
    return subprocess.run(["bash", "-c", command], cwd=cwd, env=env, text=True, encoding="utf-8",
                          errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)


def submission(p: subprocess.CompletedProcess):
    """PyxisEnvironment.execute (= mini-swe-agent LocalEnvironment._check_finished): the output
    after a first line equal to the sentinel, if the command succeeded; else None."""
    lines = p.stdout.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == srv.SENTINEL and p.returncode == 0:
        return "".join(lines[1:])
    return None


class Sandbox:
    """A throwaway git repository standing in for /testbed."""

    def __init__(self, root: Path, files: dict):
        root.mkdir(parents=True)
        self.root = root
        self.env = {**os.environ, **GIT_ENV, "HOME": str(root)}
        self.git("-c", "init.defaultBranch=main", "init", "-q")
        self.write(files)
        self.git("add", "-A")
        self.git("-c", "user.name=oracle-test", "-c", "user.email=<email>",
                 "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")

    def write(self, files: dict) -> None:
        for name, text in files.items():
            (self.root / name).write_bytes(text.encode("utf-8"))

    def git(self, *args: str) -> str:
        p = subprocess.run(["git", "-C", str(self.root), *args], env=self.env, capture_output=True,
                           check=False)
        if p.returncode != 0:
            raise AssertionError(f"git {args[-1]} failed: {p.stderr.decode(errors='replace')[-500:]}")
        return p.stdout.decode("utf-8")

    def patch_for(self, changes: dict) -> str:
        """`git diff` for `changes` against the commit; the work tree is restored afterwards."""
        self.write(changes)
        diff = self.git("diff")
        self.git("checkout", "-q", "--", ".")
        return diff

    def step(self, command: str) -> subprocess.CompletedProcess:
        return bash(command, self.root, self.env)


class Plans(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oracle-plan-"))
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_small_patch_write_check_submit(self):
        box = Sandbox(self.tmp / "testbed", {"widget.py": "ratio = 1.3334\nname = 'naive'\n"})
        patch = box.patch_for({"widget.py": "ratio = 4 / 3\nname = 'na\u00efve caf\u00e9 \u2192 \u2713'\n"})
        steps = srv.plan(patch)
        self.assertEqual([s.command for s in steps[1:]], [srv.CHECK, srv.SUBMIT])
        p = box.step(steps[0].command)
        self.assertEqual((p.returncode, p.stdout), (0, ""))
        self.assertEqual((box.root / "patch.txt").read_bytes(), patch.encode("utf-8"))
        p = box.step(steps[1].command)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIsNone(submission(p))                       # only the submit step submits
        self.assertEqual(submission(box.step(steps[2].command)), patch)

    def test_large_patch_goes_through_patch_b64_in_parts(self):
        box = Sandbox(self.tmp / "testbed", {
            "big.txt": "".join(f"line {i:05d}: the quick brown fox\n" for i in range(1500))})
        patch = box.patch_for({"big.txt": "".join(f"line {i:05d}: the lazy dog sleeps\n"
                                                  for i in range(1500))})
        b64 = len(base64.b64encode(patch.encode()))
        self.assertGreater(b64, 2 * srv.CHUNK)                 # three parts at least
        steps = srv.plan(patch)
        self.assertEqual(len(steps), -(-b64 // srv.CHUNK) + 3)
        self.assertEqual([s.command for s in steps[-3:]],
                         ["base64 -d patch.b64 > patch.txt && rm -f patch.b64", srv.CHECK, srv.SUBMIT])
        (box.root / "patch.b64").write_text("STALE LEFTOVER\n")   # the first part must replace it
        for step in steps[:-1]:
            self.assertLess(len(step.command.encode()) + 1, 128 * 1024)    # MAX_ARG_STRLEN
            p = box.step(step.command)
            self.assertEqual(p.returncode, 0, p.stdout[-500:])
        self.assertFalse((box.root / "patch.b64").exists())
        self.assertEqual((box.root / "patch.txt").read_bytes(), patch.encode())
        self.assertEqual(submission(box.step(steps[-1].command)), patch)

    def test_commands_quote_only_base64(self):
        allowed = re.compile(
            r"printf '%s' '[A-Za-z0-9+/=]+' \| base64 -d > patch\.txt|printf '%s' '[A-Za-z0-9+/=]+' "
            r">>? patch\.b64|base64 -d patch\.b64 > patch\.txt && rm -f patch\.b64|"
            + re.escape(srv.CHECK) + "|" + re.escape(srv.SUBMIT))
        for patch in (PATCH_1, "diff --git a/x b/x\n+'$(touch pwned)'`id`\n" * 3000):
            for step in srv.plan(patch):
                self.assertRegex(step.command, rf"\A(?:{allowed.pattern})\Z")

    def test_cr_in_a_patch_is_flagged_and_changes_the_submission(self):
        patch = ("diff --git a/w.bat b/w.bat\n--- a/w.bat\n+++ b/w.bat\n@@ -1 +1 @@\n"
                 "-echo off\r\n+echo on\r\n")
        oracle = srv.Oracle([srv.Task("octo__dos-1", PROBLEM_1, patch)])
        self.assertEqual(len(oracle.warnings), 1)
        self.assertTrue(oracle.warnings[0].startswith("octo__dos-1: "))
        steps = oracle.plan_for("octo__dos-1")
        self.assertEqual(bash(steps[0].command, self.tmp).returncode, 0)
        self.assertEqual((self.tmp / "patch.txt").read_bytes(), patch.encode())    # exact on disk
        got = submission(bash(steps[-1].command, self.tmp))
        self.assertEqual(got, patch.replace("\r\n", "\n"))      # text mode: the caveat
        self.assertNotEqual(got, patch)


class Replies(unittest.TestCase):
    oracle = srv.Oracle(TASKS)

    def answer(self, request):
        return srv.answer(self.oracle, request, "swe-oracle-gold-patch")

    def test_write_check_submit_then_submit_again(self):
        messages, steps = [SYSTEM, user_message(PROBLEM_1)], srv.plan(PATCH_1)
        for k in range(5):
            reply = self.answer(chat(messages))
            self.assertEqual((reply.status, reply.instance, reply.step), (200, "octo__widget-101", k))
            self.assertEqual(command_of(reply.payload), steps[min(k, 2)].command)
            content = reply.payload["choices"][0]["message"]["content"]
            self.assertEqual("(again)" in content, k > 2)
            messages = next_turn(messages, reply.payload)

    def test_reply_is_a_chat_completion_with_one_bash_call(self):
        body = self.answer(chat([SYSTEM, user_message(PROBLEM_2)])).payload
        self.assertEqual((body["object"], body["model"]), ("chat.completion", "swe-oracle-gold-patch"))
        self.assertRegex(body["id"], r"\Achatcmpl-[0-9a-f]{32}\Z")
        (choice,) = body["choices"]
        self.assertEqual((choice["index"], choice["finish_reason"]), (0, "tool_calls"))
        message = choice["message"]
        self.assertEqual(message["role"], "assistant")
        self.assertTrue(message["content"].startswith("THOUGHT: SWE-bench oracle"))
        (call,) = message["tool_calls"]
        self.assertRegex(call["id"], r"\Acall_[0-9a-f]{24}\Z")
        self.assertEqual((call["type"], call["function"]["name"]), ("function", "bash"))
        # mini-swe-agent parse_toolcall_actions: JSON object arguments with a "command"
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"command": srv.plan(PATCH_2)[0].command})
        usage = body["usage"]
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        self.assertTrue(all(isinstance(v, int) and v > 0 for v in usage.values()))
        json.dumps(body).encode("ascii")

    def test_unknown_task_submits_an_empty_patch(self):
        reply = self.answer(chat([SYSTEM, user_message("A problem statement nobody has seen yet.")]))
        self.assertEqual((reply.status, reply.instance), (200, "unknown"))
        self.assertEqual(command_of(reply.payload), srv.EMPTY_SUBMIT)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(submission(bash(command_of(reply.payload), Path(tmp))), "")

    def test_tools(self):
        def tool(name, props, required, kind="function"):
            return {"type": kind, "function": {"name": name, "parameters": {
                "type": "object", "properties": props, "required": required}}}
        s = {"type": "string"}
        for tools, want in ((None, ("bash", "command")), ([], ("bash", "command")),
                            ([tool("shell", {"cmd": s}, ["cmd"])], ("shell", "cmd")),
                            ([tool("shell", {"cmd": s}, ["cmd"]), BASH_TOOL], ("bash", "command")),
                            ([tool("two", {"a": s, "b": s}, ["a", "b"]), tool("sh", {"c": s}, ["c"])],
                             ("sh", "c"))):
            self.assertEqual(srv.pick_tool(tools), want, tools)
        for bad in ([tool("two", {"a": s, "b": s}, ["a", "b"])],
                    [tool("n", {"n": {"type": "integer"}}, ["n"])],
                    [tool("bad name", {"c": s}, ["c"])], [tool("sh", {"c": s}, ["c"], kind="retrieval")],
                    [tool("sh", {"c": s}, ["missing"])], "bash", {"type": "function"}, [None, 3]):
            self.assertIsNone(srv.pick_tool(bad), bad)
            self.assertEqual(self.answer(chat([SYSTEM, user_message(PROBLEM_1)], tools=bad)).status, 400)
        shell = [tool("shell", {"cmd": s}, ["cmd"])]
        reply = self.answer(chat([SYSTEM, user_message(PROBLEM_1)], tools=shell))
        function = reply.payload["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((function["name"], list(json.loads(function["arguments"]))), ("shell", ["cmd"]))

    def test_bad_requests_get_400(self):
        good = [SYSTEM, user_message(PROBLEM_1)]
        for request in (chat(good, stream=True), chat(good, stream="yes"), chat([]), chat("hi"),
                        {"tools": [BASH_TOOL]}):
            reply = self.answer(request)
            self.assertEqual(reply.status, 400, request)
            self.assertEqual(reply.payload["error"]["type"], "invalid_request_error")
        self.assertEqual(self.answer(chat(good, stream=False)).status, 200)


class Matching(unittest.TestCase):
    oracle = srv.Oracle(TASKS)

    def test_whitespace_and_line_endings_do_not_matter(self):
        for variant in (PROBLEM_1, PROBLEM_1.replace("\r\n", "\n"), PROBLEM_1.replace("\r\n", "\n\t  "),
                        PROBLEM_1.replace(" ", "\n")):
            self.assertEqual(self.oracle.match(f"<pr_description>\n{variant}\n</pr_description>"),
                             "octo__widget-101")

    def test_content_parts(self):
        messages = [SYSTEM, {"role": "user", "content": [
            {"type": "text", "text": "<pr_description>\n"}, {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": PROBLEM_2}]}]
        self.assertEqual(self.oracle.match(srv.first_user_text(messages)), "octo__gadget-202")

    def test_ambiguous_or_missing_is_unknown(self):
        for text in (PROBLEM_1 + PROBLEM_2, PROBLEM_1[:40], ""):
            self.assertIsNone(self.oracle.match(text))
        reply = srv.answer(self.oracle, chat([SYSTEM, user_message(PROBLEM_1 + "\n" + PROBLEM_2)]), "m")
        self.assertEqual((reply.instance, command_of(reply.payload)), ("unknown", srv.EMPTY_SUBMIT))

    def test_only_the_first_user_message_and_assistant_turns_count(self):
        messages = [SYSTEM, user_message(PROBLEM_1), {"role": "user", "content": PROBLEM_2},
                    {"role": "user", "content": "Tool call error: ..."}]   # FormatError feedback
        reply = srv.answer(self.oracle, chat(messages), "m")
        self.assertEqual((reply.instance, reply.step), ("octo__widget-101", 0))

    @unittest.skipUnless(TRAJS and HAVE_DATA, "pilot-2 trajectories or oracle data not present")
    def test_pilot2_conversations(self):
        oracle = srv.Oracle(srv.load_tasks(REAL_DATA, REAL_MANIFEST))
        seen = []
        for path in TRAJS:
            traj = json.loads(path.read_text())
            iid, messages = traj.get("instance_id") or path.parent.name, traj["messages"]
            self.assertEqual(oracle.match(srv.first_user_text(messages)), iid)
            convo = messages[:2]                                # system and first user message
            for step in oracle.plan_for(iid):
                reply = srv.answer(oracle, chat(convo), "m")
                self.assertEqual((reply.instance, command_of(reply.payload)), (iid, step.command))
                convo = next_turn(convo, reply.payload) + [
                    {"role": "user", "content": "Tool call error: ..."}]     # FormatError feedback
            seen.append(iid)
        self.assertEqual(sorted(seen), sorted(od.PILOT_IDS))

    @unittest.skipUnless(TRAJS and HAVE_DATA, "pilot-2 trajectories or oracle data not present")
    def test_every_generated_task_matches_only_itself_in_a_real_first_message(self):
        """Each task of the generated data (200 for the production rehearsal) in place of the
        pilot task's statement inside a real pilot-2 first user message (the drop's template)."""
        tasks = srv.load_tasks(REAL_DATA, REAL_MANIFEST)
        oracle = srv.Oracle(tasks)
        traj = json.loads(TRAJS[0].read_text())
        pilot_id = traj.get("instance_id") or TRAJS[0].parent.name
        message = srv.normalize(srv.first_user_text(traj["messages"]))
        own = srv.normalize(next(t.problem_statement for t in tasks if t.instance_id == pilot_id))
        self.assertEqual(message.count(own), 1)
        for task in tasks:
            text = message.replace(own, srv.normalize(task.problem_statement))
            self.assertEqual(oracle.match(text), task.instance_id)


# ----------------------------------------------------------------------------- data
def gz_doc(instances, schema=srv.DATA_SCHEMA) -> bytes:
    return gzip.compress(json.dumps({"schema": schema, "instances": instances}).encode())


def api_rows(n: int, first_ids=od.PILOT_IDS) -> list:
    ids = list(first_ids)[:n] + [f"octo__widget-{9000 + i}" for i in range(max(0, n - len(first_ids)))]
    return [{"instance_id": iid, "repo": "octo/widget", "base_commit": "0" * 40,
             "problem_statement": f"Issue {i:04d}: widget {i:04d} breaks when it is resized twice.",
             "patch": f"diff --git a/w{i}.py b/w{i}.py\n--- a/w{i}.py\n+++ b/w{i}.py\n@@ -1 +1 @@\n"
                      f"-x = {i}\n+x = {i + 1}\n"} for i, iid in enumerate(ids)]


class FakeAPI:
    """The datasets-server /rows API over `rows`; pages longer than `cut` come back truncated."""

    def __init__(self, rows, cut=None):
        self.rows, self.cut, self.calls = rows, cut, []

    def __call__(self, url):
        parts = urllib.parse.urlsplit(url)
        assert (parts.scheme, parts.hostname, parts.path) == ("https", od.API_HOST, "/rows"), url
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        assert (q["dataset"], q["config"], q["split"]) == (od.DATASET, od.CONFIG, od.SPLIT), q
        offset, length = int(q["offset"]), int(q["length"])
        self.calls.append((offset, length))
        cut = self.cut is not None and length > self.cut
        return {"rows": [{"row_idx": i, "row": self.rows[i], "truncated_cells": ["patch"] if cut else []}
                         for i in range(offset, min(offset + length, len(self.rows)))],
                "num_rows_total": len(self.rows), "num_rows_per_page": 100, "partial": False}


class Data(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oracle-data-test-"))
        self.addCleanup(shutil.rmtree, self.tmp)

    def refused(self, data: bytes, manifest=None, fragment=""):
        (self.tmp / "d.json.gz").write_bytes(data)
        mpath = None
        if manifest is not None:
            mpath = self.tmp / "m.json"
            mpath.write_text(json.dumps(manifest))
        with self.assertRaises(srv.DataError) as cm:
            srv.load_tasks(self.tmp / "d.json.gz", mpath)
        self.assertIn(fragment, str(cm.exception))
        self.assertNotIn("aspect ratio", str(cm.exception))   # names tasks, never quotes them

    def test_valid_data_loads_with_and_without_manifest(self):
        data = od.build_data(TASKS)
        (self.tmp / "d.json.gz").write_bytes(data)
        (self.tmp / "m.json").write_text(json.dumps(od.build_manifest(TASKS, data, "t")))
        self.assertEqual(srv.load_tasks(self.tmp / "d.json.gz", self.tmp / "m.json"), TASKS)
        self.assertEqual(srv.load_tasks(self.tmp / "d.json.gz"), TASKS)

    def test_bad_data_is_refused(self):
        row = ROWS[0]
        for data, fragment in (
                (gz_doc(ROWS, schema="other/1"), "is not"),
                (gz_doc([{**row, "instance_id": "../../etc/passwd"}]), "invalid instance_id"),
                (gz_doc([row, dict(row)]), "duplicate"),
                (gz_doc([{**row, "patch": "--- a/x\n+++ b/x\n"}]), "diff --git"),
                (gz_doc([{**row, "extra": 1}]), "exactly the keys"),
                (gz_doc([{**row, "problem_statement": "too short"}]), "problem_statement"),
                (gz_doc([row, {**ROWS[1], "problem_statement": PROBLEM_1 + " And more."}]), "part of"),
                (gz_doc([]), "1 to"), (gz_doc("x"), "1 to"),
                (b"\x1f\x8b not really gzip", "cannot read"), (gz_doc(ROWS)[:-12], "cannot read"),
                (gzip.compress(b"{not json"), "cannot read"), (gzip.compress(b"[1, 2]"), "is not")):
            with self.subTest(fragment=fragment):
                self.refused(data, fragment=fragment)

    def test_manifest_must_match(self):
        data = od.build_data(TASKS)
        manifest = od.build_manifest(TASKS, data, "t")
        self.refused(data, {**manifest, "data_sha256": "0" * 64}, "data_sha256")
        self.refused(data, ["not", "an", "object"], "data_sha256")
        self.refused(data, {**manifest, "instances": manifest["instances"][::-1]}, "differ")

    def test_size_caps(self):
        data = od.build_data(TASKS)
        with mock.patch.object(srv, "MAX_DATA_GZ", len(data) - 1):
            self.refused(data, fragment="larger than")
        with mock.patch.object(srv, "MAX_DATA_JSON", 100):
            self.refused(data, fragment="decompresses to more than")

    def test_build_data_is_deterministic(self):
        a, b = od.build_data(TASKS), od.build_data(list(TASKS))
        self.assertEqual(a, b)
        self.assertEqual(a[:3], b"\x1f\x8b\x08")
        self.assertEqual(a[3] & 0x08, 0)                       # no file name
        self.assertEqual(a[4:8], b"\0\0\0\0")                   # mtime 0
        doc = json.loads(gzip.decompress(a))
        self.assertEqual((doc["schema"], list(doc)), (srv.DATA_SCHEMA, sorted(doc)))

    def test_manifest_records_each_patch(self):
        data = od.build_data(TASKS)
        manifest = od.build_manifest(TASKS, data, "t")
        self.assertEqual(manifest["data_sha256"], hashlib.sha256(data).hexdigest())
        for entry, task in zip(manifest["instances"], TASKS, strict=True):
            self.assertEqual(entry["instance_id"], task.instance_id)
            self.assertEqual(entry["patch_sha256"], hashlib.sha256(task.patch.encode()).hexdigest())
            self.assertEqual((entry["plan_steps"], entry["patch_has_cr"]), (3, False))

    def test_fetch_pages_in_order(self):
        api = FakeAPI(api_rows(150))
        rows = od.fetch_rows(150, api)
        self.assertEqual(api.calls, [(0, 100), (100, 50)])
        self.assertEqual([r["instance_id"] for r in rows], [r["instance_id"] for r in api.rows])
        self.assertEqual(set(rows[0]), set(od.FIELDS))

    def test_truncated_cells_halve_the_page(self):
        api = FakeAPI(api_rows(60), cut=20)
        rows = od.fetch_rows(60, api)
        self.assertEqual(api.calls, [(0, 60), (0, 30), (0, 15), (15, 15), (30, 15), (45, 15)])
        self.assertEqual([r["instance_id"] for r in rows], [r["instance_id"] for r in api.rows])

    def test_fetch_errors(self):
        api = FakeAPI(api_rows(10))
        for count, fetch in ((20, api),                                          # too few rows
                             (10, lambda url: {**api(url), "rows": api(url)["rows"][::-1]}),
                             (5, lambda url: {"rows": "nope"}),
                             (3, FakeAPI(api_rows(10), cut=0))):                 # cut even alone
            with self.assertRaises(od.FetchError):
                od.fetch_rows(count, fetch)

    def test_fetch_json_only_talks_to_the_datasets_server_over_https(self):
        for url in ("http://datasets-server.huggingface.co/rows?x=1", "https://example.org/rows",
                    "https://datasets-server.huggingface.co.example.org/rows", "<workdir>/etc/hosts"):
            with self.assertRaises(od.FetchError):
                od.fetch_json(url)

    def test_main_writes_data_the_server_loads(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(od.main(["--count", "12"], fetch=FakeAPI(api_rows(12)), out_dir=self.tmp), 0)
        data, manifest = self.tmp / od.DATA_NAME, self.tmp / od.MANIFEST_NAME
        self.assertEqual(json.loads(manifest.read_text())["data_sha256"],
                         hashlib.sha256(data.read_bytes()).hexdigest())
        self.assertEqual(tuple(t.instance_id for t in srv.load_tasks(data, manifest))[:10], od.PILOT_IDS)
        self.assertEqual(data.stat().st_mode & 0o777, 0o644)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), sorted([od.DATA_NAME, od.MANIFEST_NAME]))

    def test_main_refuses_other_rows_and_bad_counts(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(od.main([], fetch=FakeAPI(api_rows(10, ["octo__other-1"])), out_dir=self.tmp), 1)
            for count in ("0", "501"):
                with self.assertRaises(SystemExit):
                    od.main(["--count", count], fetch=FakeAPI(api_rows(10)), out_dir=self.tmp)
        self.assertIn("the pilot runs", err.getvalue())
        self.assertEqual(list(self.tmp.iterdir()), [])

    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_generated_data(self):
        manifest, raw = json.loads(REAL_MANIFEST.read_text()), REAL_DATA.read_bytes()
        self.assertEqual((manifest["data_sha256"], manifest["data_bytes"]),
                         (hashlib.sha256(raw).hexdigest(), len(raw)))
        tasks = srv.load_tasks(REAL_DATA, REAL_MANIFEST)
        self.assertEqual(tuple(t.instance_id for t in tasks[:10]), od.PILOT_IDS)
        self.assertEqual(od.build_data(tasks), raw)               # reproducible from its rows
        for entry, task in zip(manifest["instances"], tasks, strict=True):
            self.assertEqual(entry["patch_sha256"], hashlib.sha256(task.patch.encode()).hexdigest())
            self.assertEqual(entry["plan_steps"], len(srv.plan(task.patch)))
            self.assertFalse(entry["patch_has_cr"], task.instance_id)
        self.assertEqual(srv.Oracle(tasks).warnings, [])
        self.assertLessEqual(len(raw), od.MAX_GZ)


# ----------------------------------------------------------------------------- HTTP
@contextlib.contextmanager
def serving(oracle=None, **kw):
    httpd = srv.OracleServer(("127.0.0.1", 0), oracle or srv.Oracle(TASKS), "test-model", **kw)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def exchange(port: int, data: bytes, *, half_close: bool = False) -> bytes:
    """Send raw bytes, read until the server closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        if data:
            s.sendall(data)
        if half_close:
            s.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := s.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def parse(raw: bytes) -> tuple:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    return int(lines[0].split()[1]), headers, body


POST = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._serving = serving()
        cls.port = cls._serving.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._serving.__exit__(None, None, None)

    def connect(self) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        self.addCleanup(conn.close)
        return conn

    def call(self, method, path, body=None, headers=None, conn=None):
        conn = conn or self.connect()
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp, resp.read()

    def test_health_and_models_keep_alive(self):
        conn = self.connect()
        resp, body = self.call("GET", "/health", conn=conn)
        self.assertEqual((resp.status, json.loads(body)), (200, {"status": "ok"}))
        sock = conn.sock
        resp, body = self.call("GET", "/v1/models?verbose=1", conn=conn)
        self.assertIs(conn.sock, sock)                        # one connection, kept alive
        models = json.loads(body)
        self.assertEqual((models["object"], [m["id"] for m in models["data"]]), ("list", ["test-model"]))

    def test_security_headers(self):
        resp, _ = self.call("GET", "/health")
        for name, value in srv.SECURITY_HEADERS:
            self.assertEqual(resp.getheader(name), value)
        self.assertEqual((resp.getheader("Content-Type"), resp.getheader("Server")),
                         ("application/json", "swe-oracle"))

    def test_chat_completions(self):
        conn, messages, steps = self.connect(), [SYSTEM, user_message(PROBLEM_1)], srv.plan(PATCH_1)
        for k in range(4):
            resp, body = self.call("POST", "/v1/chat/completions", json.dumps(chat(messages)).encode(),
                                   {"Content-Type": "application/json",
                                    "Authorization": f"Bearer {SECRETISH}"}, conn=conn)
            self.assertEqual(resp.status, 200, body)
            self.assertEqual(command_of(json.loads(body)), steps[min(k, 2)].command)
            messages = next_turn(messages, json.loads(body))
        self.assertTrue(logged("POST /v1/chat/completions 200 instance=octo__widget-101 step=3"))
        text = "\n".join(LOGS)
        self.assertNotIn(SECRETISH, text)
        self.assertNotIn("aspect ratio", text)                # no request bodies in the log

    def test_routes_and_methods(self):
        for method, path, status, allow in (
                ("GET", "/nope", 404, None), ("POST", "/nope", 404, None),
                ("GET", "/v1/chat/completions", 405, "POST"), ("PUT", "/v1/chat/completions", 405, "POST"),
                ("POST", "/health", 405, "GET"), ("POST", "/v1/models", 405, "GET"),
                ("DELETE", "/health", 405, "GET"), ("OPTIONS", "/health", 405, "GET")):
            with self.subTest(method=method, path=path):
                resp, body = self.call(method, path)
                self.assertEqual((resp.status, resp.getheader("Allow")), (status, allow))
                self.assertEqual(resp.getheader("Connection"), "close")
                self.assertIn("error", json.loads(body))

    def test_head_and_unknown_methods(self):
        resp, body = self.call("HEAD", "/health")
        self.assertEqual((resp.status, body, resp.getheader("Allow")), (405, b"", "GET"))
        resp, body = self.call("BREW", "/health")
        self.assertEqual((resp.status, json.loads(body)["error"]["message"]), (501, "not implemented"))

    def test_body_framing_errors(self):
        for request, status, half_close in (
                (POST + b"\r\n", 411, False),
                (POST + b"Transfer-Encoding: chunked\r\n\r\n", 411, False),
                (POST + b"Content-Length: abc\r\n\r\n", 400, False),
                (POST + b"Content-Length: -1\r\n\r\n", 400, False),
                (POST + b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n", 400, False),
                (POST + f"Content-Length: {srv.MAX_BODY + 1}\r\n\r\n".encode(), 413, False),
                (POST + b'Content-Length: 100\r\n\r\n{"messages": []}', 400, True)):     # short body
            with self.subTest(request=request[len(POST):60]):
                code, headers, body = parse(exchange(self.port, request, half_close=half_close))
                self.assertEqual((code, headers.get("connection")), (status, "close"))
                self.assertIn("error", json.loads(body))

    def test_json_errors(self):
        for body in (b"not json", b"[1, 2]", b"\xff\xfe", b"", json.dumps({"messages": []}).encode(),
                     json.dumps(chat([SYSTEM], stream=True)).encode()):
            with self.subTest(body=body[:40]):
                resp, data = self.call("POST", "/v1/chat/completions", body)
                self.assertEqual(resp.status, 400)
                self.assertEqual(json.loads(data)["error"]["type"], "invalid_request_error")

    def test_log_lines_are_sanitized(self):
        request = (b"GET /h\xc3\xa9alth\x1b[31m?token=abc HTTP/1.1\r\nHost: x\r\n"
                   + f"X-Api-Key: {SECRETISH}\r\n\r\n".encode())
        self.assertEqual(parse(exchange(self.port, request))[0], 404)
        self.assertTrue(logged("GET /h??alth?[31m 404 instance=- step=-"))
        text = "\n".join(LOGS)
        self.assertNotIn("token=abc", text)
        self.assertNotIn(SECRETISH, text)
        self.assertTrue(all(" " <= c <= "~" for c in text.replace("\n", "")))

    def test_unread_body_detection(self):
        def handler(raw: bytes, body_read: bool = False):
            return SimpleNamespace(headers=http.client.parse_headers(io.BytesIO(raw)), _body_read=body_read)
        unread = srv.OracleHandler._unread_body
        self.assertTrue(unread(handler(b"Content-Length: 5\r\n\r\n")))
        self.assertTrue(unread(handler(b"Transfer-Encoding: chunked\r\n\r\n")))
        self.assertFalse(unread(handler(b"Content-Length: 0\r\n\r\n")))
        self.assertFalse(unread(handler(b"\r\n")))
        self.assertFalse(unread(handler(b"Content-Length: 5\r\n\r\n", body_read=True)))


class Limits(unittest.TestCase):
    def test_connection_slots_then_503(self):
        with serving(max_connections=1, slot_wait=0.2) as port:
            first = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                first.request("GET", "/health")
                self.assertEqual(first.getresponse().read(), b'{"status":"ok"}')
                # `first` stays open (keep-alive), so its handler holds the only slot.
                status, headers, body = parse(exchange(port, b""))
                self.assertEqual((status, headers["retry-after"], headers["connection"]), (503, "1", "close"))
                self.assertEqual(json.loads(body)["error"]["type"], "server_busy")
                self.assertTrue(logged("busy: a connection got 503"))
            finally:
                first.close()
            deadline, status = time.monotonic() + 5, None
            while status != 200 and time.monotonic() < deadline:      # the slot comes back
                try:
                    status = parse(exchange(port, b"GET /health HTTP/1.1\r\nHost: x\r\n"
                                                  b"Connection: close\r\n\r\n"))[0]
                except OSError:
                    time.sleep(0.05)
            self.assertEqual(status, 200)

    def test_idle_connections_time_out(self):
        with serving(max_connections=1, socket_timeout=0.3) as port:
            start = time.monotonic()
            self.assertEqual(exchange(port, b""), b"")             # closed without a reply
            self.assertLess(time.monotonic() - start, 3)
            self.assertTrue(logged("request error: timed out"))
            request = b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
            self.assertEqual(parse(exchange(port, request))[0], 200)     # slot released

    def test_internal_errors_do_not_leak(self):
        def broken(*_args):
            raise RuntimeError(f"request-derived detail {SECRETISH}")
        with mock.patch.object(srv, "answer", broken), serving() as port:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps(chat([SYSTEM])).encode())
                resp = conn.getresponse()
                body = resp.read()
            finally:
                conn.close()
        self.assertEqual((resp.status, json.loads(body)["error"]["message"]), (500, "internal error"))
        self.assertTrue(logged("internal error: RuntimeError"))
        self.assertNotIn(SECRETISH.encode(), body)
        self.assertNotIn(SECRETISH, "\n".join(LOGS))


class Loopback(unittest.TestCase):
    def test_only_literal_loopback_addresses(self):
        self.assertEqual(srv.loopback_family("127.0.0.1"), socket.AF_INET)
        self.assertEqual(srv.loopback_family("127.8.9.10"), socket.AF_INET)
        self.assertEqual(srv.loopback_family("::1"), socket.AF_INET6)
        for host in ("0.0.0.0", "<ip>", "::", "", "localhost", "::ffff:127.0.0.1", "::1%lo",
                     "fe80::1", "<ip>", "127.0.0.1 ", "0177.0.0.1", "2130706433"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                srv.loopback_family(host)

    def test_server_refuses_before_any_socket_exists(self):
        with mock.patch.object(socket, "socket", side_effect=AssertionError("socket created")):
            for host in ("0.0.0.0", "::", "localhost", "<ip>"):
                with self.subTest(host=host), self.assertRaises(ValueError):
                    srv.OracleServer((host, 0), srv.Oracle(TASKS))
        with self.assertRaises(ValueError):
            srv.OracleServer(("127.0.0.1", 0), srv.Oracle(TASKS), "bad name")

    def test_printable(self):
        self.assertEqual(srv.printable("a\nb\x1b[0m\u00e9"), "a?b?[0m?")
        self.assertEqual(len(srv.printable("x" * 500)), 80)
        self.assertEqual((srv.printable(""), srv.printable(None)), ("-", "-"))


# ----------------------------------------------------------------------------- CLI
def read_until(stream, fragment: str, timeout: float = 15.0) -> str:
    deadline, seen = time.monotonic() + timeout, []
    while time.monotonic() < deadline:
        if select.select([stream], [], [], 0.1)[0]:
            line = stream.readline()
            if not line:
                break
            seen.append(line)
            if fragment in line:
                return line
    raise AssertionError(f"no line with {fragment!r} in {seen}")


class Cli(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_the_pod_command_serves_and_stops_on_sigterm(self):
        command = sp.oracle_pod("gke-node-a")["spec"]["containers"][0]["command"]
        self.assertEqual(command[:2], ["python3", "-I"])
        local = [a.replace(sp.ORACLE_MOUNT + "/", f"{ORACLE_DIR}/") for a in command[2:]]
        local[local.index("--port") + 1] = "0"
        problem = srv.load_tasks(REAL_DATA)[0].problem_statement
        proc = subprocess.Popen([sys.executable, "-I", *local], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            line = read_until(proc.stdout, " at http://")
            port = int(re.search(r" at http://127\.0\.0\.1:(\d+) ", line)[1])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            self.assertEqual(conn.getresponse().read(), b'{"status":"ok"}')
            conn.request("GET", "/v1/models")
            self.assertEqual(json.loads(conn.getresponse().read())["data"][0]["id"], sp.ORACLE_MODEL_NAME)
            conn.request("POST", "/v1/chat/completions",
                         json.dumps(chat([SYSTEM, user_message(problem)])).encode())
            self.assertTrue(command_of(json.loads(conn.getresponse().read())).startswith("printf '%s' '"))
            conn.close()
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
            out, err = proc.stdout.read(), proc.stderr.read()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()
        self.assertEqual(proc.returncode, 0, err)
        self.assertIn("instance=astropy__astropy-12907 step=0", out)
        self.assertIn("stopped", out)
        self.assertNotIn("warning", err)

    def test_refusals(self):
        missing = ["--data", "/nonexistent/oracle-data.json.gz"]
        for args, fragment in ((missing + ["--host", "0.0.0.0"], "refusing to listen on 0.0.0.0"),
                               (missing + ["--host", "localhost"], "refusing to listen on localhost"),
                               (missing + ["--port", "70000"], "--port must be"),
                               (missing, "cannot read the oracle data")):
            with self.subTest(args=args):
                p = subprocess.run([sys.executable, "-I", str(SERVER), *args], capture_output=True,
                                   text=True, timeout=30, check=False)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn(fragment, p.stderr)
                if "--host" in args:
                    self.assertNotIn("cannot read", p.stderr)      # refused before reading data


# ----------------------------------------------------------------------------- swe_pilot.py
class FakeKube:
    def __init__(self, objects=None, dry_run=False):
        self.objects, self.dry_run = dict(objects or {}), dry_run
        self.created, self.applied = [], []
        self.redact = sp.Redactor()

    def get(self, kind, name, namespaced=True):
        return self.objects.get((kind, name))

    def create(self, obj):
        self.created.append(obj)

    def apply(self, obj):
        self.applied.append(obj)


READY_NODE = {"metadata": {"labels": {"kubernetes.io/arch": "arm64"}}, "spec": {},
              "status": {"conditions": [{"type": "Ready", "status": "True"}]}}


class PilotWiring(unittest.TestCase):
    node = "gke-node-a"

    def test_oracle_pod(self):
        pod = sp.oracle_pod(self.node)
        meta, spec = pod["metadata"], pod["spec"]
        self.assertEqual((meta["name"], meta["namespace"]), (sp.POD["model"], sp.NS))
        self.assertEqual(meta["labels"],
                         {**sp.PART_OF, "kimi-swe/role": "model", "kimi-swe/variant": "oracle"})
        self.assertEqual(spec["nodeSelector"], {"kubernetes.io/hostname": self.node})
        self.assertTrue(spec["hostNetwork"])             # the node's 127.0.0.1, shared with host and client
        self.assertFalse(spec["automountServiceAccountToken"])
        (c,) = spec["containers"]
        self.assertEqual((c["name"], c["image"]), ("oracle", sp.CLIENT_IMAGE))
        cmd = c["command"]
        self.assertEqual(cmd[:3], ["python3", "-I", f"{sp.ORACLE_MOUNT}/oracle_server.py"])
        self.assertFalse(any(ch in arg for arg in cmd for ch in "$;&|`"))          # no shell
        self.assertEqual(dict(zip(cmd[3::2], cmd[4::2])), {
            "--data": f"{sp.ORACLE_MOUNT}/{sp.ORACLE_DATA}",
            "--manifest": f"{sp.ORACLE_MOUNT}/oracle-manifest.json",
            "--host": "127.0.0.1", "--port": str(sp.MODEL_PORT), "--model-name": sp.ORACLE_MODEL_NAME})
        for arg in cmd[2:]:                              # every mounted path is a ConfigMap key
            if arg.startswith(sp.ORACLE_MOUNT + "/"):
                self.assertIn(arg.rsplit("/", 1)[1], (*sp.ORACLE_TEXT, sp.ORACLE_DATA))
        self.assertEqual(c["securityContext"], {
            "runAsUser": 65534, "runAsGroup": 65534, "runAsNonRoot": True, "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]}, "readOnlyRootFilesystem": True,
            "seccompProfile": {"type": "RuntimeDefault"}})
        self.assertNotIn("gpu", json.dumps(c["resources"]))
        model_probe = sp.model_pod(self.node)["spec"]["containers"][0]["readinessProbe"]
        self.assertEqual(c["readinessProbe"], model_probe)
        self.assertIn(f"http://127.0.0.1:{sp.MODEL_PORT}/health", json.dumps(c["readinessProbe"]))
        self.assertEqual(spec["volumes"],
                         [{"name": "oracle", "configMap": {"name": sp.CM_ORACLE, "defaultMode": 0o444}}])
        self.assertEqual(c["volumeMounts"],
                         [{"name": "oracle", "mountPath": sp.ORACLE_MOUNT, "readOnly": True}])
        self.assertEqual(sp.ORACLE_MODEL_NAME, srv.MODEL_NAME)

    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_oracle_configmap(self):
        cm = sp.oracle_configmap(sp.ORACLE_DIR)
        self.assertEqual((cm["metadata"]["name"], cm["metadata"]["namespace"]), (sp.CM_ORACLE, sp.NS))
        self.assertEqual(set(cm["data"]), set(sp.ORACLE_TEXT))
        self.assertEqual(cm["data"]["oracle_server.py"].encode(), SERVER.read_bytes())
        self.assertEqual(base64.b64decode(cm["binaryData"][sp.ORACLE_DATA]), REAL_DATA.read_bytes())
        size = sum(len(v.encode()) for v in cm["data"].values()) + sum(
            len(base64.b64decode(v)) for v in cm["binaryData"].values())
        self.assertLess(size, 900 * 1024)
        self.assertEqual(sp.oracle_configmap(sp.ORACLE_DIR)["metadata"]["annotations"],
                         cm["metadata"]["annotations"])

    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_oracle_configmap_needs_complete_consistent_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(sp.oracle_configmap(d))
            for name in (*sp.ORACLE_TEXT, sp.ORACLE_DATA):
                shutil.copyfile(ORACLE_DIR / name, d / name)
            self.assertEqual(sp.oracle_configmap(d)["metadata"]["annotations"],
                             sp.oracle_configmap(sp.ORACLE_DIR)["metadata"]["annotations"])
            good = (d / "oracle-manifest.json").read_text()
            for bad in (json.dumps({**json.loads(good), "data_sha256": "0" * 64}), "[]", "{not json"):
                (d / "oracle-manifest.json").write_text(bad)
                self.assertIsNone(sp.oracle_configmap(d))
            (d / "oracle-manifest.json").write_text(good)
            (d / sp.ORACLE_DATA).unlink()
            self.assertIsNone(sp.oracle_configmap(d))

    @unittest.skipUnless(HAVE_INPUTS and HAVE_DATA, "drop scripts, artifacts or oracle data not present")
    def test_configmaps_command(self):
        four = [sp.NS, sp.CM_DROP, sp.CM_HOST, sp.CM_CLIENT, sp.CM_MODEL]
        for argv, want in ((["configmaps"], four + [sp.CM_ORACLE]),
                           (["--oracle-dir", "", "configmaps"], four)):
            kube, out = FakeKube(), io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(sp.cmd_configmaps(sp.parser().parse_args(argv), kube), 0)
            self.assertEqual([o["metadata"]["name"] for o in kube.applied], want)
            self.assertEqual(f"{sp.CM_ORACLE} (test only, for `model --oracle`): applied" in out.getvalue(),
                             sp.CM_ORACLE in want)

    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_require_oracle(self):
        args, want = SimpleNamespace(oracle_dir=str(sp.ORACLE_DIR)), sp.oracle_configmap(sp.ORACLE_DIR)
        sp.require_oracle(FakeKube({("configmap", sp.CM_ORACLE): want}), args)
        stale = json.loads(json.dumps(want))
        stale["metadata"]["annotations"][sp.DIGEST_KEY] = "0" * 64
        for objects, a, fragment in (({}, args, "kimi-swe-oracle is missing or stale"),
                                     ({("configmap", sp.CM_ORACLE): stale}, args, "missing or stale"),
                                     ({("configmap", sp.CM_ORACLE): want}, SimpleNamespace(oracle_dir=""),
                                      "run oracle/oracle_data.py"),
                                     ({("configmap", sp.CM_ORACLE): want}, SimpleNamespace(),
                                      "run oracle/oracle_data.py")):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                sp.require_oracle(FakeKube(objects), a)
            self.assertIn(fragment, err.getvalue())

    @unittest.skipUnless(HAVE_DATA, "oracle_data.py output not present")
    def test_model_oracle_dry_run(self):
        kube = FakeKube({("node", self.node): READY_NODE,
                         ("configmap", sp.CM_ORACLE): sp.oracle_configmap(sp.ORACLE_DIR)}, dry_run=True)
        args = sp.parser().parse_args(["--dry-run", "model", "--node", self.node, "--oracle"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = sp.cmd_model(args, kube)
        self.assertEqual(code, 0)
        self.assertEqual(kube.created, [sp.oracle_pod(self.node)])
        self.assertEqual(json.loads(out.getvalue()), sp.oracle_pod(self.node))

    @unittest.skipUnless(HAVE_INPUTS, "drop scripts or artifacts not present")
    def test_model_without_oracle_is_unchanged(self):
        args = sp.parser().parse_args(["--dry-run", "model", "--node", self.node])
        cms = {cm["metadata"]["name"]: cm for cm in sp.desired_configmaps(args)}
        kube = FakeKube({("node", self.node): READY_NODE, ("configmap", sp.CM_MODEL): cms[sp.CM_MODEL]},
                        dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sp.cmd_model(args, kube), 0)
        self.assertEqual(kube.created, [sp.model_pod(self.node)])

    def test_oracle_refuses_keep_default_caps(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            sp.parser().parse_args(["model", "--node", self.node, "--oracle", "--keep-default-caps"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("not allowed with argument", err.getvalue())
        args = SimpleNamespace(oracle=True, keep_default_caps=True, node=self.node, allow_busy_node=False)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sp.cmd_model(args, None)                     # before any kubectl call: kube is None
        defaults = sp.parser().parse_args(["model", "--node", self.node])
        self.assertEqual((defaults.oracle, defaults.keep_default_caps, defaults.oracle_dir),
                         (False, False, str(sp.ORACLE_DIR)))


class Hygiene(unittest.TestCase):
    def test_server_is_standard_library_python_3_12(self):
        tree = ast.parse(SERVER.read_text(), feature_version=(3, 12))      # the client image's Python
        nodes = list(ast.walk(tree))
        imported = {a.name.split(".")[0] for n in nodes if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in nodes if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(imported, set(sys.stdlib_module_names))
        self.assertNotIn("pickle", imported)
        ast.parse((ORACLE_DIR / "oracle_data.py").read_text(), feature_version=(3, 12))


if __name__ == "__main__":
    unittest.main()
