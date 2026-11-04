#!/usr/bin/env python3
"""Test-only SWE-bench "oracle" endpoint for the Kimi-K3 SWE pilot on GKE.

It answers each SWE-bench task with the task's public gold patch, so a working agent loop and
grading harness report every served task as resolved. A harness test only: never a model, never
a submission.

OpenAI-compatible subset, standard library only (runs as `python3 -I` in the client image):
  GET  /health               {"status": "ok"}
  GET  /v1/models            the one served model id (--model-name)
  POST /v1/chat/completions  one tool call per reply; no streaming

mini-swe-agent 2.3.0 (LitellmModel, tool calling) drives each task. The oracle finds the task by
its problem statement in the first user message (whitespace-normalized substring, exactly one
match) and takes the step from the number of assistant messages so far. Its replies:
  1. write the gold patch to patch.txt: printf '%s' '<base64>' | base64 -d > patch.txt
     (base64 longer than CHUNK goes to patch.b64 over several steps, then one decode step),
  2. git apply --check -v patch.txt: a sandbox check only; the plan goes on whatever it says,
  3. echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt: the template's submit
     command. The environment takes the raw output after the sentinel line as the submission
     (swebench_service PyxisEnvironment.execute; mini-swe-agent LocalEnvironment._check_finished).
Past the plan it submits again. An unknown task gets the bare sentinel at once: an empty
submission, which the harness reports as empty_patch.

Data: oracle-data.json.gz from oracle_data.py (gzip + JSON only, never pickle), validated on
load and, with --manifest, checked against the sha256 recorded in oracle-manifest.json.

TODO(security): no authentication. Accepted for this test-only endpoint: it listens on loopback
only (other addresses are refused), serves public SWE-bench data and changes nothing. Any process
on the node can reach 127.0.0.1:18000, the same exposure as the pilot's stand-in model.
"""
import argparse
import base64
import gzip
import hashlib
import http.server
import io
import ipaddress
import json
import re
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
import zlib
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import NamedTuple

DATA_SCHEMA = "kimi-swe-oracle-data/1"
MODEL_NAME = "swe-oracle-gold-patch"
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT = f"echo {SENTINEL} && cat patch.txt"
EMPTY_SUBMIT = f"echo {SENTINEL}"
CHECK = "git apply --check -v patch.txt"
# Base64 characters per command. A command reaches the sandbox as one argv element (bash -c inside
# the Enroot wrapper script), and Linux caps one argv element at MAX_ARG_STRLEN = 128 KiB.
CHUNK = 60_000

MAX_BODY = 16 * 1024 * 1024          # request body, bytes
MAX_DATA_GZ = 16 * 1024 * 1024       # oracle-data.json.gz, bytes
MAX_DATA_JSON = 64 * 1024 * 1024     # the same, decompressed
MAX_MANIFEST = 4 * 1024 * 1024
MAX_TASKS = 500                      # SWE-bench Verified has 500 rows
MAX_PROBLEM = 256 * 1024             # characters
MAX_PATCH = 1024 * 1024              # characters
MIN_PROBLEM = 20                     # normalized characters; shorter text could match the template
SOCKET_TIMEOUT = 60.0                # seconds per socket read or write
MAX_CONNECTIONS = 64                 # open connections; the pilot runs 16 agents
SLOT_WAIT = 5.0                      # seconds a new connection may wait for a slot, then 503

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}__[A-Za-z0-9][A-Za-z0-9_.-]{0,99}-[0-9]{1,9}")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
TOOL_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")           # OpenAI function and parameter names
LENGTH_RE = re.compile(r"[0-9]{1,12}")
SECURITY_HEADERS = (("X-Content-Type-Options", "nosniff"), ("Cache-Control", "no-store"),
                    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
                    ("X-Frame-Options", "DENY"))
ROUTES = {"/health": "GET", "/v1/models": "GET", "/v1/chat/completions": "POST"}

_LOG_LOCK = threading.Lock()


def log(msg: str, *, err: bool = False) -> None:
    with _LOG_LOCK:
        print(f"[swe-oracle {datetime.now(timezone.utc):%H:%M:%SZ}] {msg}",
              file=sys.stderr if err else sys.stdout, flush=True)


def printable(text, limit: int = 80) -> str:
    """Request-controlled text for a log line: printable ASCII only, bounded."""
    return "".join(c if "!" <= c <= "~" else "?" for c in str(text or "")[:limit]) or "-"


# ----------------------------------------------------------------------------- data
class Task(NamedTuple):
    instance_id: str
    problem_statement: str
    patch: str


class Step(NamedTuple):
    command: str
    note: str


class DataError(ValueError):
    """Unusable oracle data or manifest. Messages name tasks by id, never quote their text."""


def normalize(text: str) -> str:
    """Every whitespace run as one space: CRLF vs LF, wrapping and indentation do not matter."""
    return " ".join(text.split())


def validate_tasks(rows) -> list[Task]:
    """The `instances` of oracle-data.json.gz as Tasks; DataError on anything unexpected."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TASKS:
        raise DataError(f"instances must be a list of 1 to {MAX_TASKS} tasks")
    tasks: list[Task] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(Task._fields):
            raise DataError(f"task {i}: expected exactly the keys {', '.join(Task._fields)}")
        iid, problem, patch = row["instance_id"], row["problem_statement"], row["patch"]
        if not isinstance(iid, str) or not ID_RE.fullmatch(iid):
            raise DataError(f"task {i}: invalid instance_id")
        if iid in seen:
            raise DataError(f"task {i}: duplicate instance_id {iid}")
        if (not isinstance(problem, str) or len(problem) > MAX_PROBLEM
                or len(normalize(problem)) < MIN_PROBLEM):
            raise DataError(f"{iid}: problem_statement must have {MIN_PROBLEM} to {MAX_PROBLEM} "
                            "characters")
        if not isinstance(patch, str) or not patch.startswith("diff --git ") or len(patch) > MAX_PATCH:
            raise DataError(f"{iid}: patch must start with 'diff --git ' and have at most "
                            f"{MAX_PATCH} characters")
        seen.add(iid)
        tasks.append(Task(iid, problem, patch))
    norms = [normalize(t.problem_statement) for t in tasks]
    for i, inner in enumerate(norms):
        for j, outer in enumerate(norms):
            if i != j and inner in outer:
                raise DataError(f"{tasks[i].instance_id}: its problem statement is part of the one "
                                f"of {tasks[j].instance_id}, so the oracle could match neither")
    return tasks


def plan(patch: str) -> list[Step]:
    """The oracle's replies for one task, in order; the last one repeats past the end."""
    b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")      # [A-Za-z0-9+/=]: no quotes
    if len(b64) <= CHUNK:
        write = [Step(f"printf '%s' '{b64}' | base64 -d > patch.txt",
                      "write the gold patch to patch.txt")]
    else:
        parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
        write = [Step(f"printf '%s' '{part}' {'>>' if k else '>'} patch.b64",
                      f"write part {k + 1}/{len(parts)} of the base64 gold patch to patch.b64")
                 for k, part in enumerate(parts)]
        write.append(Step("base64 -d patch.b64 > patch.txt && rm -f patch.b64",
                          "decode patch.b64 into patch.txt"))
    return [*write,
            Step(CHECK, "check that patch.txt applies (sandbox only; the plan goes on either way)"),
            Step(SUBMIT, "submit patch.txt")]


UNKNOWN_PLAN = [Step(EMPTY_SUBMIT, "unknown task: submit an empty patch")]


def read_capped(path: Path, cap: int) -> bytes:
    with path.open("rb") as f:
        data = f.read(cap + 1)
    if len(data) > cap:
        raise DataError(f"{path.name} is larger than {cap} bytes")
    return data


def load_tasks(data_path: Path, manifest_path: Path | None = None) -> list[Task]:
    """Read and validate oracle-data.json.gz (and check it against the manifest, if given)."""
    try:
        raw = read_capped(data_path, MAX_DATA_GZ)
        manifest = None
        if manifest_path is not None:
            manifest = json.loads(read_capped(manifest_path, MAX_MANIFEST).decode("utf-8"))
            if (not isinstance(manifest, dict)
                    or manifest.get("data_sha256") != hashlib.sha256(raw).hexdigest()):
                raise DataError(f"{data_path.name} does not match the data_sha256 of "
                                f"{manifest_path.name}")
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
            text = gz.read(MAX_DATA_JSON + 1)
        if len(text) > MAX_DATA_JSON:
            raise DataError(f"{data_path.name} decompresses to more than {MAX_DATA_JSON} bytes")
        doc = json.loads(text.decode("utf-8"))
    except DataError:
        raise
    except (OSError, EOFError, zlib.error, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise DataError(f"cannot read the oracle data: {type(exc).__name__}: {str(exc)[:200]}") from None
    if not isinstance(doc, dict) or doc.get("schema") != DATA_SCHEMA:
        raise DataError(f"{data_path.name} is not {DATA_SCHEMA} data")
    tasks = validate_tasks(doc.get("instances"))
    if manifest is not None:
        listed = [e.get("instance_id") if isinstance(e, dict) else None
                  for e in manifest.get("instances") or []]
        if listed != [t.instance_id for t in tasks]:
            raise DataError(f"the tasks in {data_path.name} differ from those of the manifest")
    return tasks


class Oracle:
    """The tasks, their normalized problem statements and their reply plans (read-only)."""

    def __init__(self, tasks: list[Task]):
        self.tasks = list(tasks)
        self._problems = [(t.instance_id, normalize(t.problem_statement)) for t in self.tasks]
        self._plans = {t.instance_id: plan(t.patch) for t in self.tasks}
        self.warnings = [f"{t.instance_id}: the gold patch has CR characters; the agent environment "
                         "reads command output in text mode (CR becomes LF), so its submission "
                         "will differ from the gold patch" for t in self.tasks if "\r" in t.patch]

    def match(self, text: str) -> str | None:
        """The one task whose problem statement is in `text`, else None (none or ambiguous)."""
        haystack = normalize(text)
        hits = [iid for iid, problem in self._problems if problem in haystack]
        return hits[0] if len(hits) == 1 else None

    def plan_for(self, instance_id: str | None) -> list[Step]:
        return self._plans.get(instance_id or "", UNKNOWN_PLAN)


# ----------------------------------------------------------------------------- replies
class Reply(NamedTuple):
    status: int
    payload: dict
    instance: str = "-"          # for the log line: a validated id, "unknown" or "-"
    step: int | None = None


def error_body(message: str, kind: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}


def content_text(content) -> str:
    """Message content as text: a string, or the text parts of a content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict)
                       and p.get("type") == "text" and isinstance(p.get("text"), str))
    return ""


def first_user_text(messages: list) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return content_text(m.get("content"))
    return ""


def pick_tool(tools) -> tuple[str, str] | None:
    """(function name, its one required string parameter) to call, from the request's tools.

    No tools: ("bash", "command"), mini-swe-agent's BASH_TOOL. Tools without a usable function
    (exactly one required parameter, of type string): None. Prefers a function named bash.
    """
    if tools is None or tools == []:
        return ("bash", "command")
    if not isinstance(tools, list):
        return None
    usable = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        params = params if isinstance(params, dict) else {}
        required, props = params.get("required"), params.get("properties")
        if not (isinstance(required, list) and len(required) == 1 and isinstance(required[0], str)
                and isinstance(props, dict) and isinstance(props.get(required[0]), dict)):
            continue
        if (props[required[0]].get("type") == "string" and TOOL_RE.fullmatch(fn["name"])
                and TOOL_RE.fullmatch(required[0])):
            usable.append((fn["name"], required[0]))
    return next((u for u in usable if u[0] == "bash"), usable[0] if usable else None)


def answer(oracle: Oracle, request: dict, model_name: str) -> Reply:
    """The reply to one POST /v1/chat/completions body (a JSON object)."""
    if request.get("stream") not in (None, False):
        return Reply(400, error_body("stream=true is not supported: the SWE oracle sends one "
                                     "complete JSON reply; send stream=false"))
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return Reply(400, error_body("messages must be a non-empty list"))
    tool = pick_tool(request.get("tools"))
    if tool is None:
        return Reply(400, error_body("tools: no function tool with exactly one required string "
                                     "parameter"))
    step = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
    instance = oracle.match(first_user_text(messages))
    steps = oracle.plan_for(instance)
    index = min(step, len(steps) - 1)
    command, note = steps[index]
    content = (f"THOUGHT: SWE-bench oracle (public gold patch; grading-harness test, not a model). "
               f"Step {index + 1}/{len(steps)}: {note}{' (again)' if step > index else ''}.")
    arguments = json.dumps({tool[1]: command})
    prompt_chars = sum(len(content_text(m.get("content"))) for m in messages if isinstance(m, dict))
    prompt_tokens, completion_tokens = prompt_chars // 4 + 1, (len(content) + len(arguments)) // 4 + 1
    payload = {
        "id": f"chatcmpl-{secrets.token_hex(16)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "system_fingerprint": None,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": [{
                "id": f"call_{secrets.token_hex(12)}", "type": "function",
                "function": {"name": tool[0], "arguments": arguments}}]},
            "finish_reason": "tool_calls",
        }],
        # Rough sizes at about 4 characters per token; nothing here runs a tokenizer.
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
    return Reply(200, payload, instance or "unknown", step)


# ----------------------------------------------------------------------------- HTTP
def loopback_family(host: str) -> int:
    """The socket family for a literal loopback address (127.0.0.0/8 or ::1); ValueError else."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"refusing to listen on {printable(host)}: give a literal loopback "
                         "address such as 127.0.0.1") from None
    if ip.version == 4 and ip.is_loopback:
        return socket.AF_INET
    if ip == ipaddress.IPv6Address("::1"):
        return socket.AF_INET6
    raise ValueError(f"refusing to listen on {printable(host)}: loopback only (127.0.0.0/8 or ::1)")


def _busy_response() -> bytes:
    body = json.dumps(error_body("too many open connections; retry", "server_busy")).encode()
    head = ["HTTP/1.1 503 Service Unavailable", "Content-Type: application/json",
            f"Content-Length: {len(body)}", *(f"{k}: {v}" for k, v in SECURITY_HEADERS),
            "Retry-After: 1", "Connection: close"]
    return ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body


class OracleServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer on a loopback address with a bounded number of open connections."""

    daemon_threads = True
    allow_reuse_port = False             # no second socket on the same port (SO_REUSEPORT)
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], oracle: Oracle, model_name: str = MODEL_NAME, *,
                 max_connections: int = MAX_CONNECTIONS, socket_timeout: float = SOCKET_TIMEOUT,
                 slot_wait: float = SLOT_WAIT):
        self.address_family = loopback_family(address[0])      # before any socket exists
        if not NAME_RE.fullmatch(model_name or ""):
            raise ValueError("invalid --model-name")
        self.oracle, self.model_name = oracle, model_name
        self.socket_timeout, self.slot_wait = socket_timeout, slot_wait
        self.created = int(time.time())
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, OracleHandler)

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)               # without HTTPServer's getfqdn()
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(timeout=self.slot_wait):
            log("busy: a connection got 503 (all connection slots in use)")
            try:
                request.sendall(_busy_response())
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)    # starts the handler thread
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address) -> None:
        # No traceback: it could carry request data. The exception type is enough here.
        log(f"connection error: {type(sys.exc_info()[1]).__name__}", err=True)


class OracleHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "swe-oracle"
    sys_version = ""

    def setup(self) -> None:
        self.timeout = self.server.socket_timeout              # read by StreamRequestHandler
        super().setup()

    def version_string(self) -> str:
        return self.server_version                             # no Python version in the header

    def parse_request(self) -> bool:
        self._body_read = False            # one handler serves all requests of a keep-alive connection
        return super().parse_request()

    # The base class logs request lines and error details; this server writes one sanitized line
    # per response instead (method, path, status, instance, step), never headers or bodies.
    def log_message(self, format, *args) -> None:              # noqa: A002 (base-class signature)
        pass

    def log_request(self, code="-", size="-") -> None:
        pass

    def log_error(self, format, *args) -> None:                # noqa: A002
        log("request error: " + ("timed out" if "timed out" in format else "malformed"), err=True)

    def send_error(self, code, message=None, explain=None) -> None:
        """Errors the base class finds (bad request line, header limits, unknown method): JSON."""
        try:
            phrase = HTTPStatus(code).phrase.lower()
        except ValueError:
            phrase = "error"
        self._reply(code, error_body(phrase))

    def _reply(self, status: int, payload: dict, *, allow: str | None = None,
               instance: str = "-", step: int | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("ascii")
        head = self.command == "HEAD"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0" if head else str(len(body)))
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        if allow:
            self.send_header("Allow", allow)
        if status >= 400 or self._unread_body():
            # Errors may leave an unread body on the socket: never reuse such a connection.
            self.send_header("Connection", "close")            # also sets close_connection
        self.end_headers()
        if not head:
            self.wfile.write(body)
        path = (getattr(self, "path", "") or "").split("?", 1)[0]
        log(f"{printable(self.command, 16)} {printable(path)} {status} instance={instance} "
            f"step={'-' if step is None else step}")

    def _unread_body(self) -> bool:
        headers = getattr(self, "headers", None) or {}
        if getattr(self, "_body_read", False):
            return False
        return "Transfer-Encoding" in headers or headers.get("Content-Length", "0").strip() not in ("", "0")

    def _route(self) -> str:
        return (self.path or "").split("?", 1)[0]

    def _no_route(self, path: str) -> None:
        allowed = ROUTES.get(path)
        if allowed:
            self._reply(405, error_body(f"method not allowed; use {allowed}"), allow=allowed)
        else:
            self._reply(404, error_body("not found"))

    def do_GET(self) -> None:                                  # noqa: N802
        path = self._route()
        if path == "/health":
            self._reply(200, {"status": "ok"})
        elif path == "/v1/models":
            self._reply(200, {"object": "list", "data": [{
                "id": self.server.model_name, "object": "model", "created": self.server.created,
                "owned_by": "kimi-swe-oracle"}]})
        else:
            self._no_route(path)

    def do_POST(self) -> None:                                 # noqa: N802
        path = self._route()
        if path != "/v1/chat/completions":
            return self._no_route(path)
        body = self._read_body()
        if body is None:
            return
        try:
            request = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return self._reply(400, error_body("the request body is not UTF-8 JSON"))
        if not isinstance(request, dict):
            return self._reply(400, error_body("the request body must be a JSON object"))
        try:
            reply = answer(self.server.oracle, request, self.server.model_name)
        except Exception as exc:                               # a bug: report it without request data
            log(f"internal error: {type(exc).__name__}", err=True)
            return self._reply(500, error_body("internal error", "server_error"))
        self._reply(reply.status, reply.payload, instance=reply.instance, step=reply.step)

    def _not_allowed(self) -> None:
        self._no_route(self._route())

    do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = _not_allowed

    def _read_body(self) -> bytes | None:
        """The request body, or None after replying 411/413/400 (and closing the connection)."""
        if "Transfer-Encoding" in self.headers:
            self._reply(411, error_body("send the body with Content-Length, not Transfer-Encoding"))
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            self._reply(411, error_body("Content-Length is required"))
            return None
        if len(lengths) != 1 or not LENGTH_RE.fullmatch(lengths[0].strip()):
            self._reply(400, error_body("invalid Content-Length"))
            return None
        size = int(lengths[0].strip())
        if size > MAX_BODY:
            self._reply(413, error_body(f"the request body is larger than {MAX_BODY} bytes"))
            return None
        body = self.rfile.read(size)
        self._body_read = True
        if len(body) != size:
            self._reply(400, error_body("the request body is shorter than its Content-Length"))
            return None
        return body


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Test-only SWE-bench oracle endpoint: answers each "
                                             "task with its public gold patch.")
    ap.add_argument("--data", required=True, help="oracle-data.json.gz from oracle_data.py")
    ap.add_argument("--manifest", help="oracle-manifest.json: refuse data with another sha256")
    ap.add_argument("--host", default="127.0.0.1", help="a literal loopback address")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--model-name", default=MODEL_NAME, help="the model id served by /v1/models")
    args = ap.parse_args(argv)
    try:
        loopback_family(args.host)                              # refuse before reading anything
        if not 0 <= args.port <= 65535:
            raise ValueError("--port must be 0 to 65535")
        oracle = Oracle(load_tasks(Path(args.data), Path(args.manifest) if args.manifest else None))
        httpd = OracleServer((args.host, args.port), oracle, args.model_name)
    except (OSError, ValueError) as exc:
        log(f"ERROR: {exc}", err=True)
        return 1

    def stop(signum, _frame) -> None:
        log(f"signal {signum}: stopping")
        threading.Thread(target=httpd.shutdown, daemon=True).start()   # not from serve_forever's thread

    signal.signal(signal.SIGTERM, stop)                         # PID 1 in the pod: no default action
    signal.signal(signal.SIGINT, stop)
    host, port = httpd.server_address[:2]
    url_host = f"[{host}]" if ":" in host else host
    log(f"serving {len(oracle.tasks)} SWE-bench tasks with their public gold patches as model "
        f"{args.model_name} at http://{url_host}:{port} (grading-harness test only, not a model)")
    for warning in oracle.warnings:
        log(f"warning: {warning}", err=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
