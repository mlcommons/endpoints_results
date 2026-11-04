#!/usr/bin/env python3
"""Fetch the public SWE-bench Verified rows that the test-only oracle serves (oracle_server.py).

Pages through the Hugging Face datasets-server rows API (HTTPS GET, no token, no redirects) and
keeps instance_id, problem_statement and patch (the public gold patch) of the first --count rows
of princeton-nlp/SWE-bench_Verified, split test: the rows the SWE service runs. The first 10
must be the pilot's 10 instances, in order. The rows pass the server's own loader rules
(oracle_server.validate_tasks), then go next to this file:

  oracle-data.json.gz    deterministic: sorted keys, ASCII JSON, gzip mtime 0 (same rows, same bytes)
  oracle-manifest.json   source, split, count, fetch time, sha256 of each patch and of the .gz

The pilot runs the first 10 rows (the default). The production rehearsal uses --count 200 (the
drop's task list; the first 10 are the pilot's, so the pilot still works on it) with
swe_pilot.py --num-instances 200. Next steps: swe_pilot.py configmaps, then
swe_pilot.py model --oracle. Test only: never a submission.
"""
import argparse
import base64
import gzip
import hashlib
import http.client
import importlib.util
import io
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True       # no __pycache__ next to the sources when loading the server
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("oracle_server", HERE / "oracle_server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)

API = "https://datasets-server.huggingface.co/rows"
API_HOST = "datasets-server.huggingface.co"
DATASET, CONFIG, SPLIT = "princeton-nlp/SWE-bench_Verified", "default", "test"
PAGE = 100                           # the API's largest page
MAX_RESPONSE = 32 * 1024 * 1024      # bytes per page
TIMEOUT = 60                         # seconds per request
ATTEMPTS = 4
MAX_GZ = 700 * 1024                  # keeps ConfigMap kimi-swe-oracle well below its 1 MiB limit
DATA_NAME, MANIFEST_NAME = "oracle-data.json.gz", "oracle-manifest.json"
MANIFEST_SCHEMA = "kimi-swe-oracle-manifest/1"
FIELDS = ("instance_id", "problem_statement", "patch")
PILOT_IDS = ("astropy__astropy-12907", "astropy__astropy-13033", "astropy__astropy-13236",
             "astropy__astropy-13398", "astropy__astropy-13453", "astropy__astropy-13579",
             "astropy__astropy-13977", "astropy__astropy-14096", "astropy__astropy-14182",
             "astropy__astropy-14309")


class FetchError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect not followed", headers, fp)


def page_url(offset: int, length: int) -> str:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                    "offset": offset, "length": length})
    return f"{API}?{query}"


def fetch_json(url: str) -> dict:
    """GET one page from the datasets server: HTTPS, verified TLS, bounded size and time."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname != API_HOST:
        raise FetchError(f"refusing to fetch from anywhere but https://{API_HOST}")
    opener = urllib.request.build_opener(
        _NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "kimi-swe-oracle-data/1"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with opener.open(request, timeout=TIMEOUT) as resp:
                body = resp.read(MAX_RESPONSE + 1)
            if len(body) > MAX_RESPONSE:
                raise FetchError(f"a page from {API_HOST} is larger than {MAX_RESPONSE} bytes")
            doc = json.loads(body.decode("utf-8"))
            if not isinstance(doc, dict):
                raise FetchError(f"{API_HOST} did not return a JSON object")
            return doc
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code not in (429, 500, 502, 503, 504) or attempt == ATTEMPTS:
                raise FetchError(f"HTTP {exc.code} from {API_HOST}") from None
        except (OSError, http.client.HTTPException) as exc:     # URLError, timeouts, TLS, short reads
            if attempt == ATTEMPTS:
                raise FetchError(f"cannot reach {API_HOST}: {type(exc).__name__}: {exc}") from None
        except (UnicodeDecodeError, ValueError):
            raise FetchError(f"{API_HOST} returned a page that is not UTF-8 JSON") from None
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_rows(count: int, fetch=fetch_json) -> list[dict]:
    """instance_id, problem_statement and patch of rows 0..count-1, in order."""
    rows: list[dict] = []
    page = PAGE
    while len(rows) < count:
        offset = len(rows)
        length = min(page, count - offset)
        doc = fetch(page_url(offset, length))
        items, total = doc.get("rows"), doc.get("num_rows_total")
        if not isinstance(items, list) or not isinstance(total, int):
            raise FetchError("unexpected page shape (no rows / num_rows_total)")
        if total < count:
            raise FetchError(f"{DATASET} {SPLIT} has {total} rows, fewer than --count {count}")
        if len(items) != length:
            raise FetchError(f"asked for {length} rows at offset {offset}, got {len(items)}")
        truncated = any(set(item.get("truncated_cells") or []) & set(FIELDS)
                        for item in items if isinstance(item, dict))
        if truncated:                    # the API cuts cells of large pages: ask for fewer rows
            if length == 1:
                raise FetchError(f"row {offset}: the API truncates a needed cell even alone")
            page = max(1, length // 2)
            continue
        for k, item in enumerate(items):
            if (not isinstance(item, dict) or item.get("row_idx") != offset + k
                    or not isinstance(item.get("row"), dict)):
                raise FetchError(f"row {offset + k}: unexpected row_idx or shape")
            rows.append({name: item["row"].get(name) for name in FIELDS})
    return rows


def check_pilot_prefix(tasks: list) -> None:
    want = PILOT_IDS[:min(len(tasks), len(PILOT_IDS))]
    got = tuple(t.instance_id for t in tasks[:len(want)])
    if got != want:
        raise srv.DataError(f"the first rows are {', '.join(got)}; the pilot runs "
                            f"{', '.join(want)} (in that order)")


def build_data(tasks: list) -> bytes:
    """oracle-data.json.gz: the same tasks always give the same bytes."""
    doc = {"schema": srv.DATA_SCHEMA, "dataset": DATASET, "config": CONFIG, "split": SPLIT,
           "instances": [t._asdict() for t in tasks]}
    raw = json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(tasks: list, data: bytes, fetched_at: str) -> dict:
    entries = []
    for t in tasks:
        patch = t.patch.encode("utf-8")
        entries.append({"instance_id": t.instance_id, "patch_sha256": sha256(patch),
                        "patch_bytes": len(patch), "patch_base64_chars": len(base64.b64encode(patch)),
                        "plan_steps": len(srv.plan(t.patch)), "patch_has_cr": "\r" in t.patch})
    return {"schema": MANIFEST_SCHEMA,
            "purpose": "test only: public SWE-bench gold patches for the grading-harness oracle; "
                       "never a submission",
            "source": API, "dataset": DATASET, "config": CONFIG, "split": SPLIT,
            "count": len(tasks), "fetched_at": fetched_at, "data_file": DATA_NAME,
            "data_sha256": sha256(data), "data_bytes": len(data), "chunk_chars": srv.CHUNK,
            "max_patch_base64_chars": max(e["patch_base64_chars"] for e in entries),
            "instances": entries}


def write_file(path: Path, data: bytes) -> None:
    """Atomic replace, mode 0644: public data that goes into a ConfigMap."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None, *, fetch=fetch_json, out_dir: Path = HERE) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=len(PILOT_IDS),
                    help=f"first rows to keep (default {len(PILOT_IDS)}; production 200)")
    args = ap.parse_args(argv)
    if not 1 <= args.count <= srv.MAX_TASKS:
        ap.error(f"--count must be 1 to {srv.MAX_TASKS}")
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        tasks = srv.validate_tasks(fetch_rows(args.count, fetch))
        check_pilot_prefix(tasks)
        data = build_data(tasks)
        if len(data) > MAX_GZ:
            raise srv.DataError(f"{DATA_NAME} would be {len(data)} bytes, above {MAX_GZ}")
        manifest = build_manifest(tasks, data, fetched_at)
        text = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
        with tempfile.TemporaryDirectory(prefix="oracle-data-") as tmp:     # round trip first
            (Path(tmp) / DATA_NAME).write_bytes(data)
            (Path(tmp) / MANIFEST_NAME).write_bytes(text)
            if srv.load_tasks(Path(tmp) / DATA_NAME, Path(tmp) / MANIFEST_NAME) != tasks:
                raise srv.DataError("the data does not read back to the same tasks")
    except (FetchError, srv.DataError) as exc:
        print(f"[oracle-data] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    write_file(out_dir / DATA_NAME, data)
    write_file(out_dir / MANIFEST_NAME, text)
    print(f"[oracle-data] {len(tasks)} rows of {DATASET} ({SPLIT}) fetched at {fetched_at}")
    for e in manifest["instances"]:
        print(f"  {e['instance_id']:<28} patch {e['patch_bytes']:>6} B  base64 "
              f"{e['patch_base64_chars']:>6}  steps {e['plan_steps']}"
              + ("  WARNING: CR in the patch (the agent environment turns CR into LF)"
                 if e["patch_has_cr"] else ""))
    print(f"[oracle-data] wrote {out_dir / DATA_NAME} ({len(data)} B, sha256 {manifest['data_sha256']})"
          f" and {out_dir / MANIFEST_NAME}; longest base64 {manifest['max_patch_base64_chars']} chars")
    print("[oracle-data] next: swe_pilot.py configmaps, then swe_pilot.py model --oracle --node NODE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
