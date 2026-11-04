#\!/usr/bin/env python3
"""
Generate a trtllm-serve disaggregated master/frontend server config YAML.

Accepts comma-separated URL lists for context and generation servers,
combines them with the frontend's hostname and port, and writes a valid
server_config.yaml.

Optionally appends "hostname:port" to a FRONTEND_URLS registry file so the
MLPerf harness can discover all live frontend endpoints.

Usage:
    python3 generate_master_yaml.py \
        --hostname <ip> \
        --port 8000 \
        --ctx-urls <ip>:8337,<ip>:8336 \
        --gen-urls <ip>:8338 \
        --output    /path/to/server_config.yaml \
        [--register-url-to /path/to/frontend_urls.txt]
"""
import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hostname", required=True, help="IP/hostname this frontend listens on")
    p.add_argument("--port", required=True, type=int, help="Port this frontend listens on")
    p.add_argument("--ctx-urls", required=True,
                   help="Comma-separated list of context server URLs, e.g. <ip>:8337,<ip>:8337")
    p.add_argument("--gen-urls", required=True,
                   help="Comma-separated list of generation server URLs, e.g. <ip>:8338")
    p.add_argument("--output", required=True, type=Path, help="Destination server_config.yaml")
    p.add_argument("--register-url-to", type=Path, default=None,
                   help="Append 'hostname:port' to this file (e.g. frontend_urls.txt)")
    p.add_argument("--ctx-router-type", default="round_robin",
                   choices=["round_robin", "load_balancing", "kv_cache_aware", "conversation"],
                   help="Frontend routing policy across CONTEXT servers (default: round_robin, "
                        "which omits the router section for byte-identical legacy output)")
    p.add_argument("--gen-router-type", default="round_robin",
                   choices=["round_robin", "load_balancing", "kv_cache_aware", "conversation"],
                   help="Frontend routing policy across GENERATION servers (default: round_robin)")
    p.add_argument("--router-use-tokens", default="0", choices=["0", "1"],
                   help="For load_balancing: balance by active TOKENS (1) instead of active "
                        "request count (0). Ignored for other router types.")
    return p.parse_args()


def parse_urls(raw: str) -> list[str]:
    """Split a comma-separated URL string, stripping whitespace and empty entries."""
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        print("ERROR: no URLs provided", file=sys.stderr)
        sys.exit(1)
    return urls


def make_router_config(router_type: str, use_tokens: bool) -> dict | None:
    """Router section for a server group; None for round_robin (trtllm-serve default),
    keeping the generated YAML byte-identical to the pre-flag output."""
    if router_type == "round_robin":
        return None
    router: dict = {"type": router_type}
    if router_type == "load_balancing" and use_tokens:
        # extra keys inside `router:` are forwarded as router kwargs
        # (extract_router_config in tensorrt_llm/llmapi/disagg_utils.py)
        router["use_tokens"] = True
    return router


def make_yaml_config(hostname: str, port: int, ctx_urls: list[str], gen_urls: list[str],
                     ctx_router: dict | None = None, gen_router: dict | None = None) -> dict:
    """Build the server config dict for a trtllm-serve disaggregated frontend."""
    cfg = {
        "hostname": hostname,
        "port": port,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": len(ctx_urls),
            "urls": ctx_urls,
        },
        "generation_servers": {
            "num_instances": len(gen_urls),
            "urls": gen_urls,
        },
    }
    if ctx_router:
        cfg["context_servers"]["router"] = ctx_router
    if gen_router:
        cfg["generation_servers"]["router"] = gen_router
    return cfg


def main() -> None:
    args = parse_args()

    ctx_urls = parse_urls(args.ctx_urls)
    gen_urls = parse_urls(args.gen_urls)

    use_tokens = args.router_use_tokens == "1"
    config = make_yaml_config(
        args.hostname, args.port, ctx_urls, gen_urls,
        ctx_router=make_router_config(args.ctx_router_type, use_tokens),
        gen_router=make_router_config(args.gen_router_type, use_tokens),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Written: {args.output}")
    with open(args.output) as f:
        print(f.read())

    if args.register_url_to:
        url = f"{args.hostname}:{args.port}"
        with open(args.register_url_to, "a") as f:
            f.write(f"{url}\n")
        print(f"Registered URL: {url} → {args.register_url_to}")


if __name__ == "__main__":
    main()
