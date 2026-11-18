#!/usr/bin/env python3
"""Pin a performance client to Dynamo's server-generated client CPU mask."""
import argparse
import json
import os
from pathlib import Path

import dynamo_cpu_affinity as affinity


def fail(message):
    raise SystemExit(f"colocated client: {message}")


def client_mask(path):
    try:
        plan = json.loads(path.read_text())
        client = affinity.cpus(plan["client_cpus"])
        used = affinity.cpus(plan["frontend_cpus"])
        for worker in plan["workers"]:
            worker_cpus = affinity.cpus(worker["cpus"])
            if used & worker_cpus:
                fail("topology frontend and worker masks overlap")
            used |= worker_cpus
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        fail(f"cannot read topology {path}: {error}")
    except SystemExit as error:
        fail(str(error))
    if not client:
        fail("topology client mask is empty")
    if client & used:
        fail("topology client mask overlaps frontend or worker mask")
    return client


def configure(source):
    try:
        import yaml
    except ImportError as error:
        fail(f"PyYAML is required to update {source}: {error}")
    try:
        config = yaml.safe_load(source.read_text())
        config["settings"]["client"]["num_workers"] = 8
        # Endpoints otherwise replaces taskset with a host-wide /sys mask.
        config["enable_cpu_affinity"] = False
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        fail(f"cannot update client configuration {source}: {error}")

    # GB client.sh reads CLIENT_CONFIG then makes its normal task-output copy.
    # Keep that SFlow artifact intact. VR already supplies RUN_DIR/client.yaml,
    # a run-specific copy, so it may be updated in place.
    if os.environ.get("CLIENT_CONFIG") == str(source):
        try:
            output = Path(os.environ["SFLOW_TASK_OUTPUT_DIR"]) / "colocated-client.yaml"
        except KeyError as error:
            fail(f"cannot copy CLIENT_CONFIG: {error}")
        os.environ["CLIENT_CONFIG"] = str(output)
    else:
        output = source
    try:
        output.write_text(yaml.safe_dump(config, sort_keys=False))
    except OSError as error:
        fail(f"cannot write client configuration {output}: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default=os.environ.get("CLIENT_TOPOLOGY", ""))
    parser.add_argument("--client-config", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if os.environ.get("MODE") != "perf":
        fail("only MODE=perf is supported while SWE service placement is separate")
    if not args.topology or not args.command:
        fail("set CLIENT_TOPOLOGY and provide a client command after --")

    mask = client_mask(Path(args.topology))
    allowed = os.sched_getaffinity(0)
    if not mask <= allowed:
        fail(f"client mask {affinity.rendered(mask)} is outside Slurm CPUs "
             f"{affinity.rendered(allowed)}")
    configure(args.client_config)
    os.sched_setaffinity(0, mask)
    if os.sched_getaffinity(0) != mask:
        fail("kernel did not apply client mask")
    os.environ["CPUSET"] = affinity.rendered(mask)
    print(json.dumps({"colocated_client": {"cpus": os.environ["CPUSET"],
                                            "topology": args.topology,
                                            "workers": 8}}, sort_keys=True), flush=True)
    os.execvpe(args.command[0], args.command, os.environ)


if __name__ == "__main__":
    main()
