"""Assemble complete submission metadata from disclosures and observed facts."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from urllib.parse import quote

import yaml

if __package__:
    from .submission_requirements import APPROVED_REVISIONS
    from .checker_runtime import load_checker_policy
else:
    from submission_requirements import APPROVED_REVISIONS
    from checker_runtime import load_checker_policy

ROOT = Path(__file__).resolve().parent.parent
DATASET_LINK = (
    "https://endpoints.mlcommons-storage.org/metadata/"
    "mlperf_agentic_inference_dataset.uri"
)
DATASET_NAME = "MLPerf Agentic Inference"
DATASET_ID = "agentic_combined"
SHARED_SRC = "src/endpoints_agentic"
SHARED_DOCS = "docs"
POINT_DIMENSIONS = {
    "tp_size",
    "replica_count",
    "prefill_tp_size",
    "prefill_replica_count",
    "decode_tp_size",
    "decode_replica_count",
}
ACCELERATOR_FIELDS = {
    "accelerator_model_name",
    "accelerators_per_node",
    "accelerator_memory_capacity",
    "accelerator_memory_type",
    "accelerator_interconnect",
    "accelerator_host_interconnect",
}
AVAILABILITY_ALIASES = {
    "availability_status",
    "system_availability_status",
}
# The pinned collector still emits legacy point disclosures and its own
# provenance. Recognize these explicitly, but do not import them as host facts.
COLLECTOR_EXTRA_FIELDS = {
    "system_category", "model_id", "model_name", "model_precision",
    "link_to_model", "link_to_model_transformation", "model_notes",
    "dataset_id", "dataset_name", "input_token_average", "output_token_average",
    "dataset_type", "dataset_link", "mlc_scripts_version",
}
FACT_SYSTEM_FIELDS = (
    "system_node_ensemble_count",
    "system_node_ensemble_total",
    "serving_framework",
    "node_config",
)
FACT_NODE_FIELDS = (
    "system_node_ensemble_id",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_processor_vcpu_count",
    "host_memory_capacity",
    "host_memory_configuration",
    "accelerator_info",
    "host_network_card_count",
    "host_networking",
    "host_storage_capacity",
    "host_storage_type",
    "other_hardware",
    "hw_notes",
    "inference_backend",
    "driver",
    "operating_system",
    "filesystem",
    "container_link",
    "other_software_stack",
    "sw_notes",
)
LEGACY_ACCELERATOR_FIELDS = ACCELERATOR_FIELDS - {"accelerator_info"}
# These factual fields are emitted by the pinned collector but have no
# destination in the selected Endpoints schema. Preserve them in raw JSON.
COLLECTOR_EXTRA_NODE_FIELDS = {
    "host_processor_frequency", "host_processor_caches",
    "host_processor_interconnect", "accelerator_frequency",
    "accelerator_memory_configuration", "accelerator_on-chip_memories",
    "accelerator_interconnect_topology",
}


class MetadataDraftError(ValueError):
    """Submission metadata could not be safely constructed."""


def _schema_models():
    try:
        from submission_checker.models.file import (
            ConfigSummary,
            DatasetAccuracyScores,
            NodeType,
            SystemDescription,
        )
    except ImportError as exc:
        raise MetadataDraftError(
            "The selected submission schema is unavailable; run "
            "prepare_submission.sh through init_submission_metadata.sh."
        ) from exc
    return SystemDescription, NodeType, ConfigSummary, DatasetAccuracyScores


def _known_fields(model) -> set[str]:
    return set(model.model_fields)


def _reject_unknown_fields(value: dict, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise MetadataDraftError(
            f"{label} has unknown fields: {sorted(unknown, key=str)}"
        )


def _schema_system_defaults() -> dict:
    return {
        "system_name": "Draft system",
        "shortened_system_name": "Draft system",
        # These values only let the current checker validate partial facts
        # imported from the native collector. They are never written as
        # submitter disclosures.
        "publication_status": "RDI",
        "max_supported_concurrency": 1,
        "system_size": "Unknown",
        "system_node_ensemble_count": 1,
        "system_node_ensemble_total": 1,
        "node_types": [],
        "division": "Standardized",
    }


def _mlcflow_pins(root: Path = ROOT) -> tuple[str, str]:
    try:
        text = (root / "configs/mlcflow.conf").read_text()
        version = re.search(r"^MLCFLOW_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$", text, re.M).group(1)
        revision = re.search(r"^MLC_AUTOMATIONS_REF=([0-9a-f]{40})$", text, re.M).group(1)
    except (OSError, AttributeError) as exc:
        raise MetadataDraftError("Missing or invalid configs/mlcflow.conf.") from exc
    return version, revision


def _validate_collector_provenance(value, revision: str) -> None:
    if not isinstance(value, dict) or (
        value.get("source") != "git"
        or value.get("commit") != revision
        or value.get("dirty") is not False
    ):
        raise MetadataDraftError(
            "System description must contain mlc_scripts_version from the "
            f"clean pinned automation Git checkout ({revision})."
        )
    if "consistent" in value and value["consistent"] is not True:
        raise MetadataDraftError("Collector node revisions are not consistent.")
    if "nodes" in value:
        if not isinstance(value["nodes"], dict) or not value["nodes"]:
            raise MetadataDraftError("Collector node provenance must be non-empty.")
        for node_version in value["nodes"].values():
            _validate_collector_provenance(node_version, revision)


def _normalized_imported_system(
    path: Path, *, revision: str | None = None
) -> tuple[dict, str]:
    if not path.is_absolute():
        raise MetadataDraftError("SUBMISSION_SYSTEM_DESCRIPTION must be absolute.")
    raw_bytes: bytes
    try:
        raw_bytes = path.read_bytes()
        source = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataDraftError(
            f"Cannot read SUBMISSION_SYSTEM_DESCRIPTION at {path}: {exc}"
        ) from exc
    if not isinstance(source, dict):
        raise MetadataDraftError(
            "SUBMISSION_SYSTEM_DESCRIPTION must contain a JSON object."
        )
    # Accept a single explicit wrapper used by some _endpoints exports.
    if "system" in source:
        if set(source) != {"system"} or not isinstance(source["system"], dict):
            raise MetadataDraftError(
                "A wrapped system description may contain only a system object."
            )
        source = source["system"]

    _validate_collector_provenance(
        source.get("mlc_scripts_version"),
        revision or _mlcflow_pins()[1],
    )
    SystemDescription, NodeType, ConfigSummary, DatasetAccuracyScores = _schema_models()
    allowed_system = _known_fields(SystemDescription) | AVAILABILITY_ALIASES | COLLECTOR_EXTRA_FIELDS
    _reject_unknown_fields(source, allowed_system, "system description")
    summary = source.get("config_summary")
    if isinstance(summary, dict):
        _reject_unknown_fields(
            summary, _known_fields(ConfigSummary),
            "system description config_summary",
        )
    scores = source.get("measured_accuracy_score")
    if isinstance(scores, dict):
        for dataset_name, dataset_scores in scores.items():
            if isinstance(dataset_scores, dict):
                _reject_unknown_fields(
                    dataset_scores,
                    _known_fields(DatasetAccuracyScores),
                    f"system description measured_accuracy_score[{dataset_name!r}]",
                )

    def detected(value):
        if isinstance(value, str) and (
            not value.strip() or value.strip() == "N/A"
            or re.fullmatch(r"Insert .* here", value.strip())
        ):
            return None
        return value

    nodes = source.get("node_types", [])
    if not isinstance(nodes, list):
        raise MetadataDraftError("system description node_types must be a list.")
    allowed_node = (
        _known_fields(NodeType) | LEGACY_ACCELERATOR_FIELDS | COLLECTOR_EXTRA_NODE_FIELDS
    )
    normalized_nodes = []
    for index, node in enumerate(nodes):
        label = f"system description node_types[{index}]"
        if not isinstance(node, dict):
            raise MetadataDraftError(f"{label} must be an object.")
        _reject_unknown_fields(node, allowed_node, label)
        accelerators = node.get("accelerator_info")
        if accelerators is not None:
            if not isinstance(accelerators, list):
                raise MetadataDraftError(f"{label}.accelerator_info must be a list.")
            for accelerator_index, accelerator in enumerate(accelerators):
                accelerator_label = (
                    f"{label}.accelerator_info[{accelerator_index}]"
                )
                if not isinstance(accelerator, dict):
                    raise MetadataDraftError(f"{accelerator_label} must be an object.")
                _reject_unknown_fields(
                    accelerator, ACCELERATOR_FIELDS, accelerator_label
                )
        node_facts = {
            name: detected(value) for name, value in node.items()
            if name in {*FACT_NODE_FIELDS, *LEGACY_ACCELERATOR_FIELDS}
        }
        candidate = {
            "host_memory_configuration": "",
            "host_network_card_count": "",
            "driver": "",
            "filesystem": "",
            # The schema requires one CPU count even for a partial draft. This
            # placeholder is only for validating source compatibility.
            "host_processor_vcpu_count": 1,
            **{name: value for name, value in node_facts.items() if value is not None},
        }
        try:
            normalized_nodes.append(NodeType.model_validate(candidate))
        except Exception as exc:
            raise MetadataDraftError(f"Invalid {label}: {exc}") from exc

    defaults = _schema_system_defaults()
    candidate_system = {
        **defaults,
        **{name: detected(source[name]) for name in FACT_SYSTEM_FIELDS
           if name in source and detected(source[name]) is not None},
        "node_types": [
            node.model_dump(mode="json") for node in normalized_nodes
        ],
    }
    try:
        normalized_system = SystemDescription.model_validate(candidate_system)
    except Exception as exc:
        raise MetadataDraftError(
            f"Invalid MLCommons system description: {exc}"
        ) from exc

    data = normalized_system.model_dump(mode="json")
    result = {name: data.get(name) for name in FACT_SYSTEM_FIELDS}
    result["node_types"] = []
    for index, normalized in enumerate(normalized_nodes):
        original = nodes[index]
        normalized_data = normalized.model_dump(mode="json")
        entry = {
            name: normalized_data.get(name)
            if name in original and detected(original[name]) is not None else None
            for name in FACT_NODE_FIELDS
        }
        if "accelerator_info" not in original:
            has_legacy_accelerator_fact = any(
                detected(original.get(name)) is not None
                for name in LEGACY_ACCELERATOR_FIELDS.intersection(original)
            )
            entry["accelerator_info"] = (
                normalized_data.get("accelerator_info")
                if has_legacy_accelerator_fact
                else None
            )
        # Whole-system and node cooling are intentionally left for disclosure.
        entry["cooling"] = None
        result["node_types"].append(entry)
    return result, hashlib.sha256(raw_bytes).hexdigest()


def _collect_mlcflow(
    *, capture_dir: Path, env: dict[str, str], pins: tuple[str, str], record: dict
) -> Path:
    if __package__:
        from .host_collector import collect
    else:
        from host_collector import collect
    return collect(
        root=Path(env.get("SUBMISSION_SOURCE_ROOT", str(ROOT))),
        capture_dir=capture_dir, env=env, pins=pins, record=record,
    )


def _load_config(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        config = yaml.safe_load(raw)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise MetadataDraftError(f"Cannot read campaign config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise MetadataDraftError("Campaign config must be a YAML mapping.")
    unknown = set(config) - {"points", "metadata", "disclosures"}
    if unknown:
        raise MetadataDraftError(
            f"Campaign config has unknown fields: {sorted(unknown, key=str)}"
        )
    metadata = config.setdefault("metadata", "submission_metadata.yaml")
    if not isinstance(metadata, str) or not metadata:
        raise MetadataDraftError("Campaign config metadata must be a non-empty path.")
    metadata_path = Path(metadata)
    if metadata_path.is_absolute():
        raise MetadataDraftError(
            "Campaign config metadata path must be relative to the config file."
        )
    disclosures = config.setdefault("disclosures", "submission_disclosures.yaml")
    if not isinstance(disclosures, str) or not disclosures.strip() or Path(disclosures).is_absolute():
        raise MetadataDraftError("Campaign disclosures must name a relative YAML path.")
    points = config.get("points")
    if not isinstance(points, list) or not points:
        raise MetadataDraftError("Campaign config points must be a non-empty list.")
    seen = set()
    for index, point in enumerate(points):
        label = f"points[{index}]"
        if not isinstance(point, dict):
            raise MetadataDraftError(f"{label} must be a mapping.")
        unknown = set(point) - {"concurrency", "stack_profile"} - POINT_DIMENSIONS
        if unknown:
            raise MetadataDraftError(
                f"{label} has unsupported fields: {sorted(unknown, key=str)}"
            )
        concurrency = point.get("concurrency")
        if type(concurrency) is not int or not 1 <= concurrency <= 2147483647:
            raise MetadataDraftError(
                f"{label}.concurrency must be a positive integer <= 2147483647."
            )
        if concurrency in seen:
            raise MetadataDraftError("Campaign point concurrencies must be unique.")
        seen.add(concurrency)
        profile = point.get("stack_profile")
        if profile is not None and (
            not isinstance(profile, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", profile)
        ):
            raise MetadataDraftError(f"{label}.stack_profile must be a profile name.")
        for name in POINT_DIMENSIONS.intersection(point):
            if type(point[name]) is not int or not 1 <= point[name] <= 2147483647:
                raise MetadataDraftError(f"{label}.{name} must be a positive integer.")
    return config, raw


HUMAN_REQUIRED_FIELDS = (
    "submitter_org_names", "submitter_contact", "submission_id", "system_name",
    "shortened_system_name", "publication_status", "system_size", "division",
)
HUMAN_SYSTEM_FIELDS = {
    *HUMAN_REQUIRED_FIELDS, "availability_url", "target_availability_date",
    "submission_date", "publish_date", "node_types", "node_config",
    "config_summary_notes",
    "system_node_ensemble_count", "system_node_ensemble_total",
}
HUMAN_POINT_FIELDS = {
    "model_notes", "link_to_model_transformation", "runtime_settings",
    "nodes_used", "dp_shortfall",
}
HUMAN_RUNTIME_FIELDS = {"min_duration_ms", "min_sample_count", "stream_all_chunks"}


def _populated(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not (
        value.strip().casefold() in {"todo", "tbd", "null", "n/a"}
        or re.fullmatch(r"Insert .* here", value.strip(), re.I)
    )


def load_disclosures(path: Path, *, raw: bytes | None = None) -> dict:
    """Validate manual inputs before collecting facts or preparing images."""
    try:
        value = yaml.safe_load(path.read_bytes() if raw is None else raw)
    except (OSError, yaml.YAMLError) as exc:
        raise MetadataDraftError(
            f"Populate submission disclosures at {path}; cannot read YAML: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MetadataDraftError(f"Populate {path}: disclosures must be a YAML mapping.")
    _reject_unknown_fields(value, {"system", "system_power", "point", "warmup"}, str(path))
    system = value.get("system")
    power = value.get("system_power")
    if not isinstance(system, dict):
        system = {}
    if not isinstance(power, dict):
        power = {}
    missing = [f"system.{name}" for name in HUMAN_REQUIRED_FIELDS if not _populated(system.get(name))]
    for name in ("cooling", "node_sets", "scale_out"):
        if power.get(name) is None:
            missing.append(f"system_power.{name}")
    if missing:
        raise MetadataDraftError(f"Populate {path}: missing or invalid disclosures: {', '.join(missing)}")
    _reject_unknown_fields(system, HUMAN_SYSTEM_FIELDS, f"{path}: system")
    point = value.get("point", {})
    if not isinstance(point, dict):
        raise MetadataDraftError(f"{path}: point must be a mapping when supplied.")
    _reject_unknown_fields(point, HUMAN_POINT_FIELDS, f"{path}: point")
    settings = point.get("runtime_settings", {})
    if not isinstance(settings, dict):
        raise MetadataDraftError(f"{path}: point.runtime_settings must be a mapping.")
    _reject_unknown_fields(settings, HUMAN_RUNTIME_FIELDS, f"{path}: point.runtime_settings")
    nodes = system.get("node_types", [])
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise MetadataDraftError(f"{path}: system.node_types must be a list of mappings.")
    SystemDescription, NodeType, _, _ = _schema_models()
    for index, node in enumerate(nodes):
        _reject_unknown_fields(node, _known_fields(NodeType), f"{path}: system.node_types[{index}]")
        owned_software = {"inference_backend", "other_software_stack", "container_link"}
        if owned_software.intersection(node):
            raise MetadataDraftError(f"{path}: serving software must be observed in the container, not supplied manually.")
    try:
        SystemDescription.model_validate({
            **_schema_system_defaults(),
            **{key: item for key, item in system.items() if key != "node_types" and item is not None},
        })
        if __package__:
            from .submission_package import _api, system_identifier, submission_identifier
        else:
            from submission_package import _api, system_identifier, submission_identifier
        submission_identifier(system["submission_id"])
        _api().SystemPower.model_validate({
            **power, "system_desc_id": system_identifier(system["system_name"]),
        })
    except Exception as exc:
        raise MetadataDraftError(f"Populate {path}: invalid disclosure values: {exc}") from exc
    return value


def _omit_nulls(value):
    if isinstance(value, dict):
        return {key: _omit_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_omit_nulls(item) for item in value if item is not None]
    return value


def _apply_disclosures(draft: dict, disclosures: dict, policy: dict) -> dict:
    draft = copy.deepcopy(draft)
    human_system = disclosures["system"]
    node_disclosures = human_system.get("node_types", [])
    if len(node_disclosures) > len(draft["system"]["node_types"]):
        raise MetadataDraftError("Manual node disclosures exceed the observed node types.")
    for index, provided in enumerate(node_disclosures):
        observed = draft["system"]["node_types"][index]
        for name, value in provided.items():
            if value is None:
                continue
            if observed.get(name) is not None and observed[name] != value:
                raise MetadataDraftError(f"system.node_types[{index}].{name} contradicts an observed host fact.")
            observed[name] = value
    for name, value in human_system.items():
        if name == "node_types" or value is None:
            continue
        if draft["system"].get(name) is not None and draft["system"][name] != value:
            raise MetadataDraftError(f"system.{name} contradicts an observed host fact.")
        draft["system"][name] = value
    draft["point"]["division"] = human_system["division"]
    supplied_point = copy.deepcopy(disclosures.get("point", {}))
    settings = supplied_point.pop("runtime_settings", {})
    # The benchmark template determines this observed client setting.
    settings.pop("stream_all_chunks", None)
    draft["point"].update(supplied_point)
    draft["point"]["runtime_settings"].update(settings)
    draft["point"]["seed_set"] = policy["seed_set"]
    draft["point"]["target_cohort"] = policy["target_cohort"]
    draft["point"]["runtime_settings"]["runtime"] = policy["seeds"]
    if __package__:
        from .submission_package import system_identifier
    else:
        from submission_package import system_identifier
    system_id = system_identifier(draft["system"]["system_name"])
    power = copy.deepcopy(disclosures["system_power"])
    if power.get("system_desc_id") not in {None, system_id}:
        raise MetadataDraftError("system_power.system_desc_id contradicts system.system_name.")
    power["system_desc_id"] = system_id
    draft["system_power"] = power
    if "warmup" in disclosures:
        draft["warmup"] = disclosures["warmup"]
    return _omit_nulls(draft)


def _collect_runtime_metadata(points: list[dict], env: dict[str, str], root: Path) -> list[dict]:
    """Probe selected images without model preparation; cache repeated image refs."""
    cache = {}
    rows = []
    for point in points:
        key = (point.get("RUNTIME_IMAGE"), point.get("RUNTIME_DEVICE"))
        if key not in cache:
            point_env = {name: value for name, value in env.items() if not name.startswith("SUBMISSION_POINT_")}
            point_env["TARGET_CONCURRENCY"] = str(point["concurrency"])
            point_env["STACK_PROFILE"] = str(point["STACK_PROFILE"])
            # The resolver has already applied topology overrides; preserve them.
            for name in POINT_DIMENSIONS:
                resolved = point.get(name.upper())
                if resolved is not None:
                    point_env["SUBMISSION_POINT_" + name.upper()] = str(resolved)
            if env.get("SUBMISSION_VALIDATE_ONLY") == "1":
                point_env["SUBMISSION_RUNTIME_LOCAL_ONLY"] = "1"
            log_root = env.get("SUBMISSION_METADATA_CAPTURE_DIR")
            log_path = Path(log_root) / f"runtime-{point['concurrency']}.log" if log_root else None
            try:
                result = subprocess.run(
                    ["bash", str(root / "collect_submission_runtime_metadata.sh")],
                    cwd=root, env=point_env, capture_output=True, text=True, timeout=1800,
                )
            except subprocess.TimeoutExpired as exc:
                if log_path:
                    output = exc.stderr or b""
                    _create_exclusive(log_path, output.encode() if isinstance(output, str) else output)
                    stdout = exc.stdout or b""
                    _create_exclusive(log_path.with_suffix(".stdout"), stdout.encode() if isinstance(stdout, str) else stdout)
                raise MetadataDraftError("Container software collection timed out.") from exc
            if log_path:
                _create_exclusive(log_path, result.stderr.encode())
                _create_exclusive(log_path.with_suffix(".stdout"), result.stdout.encode())
            if result.returncode:
                raise MetadataDraftError(
                    f"Container software collection failed for {point.get('STACK_PROFILE')}: "
                    f"{result.stderr.strip()[-1000:]}"
                )
            try:
                facts = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise MetadataDraftError("Container software probe did not return JSON.") from exc
            if not isinstance(facts, dict):
                raise MetadataDraftError("Container software facts must be a mapping.")
            required = ("image_id", "image_ref", "vllm_version", "python_version", "torch_version", "platform", "container_os")
            missing = [name for name in required if not _populated(facts.get(name))]
            if missing or not re.fullmatch(r"sha256:[0-9a-f]{64}", facts.get("image_id", "")):
                raise MetadataDraftError(f"Container software probe lacks valid facts: {missing or ['image_id']}")
            cache[key] = facts
        rows.append({**cache[key], "concurrency": point["concurrency"]})
    return rows


def _apply_runtime_metadata(draft: dict, rows: list[dict], disclosures: dict | None = None) -> dict:
    if not rows:
        raise MetadataDraftError("No container software observations were collected.")
    draft["system"]["serving_framework"] = "; ".join(sorted({f"vLLM {row['vllm_version']}" for row in rows}))
    stacks = []
    for row in rows:
        packages = {
            "Python": row["python_version"], "PyTorch": row["torch_version"],
            "container OS": row["container_os"], **row.get("software", {}),
        }
        stacks.append(
            f"vLLM {row['vllm_version']} [{row['image_id']}]: "
            + "; ".join(f"{name} {version}" for name, version in packages.items() if version)
        )
    manual_nodes = (disclosures or {}).get("system", {}).get("node_types", [])
    for index, node in enumerate(draft["system"]["node_types"]):
        node["inference_backend"] = "; ".join(sorted({f"vLLM {row['platform']}" for row in rows}))
        existing = node.get("other_software_stack")
        node["other_software_stack"] = "; ".join(sorted({*stacks, *([existing] if existing else [])}))
        drivers = sorted({row["driver_version"] for row in rows if row.get("driver_version")})
        if drivers:
            observed_driver = "; ".join(drivers)
            manual_driver = manual_nodes[index].get("driver") if index < len(manual_nodes) else None
            if manual_driver is not None and manual_driver != observed_driver:
                raise MetadataDraftError(f"system.node_types[{index}].driver contradicts the observed container driver.")
            node["driver"] = observed_driver
        elif not node.get("driver") and all(row["platform"].casefold() == "cpu" for row in rows):
            node["driver"] = "Not applicable (CPU inference; no accelerator driver required)"
    return _omit_nulls(draft)


def _resolve_points(config: dict, env: dict[str, str], root: Path) -> list[dict]:
    results = []
    resolver = root / "resolve_submission_point.sh"
    if not resolver.is_file():
        raise MetadataDraftError(f"Missing point resolver: {resolver}")
    for definition in config["points"]:
        point_env = {
            key: value
            for key, value in env.items()
            if not key.startswith("SUBMISSION_POINT_")
        }
        point_env["TARGET_CONCURRENCY"] = str(definition["concurrency"])
        profile = definition.get("stack_profile")
        if profile is not None:
            point_env["STACK_PROFILE"] = profile
        for name in POINT_DIMENSIONS.intersection(definition):
            point_env["SUBMISSION_POINT_" + name.upper()] = str(definition[name])
        try:
            result = subprocess.run(
                ["bash", str(resolver)],
                cwd=root,
                env=point_env,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MetadataDraftError(
                f"Could not resolve concurrency {definition['concurrency']}: {exc}"
            ) from exc
        if result.returncode:
            detail = (result.stderr or "").strip()[:500]
            raise MetadataDraftError(
                f"Point resolver failed for concurrency {definition['concurrency']}"
                + (f": {detail}" if detail else ".")
            )
        try:
            serving = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MetadataDraftError(
                f"Point resolver returned invalid JSON for concurrency "
                f"{definition['concurrency']}."
            ) from exc
        if not isinstance(serving, dict):
            raise MetadataDraftError("Point resolver must return a JSON object.")
        results.append({"concurrency": definition["concurrency"], **serving})
    return results


def _consistent_value(points: list[dict], field: str) -> str | None:
    values = {
        value.strip()
        for point in points
        if isinstance((value := point.get(field)), str) and value.strip()
    }
    if len(values) > 1:
        raise MetadataDraftError(
            f"Resolved points disagree about {field}: {sorted(values)}"
        )
    return next(iter(values), None)


def _benchmark_config(env: dict[str, str], root: Path) -> dict:
    selected = env.get("BENCHMARK_CONFIG_TEMPLATE")
    path = Path(selected) if selected else root / "configs/qwen_agentic_benchmark.yaml"
    if not path.is_absolute():
        path = root / path
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _precision(points: list[dict], model_name: str | None) -> str | None:
    resolved = _consistent_value(points, "WEIGHT_DTYPE")
    if resolved and resolved.casefold() not in {"profile", "auto"}:
        return resolved.upper()
    model = (model_name or "").casefold()
    if "nvfp4" in model:
        return "NVFP4"
    if "fp8" in model:
        return "FP8"
    if "w8a8" in model:
        return "INT8"
    if model.endswith("qwen3.6-35b-a3b") or model == "qwen/qwen3.6-35b-a3b":
        return "BF16"
    return None


def _approved_revision(model_name: str, resolved_revision: str | None) -> str | None:
    approved = APPROVED_REVISIONS.get(model_name)
    if approved is None:
        return None
    if resolved_revision and resolved_revision.casefold() != approved.casefold():
        return None
    return approved


def _model_link(model_name: str | None, revision: str | None) -> str | None:
    if not model_name or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        model_name,
    ):
        return None
    base = "https://huggingface.co/" + quote(model_name, safe="/-._")
    approved = _approved_revision(model_name, revision)
    if approved:
        return f"{base}/tree/{quote(approved, safe='')}"
    return base


def _empty_node() -> dict:
    # Internal assembly shape; required gaps are rejected and optional nulls
    # are omitted before writing metadata.
    _, NodeType, _, _ = _schema_models()
    return {name: None for name in NodeType.model_fields} | {"cooling": None}


def _build_draft(
    *,
    points: list[dict],
    facts: dict,
    env: dict[str, str],
    root: Path,
) -> dict:
    maximum = max(point["concurrency"] for point in points)
    source_model = _consistent_value(points, "MODEL_NAME")
    benchmark = _benchmark_config(env, root)
    model_name = benchmark.get("model_params", {}).get("name")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = source_model
    model_revision = _consistent_value(points, "MODEL_REVISION")
    precision = _precision(points, source_model or model_name)
    nodes = facts.get("node_types")
    if not isinstance(nodes, list):
        nodes = []
    point = {
        "division": None,
        "max_supported_concurrency": maximum,
        "model_name": model_name,
        "model_precision": precision,
        "link_to_model": _model_link(source_model or model_name, model_revision)
        if source_model or model_name else None,
        "link_to_model_transformation": None,
        "model_notes": None,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_type": "Accuracy + Performance",
        "dataset_link": DATASET_LINK,
        "shared_src": SHARED_SRC,
        "shared_docs": SHARED_DOCS,
        "seed_set": None,
        "target_cohort": None,
        "runtime_settings": {
            "load_pattern": "agentic_inference",
            "min_sample_count": None,
            "stream_all_chunks": benchmark.get("settings", {}).get("client", {}).get(
                "stream_all_chunks", False
            ),
            "runtime": {
                "scheduler_rng_seed": None,
                "sample_index_rng_seed": None,
                "model_seed": None,
            },
        },
    }
    system = {
        "submitter_org_names": None,
        "submitter_contact": None,
        "submission_id": None,
        "submission_date": None,
        "publish_date": None,
        "system_name": None,
        "shortened_system_name": None,
        "publication_status": None,
        "availability_url": None,
        "target_availability_date": None,
        "max_supported_concurrency": maximum,
        "system_size": None,
        "system_node_ensemble_count": facts.get("system_node_ensemble_count"),
        "system_node_ensemble_total": facts.get("system_node_ensemble_total"),
        "node_types": nodes if nodes else [_empty_node()],
        "serving_framework": facts.get("serving_framework"),
        "node_config": facts.get("node_config"),
        "config_summary": None,
        "config_summary_notes": None,
        "division": None,
    }
    # An imported description may include multiple factual node types. Ensure
    # cooling remains an undisclosed field in every row.
    for node in system["node_types"]:
        node["cooling"] = None
    return {
        "system": system,
        "system_power": {"provisioned_power_w": None, "cooling": None},
        "point": point,
    }


def _validate_partial_fact_types(draft: dict) -> None:
    SystemDescription, NodeType, _, _ = _schema_models()
    node_defaults = {
        "host_memory_configuration": "",
        "host_network_card_count": "",
        "driver": "",
        "filesystem": "",
        "host_processor_vcpu_count": 1,
    }
    system_candidate = _schema_system_defaults()
    system_candidate.update(
        {
            key: value
            for key, value in draft["system"].items()
            if value is not None
        }
    )
    system_candidate["node_types"] = []
    for node in draft["system"]["node_types"]:
        candidate = {**node_defaults, **{k: v for k, v in node.items() if v is not None}}
        if candidate.get("accelerator_info") is None:
            candidate["accelerator_info"] = []
        try:
            NodeType.model_validate(candidate)
        except Exception as exc:
            raise MetadataDraftError(f"Invalid factual node field: {exc}") from exc
    # Validate the present system-level fact types with placeholders only for
    # required human disclosures; never serialize these placeholders.
    try:
        SystemDescription.model_validate(system_candidate)
    except Exception as exc:
        raise MetadataDraftError(f"Invalid factual system field: {exc}") from exc


def _metadata_destination(config_path: Path, metadata: str) -> Path:
    candidate = config_path.parent / metadata
    # Resolve parent symlinks but preserve the final name so O_EXCL can reject a
    # symlink at the destination itself, including a dangling link.
    return candidate.parent.resolve() / candidate.name


def _create_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise MetadataDraftError(
            f"Refusing to overwrite existing metadata path: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_provenance_value(value: object, *, kind: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if kind == "model":
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value):
            return value
    elif kind == "revision":
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", value):
            return value
    elif kind == "dtype":
        if re.fullmatch(r"[A-Za-z0-9._+-]{1,32}", value):
            return value
    elif kind == "profile":
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            return value
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _provenance(
    *,
    config_bytes: bytes,
    source_digest: str | None,
    points: list[dict],
    draft_bytes: bytes,
    collection: dict,
) -> dict:
    resolved = []
    for point in points:
        image = point.get("RUNTIME_IMAGE")
        resolved.append({
            "concurrency": point["concurrency"],
            "stack_profile": _safe_provenance_value(
                point.get("STACK_PROFILE"), kind="profile"
            ),
            "model_name": _safe_provenance_value(point.get("MODEL_NAME"), kind="model"),
            "model_revision": _safe_provenance_value(
                point.get("MODEL_REVISION"), kind="revision"
            ),
            "runtime_image_sha256": (
                hashlib.sha256(image.encode()).hexdigest()
                if isinstance(image, str) and image
                else None
            ),
            "weight_dtype": _safe_provenance_value(
                point.get("WEIGHT_DTYPE"), kind="dtype"
            ),
            "kv_cache_dtype": _safe_provenance_value(
                point.get("KV_CACHE_DTYPE"), kind="dtype"
            ),
        })
    return {
        "format_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "system_description_sha256": source_digest,
        "point_concurrencies": [point["concurrency"] for point in points],
        "resolved_points": resolved,
        "metadata_draft_sha256": hashlib.sha256(draft_bytes).hexdigest(),
        "collection": collection,
    }


def _missing_disclosures(draft: dict, *, limit: int = 60) -> list[str]:
    missing: list[str] = []

    def visit(value, path: str) -> None:
        if value is None:
            missing.append(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(draft, "")
    return missing[:limit] + (
        [f"... {len(missing) - limit} additional null disclosures"]
        if len(missing) > limit
        else []
    )


def _output_directory(env: dict[str, str], root: Path, config_path: Path) -> Path:
    value = env.get("OUTPUT_ROOT", str(root / "run_output"))
    output_root = Path(value)
    if not output_root.is_absolute():
        raise MetadataDraftError("OUTPUT_ROOT must be an absolute path.")
    output_root = output_root.resolve()
    source_root = root.resolve()
    packaged_output = source_root / "run_output"
    if output_root.is_relative_to(source_root) and not output_root.is_relative_to(packaged_output):
        raise MetadataDraftError(
            "OUTPUT_ROOT inside the source tree must be under run_output."
        )
    if output_root == config_path.parent:
        raise MetadataDraftError(
            "OUTPUT_ROOT must keep provenance separate from the campaign config."
        )
    try:
        output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise MetadataDraftError(f"Cannot create OUTPUT_ROOT: {exc}") from exc
    if not output_root.is_dir():
        raise MetadataDraftError("OUTPUT_ROOT must name a directory.")
    return output_root


def create_draft(
    *,
    config_path: Path | str | None = None,
    env: dict[str, str] | None = None,
    root: Path | str = ROOT,
    preview: bool = True,
    resolved_points: list[dict] | None = None,
) -> tuple[Path, Path, dict]:
    root = Path(root).resolve()
    process_env = dict(os.environ if env is None else env)
    configured = config_path or process_env.get(
        "SUBMISSION_CONFIG", str(root / "configs/submission.yaml")
    )
    config_path = Path(configured)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve(strict=True)
    config, config_bytes = _load_config(config_path)
    disclosure_path = (config_path.parent / config["disclosures"]).resolve()
    try:
        disclosure_bytes = disclosure_path.read_bytes()
    except OSError:
        # Reuse the field-oriented missing-file diagnostic.
        load_disclosures(disclosure_path)
        raise
    disclosures = load_disclosures(disclosure_path, raw=disclosure_bytes)
    policy = load_checker_policy()
    destination = _metadata_destination(config_path, config["metadata"])
    system_cache = destination.with_suffix(".system_desc.json")
    cache_provenance = system_cache.with_suffix(".provenance.json")
    resume = process_env.get("SUBMISSION_RESUME_DIR")
    if resume:
        resume_path = Path(resume)
        if not resume_path.is_absolute() or not resume_path.is_dir():
            raise MetadataDraftError("SUBMISSION_RESUME_DIR must be an existing absolute campaign directory.")
        snapshot = resume_path / "metadata"
        if (snapshot / "system_desc.json").is_file():
            system_cache = snapshot / "system_desc.json"
            cache_provenance = snapshot / "provenance.json"
    if preview and os.path.lexists(destination):
        raise MetadataDraftError(
            f"Refusing to overwrite existing metadata path: {destination}"
        )

    points = resolved_points if resolved_points is not None else _resolve_points(config, process_env, root)
    pins = _mlcflow_pins(root)
    output_root = _output_directory(process_env, root, config_path)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex
    )
    provenance_dir = output_root / f"submission-metadata-{run_id}"
    try:
        provenance_dir.mkdir(mode=0o700)
    except FileExistsError:
        # UUID collision is exceedingly unlikely; retry with a fresh name.
        provenance_dir = output_root / f"submission-metadata-{uuid.uuid4().hex}"
        provenance_dir.mkdir(mode=0o700)
    if not preview:
        destination = provenance_dir / "metadata.yaml"
    _create_exclusive(provenance_dir / "disclosures.yaml", disclosure_bytes)
    imported = process_env.get("SUBMISSION_SYSTEM_DESCRIPTION")
    reused = not imported and system_cache.is_file()
    system_revision = pins[1]
    collection = {
        "mode": "imported" if imported else "automatic_host",
        "mlcflow_version": None,
        "automation_revision": pins[1],
        "status": "pending",
    }
    try:
        if reused:
            source_bytes = system_cache.read_bytes()
            if not cache_provenance.is_file():
                raise MetadataDraftError(
                    "Cached system description has no provenance sidecar. Set "
                    "SUBMISSION_SYSTEM_DESCRIPTION to import its absolute JSON "
                    "path, or configure a fresh cache and rerun "
                    "init_submission_metadata.sh."
                )
            cached = json.loads(cache_provenance.read_bytes())
            if not isinstance(cached, dict) or not isinstance(cached.get("collection"), dict):
                raise MetadataDraftError("Cached system description provenance is invalid.")
            if cached["collection"].get("status") != "validated":
                raise MetadataDraftError("Cached system description was not validated.")
            if cached.get("system_description_sha256") != hashlib.sha256(source_bytes).hexdigest():
                raise MetadataDraftError("Cached system description hash differs from its provenance.")
            collection = copy.deepcopy(cached["collection"])
            system_revision = collection.get("automation_revision", pins[1])
            collection.pop("artifact_sha256", None)
            collection["reused_system_description"] = str(system_cache)
        elif imported:
            source = Path(imported)
            if not source.is_absolute():
                raise MetadataDraftError("SUBMISSION_SYSTEM_DESCRIPTION must be absolute.")
            source_bytes = source.read_bytes()
        else:
            source = _collect_mlcflow(
                capture_dir=provenance_dir, env={**process_env, "SUBMISSION_SOURCE_ROOT": str(root)},
                pins=pins, record=collection,
            )
            source_bytes = source.read_bytes()
        # Normalize the saved bytes, so provenance and validation refer to the
        # same source even when the original import is subsequently changed.
        captured_source = provenance_dir / "system_desc.json"
        _create_exclusive(captured_source, source_bytes)
        collection["raw_system_description"] = captured_source.name
        facts, source_digest = _normalized_imported_system(
            captured_source, revision=system_revision
        )
        collection["automation_provenance"] = (
            collection.get("automation_provenance", "declared_output") if reused
            else "verified_runtime_and_declared_output" if not imported else "declared_output"
        )
        draft = _build_draft(points=points, facts=facts, env=process_env, root=root)
        draft = _apply_disclosures(draft, disclosures, policy)
        if __package__:
            from .submission_package import validate_metadata
        else:
            from submission_package import validate_metadata
        # Reject missing host disclosures and invalid power before image work.
        try:
            validate_metadata(draft, require_software=False)
        except ValueError as exc:
            raise MetadataDraftError(f"Populate {disclosure_path}: {exc}") from exc
        runtime_facts = _collect_runtime_metadata(
            points, {**process_env, "SUBMISSION_METADATA_CAPTURE_DIR": str(provenance_dir)}, root,
        )
        draft = _apply_runtime_metadata(draft, runtime_facts, disclosures)
        try:
            validate_metadata(draft)
        except ValueError as exc:
            raise MetadataDraftError(f"Populate {disclosure_path}: {exc}") from exc
        _create_exclusive(
            provenance_dir / "runtime-software.json",
            (json.dumps(runtime_facts, indent=2, sort_keys=True) + "\n").encode(),
        )
        _create_exclusive(provenance_dir / "seed_sets.yaml", policy["registry_bytes"])
        draft_bytes = yaml.safe_dump(
            draft, sort_keys=False, allow_unicode=True,
        ).encode("utf-8")
        collection["status"] = "validated"
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        collection["status"] = "failed"
        collection["artifact_sha256"] = {
            str(path.relative_to(provenance_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*provenance_dir.iterdir(), *provenance_dir.glob("collector/*.log"))
            if path.is_file()
        }
        _create_exclusive(
            provenance_dir / "collection.json",
            (json.dumps(collection, indent=2, sort_keys=True) + "\n").encode(),
        )
        raise MetadataDraftError(
            f"{exc} Capture diagnostics: {provenance_dir}"
        ) from exc
    collection["artifact_sha256"] = {
        str(path.relative_to(provenance_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (captured_source, provenance_dir / "collector.log", provenance_dir / "collector-observations.json", provenance_dir / "collector-preparation.log", *sorted(provenance_dir.glob("runtime-*")), *sorted(provenance_dir.glob("collector/*.log")))
        if path.is_file()
    }
    provenance = _provenance(
        config_bytes=config_bytes,
        source_digest=source_digest,
        points=points,
        draft_bytes=draft_bytes,
        collection=collection,
    )
    provenance.update({
        "disclosures_sha256": hashlib.sha256(disclosure_bytes).hexdigest(),
        "checker_ref": policy["checker_ref"],
        "registry_sha256": policy["registry_sha256"],
        "seed_registry": "seed_sets.yaml",
        "runtime_software": runtime_facts,
    })
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MetadataDraftError(
            f"Cannot create metadata destination directory: {exc}"
        ) from exc
    _create_exclusive(destination, draft_bytes)
    provenance_path = provenance_dir / "provenance.json"
    _create_exclusive(
        provenance_path,
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
    )
    if not imported and not reused:
        # Persist only a fully validated capture. Later preflight/resume runs
        # still validate disclosures and runtime software, but need no sudo.
        _create_exclusive(system_cache, source_bytes)
        _create_exclusive(
            cache_provenance,
            (json.dumps({
                "system_description_sha256": source_digest,
                "collection": collection,
            }, indent=2, sort_keys=True) + "\n").encode(),
        )
    return destination, provenance_path, draft


def main() -> int:
    if len(sys.argv) != 1:
        raise MetadataDraftError(
            "init_submission_metadata.sh accepts no positional arguments."
        )
    destination, provenance_path, draft = create_draft()
    print(f"Created complete submission metadata: {destination}")
    print(f"Secret-safe provenance: {provenance_path}")
    print("Metadata validated; final measured-run checks occur during packaging.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MetadataDraftError, OSError, ValueError) as exc:
        print(f"Submission metadata failed: {exc}", file=sys.stderr)
        sys.exit(1)
