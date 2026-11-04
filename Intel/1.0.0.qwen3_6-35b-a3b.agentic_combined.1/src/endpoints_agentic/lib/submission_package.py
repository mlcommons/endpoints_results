"""Build and check an official MLPerf Endpoints submission bundle locally."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if __package__:
    from .checker_runtime import installed_checker_ref
    from .submission_requirements import benchmark_datasets
else:
    from checker_runtime import installed_checker_ref
    from submission_requirements import benchmark_datasets
_SEEDS = ("scheduler_rng_seed", "sample_index_rng_seed", "model_seed")
_WARMUP_FIELDS = {
    "data_source", "enabled", "n_requests", "initialization_steps",
    "logs_retained", "link_logs",
}
PARALLEL_KEYS = (
    "RUNTIME_DEVICE", "TOPOLOGY_MODE", "TP_SIZE", "REPLICA_COUNT",
    "PREFILL_TP_SIZE", "PREFILL_REPLICA_COUNT", "DECODE_TP_SIZE",
    "DECODE_REPLICA_COUNT", "VLLM_TENSOR_PARALLEL_SIZE",
)
_TP_KEYS = (
    "TP_SIZE", "PREFILL_TP_SIZE", "DECODE_TP_SIZE", "VLLM_TENSOR_PARALLEL_SIZE",
)
_DP_KEYS = ("REPLICA_COUNT", "PREFILL_REPLICA_COUNT", "DECODE_REPLICA_COUNT")


class SubmissionPackageError(RuntimeError):
    """A local submission could not be built or checked."""


@dataclass(frozen=True)
class _API:
    SystemDescription: Any
    SystemPower: Any
    PointConfig: Any
    build: Any
    archive: Any


def _api() -> _API:
    try:
        installed_checker_ref()
        from endpoints_submission_cli.submissions.builder import (
            build_submission_folder,
            create_bundle_archive,
        )
        from submission_checker.models.file import PointConfig, SystemDescription, SystemPower
        if not callable(getattr(SystemPower, "compute", None)):
            raise SubmissionPackageError("Selected checker lacks the supported Appendix E power API")
    except SubmissionPackageError:
        raise
    except (ImportError, ValueError) as exc:
        raise SubmissionPackageError(
            f"Selected upstream checker is unavailable or incompatible; run prepare_submission.sh: {exc}"
        ) from exc
    return _API(
        SystemDescription, SystemPower, PointConfig, build_submission_folder,
        create_bundle_archive,
    )


def system_identifier(name: str) -> str:
    value = name.strip().replace(" ", "_")
    if not value or value in {".", ".."} or "/" in value or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("system.system_name must map to one safe results directory")
    return value


def submission_identifier(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("system.submission_id must name one submission directory")
    return value


def _validate_metadata(metadata: dict, api: _API, *, require_software: bool = True) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    system, point = metadata.get("system"), metadata.get("point")
    if not isinstance(system, dict) or not isinstance(point, dict):
        raise ValueError("metadata.system and metadata.point must be objects")
    power = metadata.get("system_power")
    if not isinstance(power, dict):
        raise ValueError("metadata.system_power must be an object")
    required_strings = (
        "submitter_org_names", "submitter_contact", "submission_id",
        "system_name", "shortened_system_name", "publication_status",
        "system_size", "division",
    )
    missing = [f"system.{name}" for name in required_strings
               if not isinstance(system.get(name), str) or not system[name].strip()]
    nodes = system.get("node_types")
    if not isinstance(nodes, list) or not nodes:
        missing.append("system.node_types (at least one observed node)")
        nodes = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            missing.append(f"system.node_types[{index}]")
            continue
        required_node_strings = ["host_memory_configuration", "host_network_card_count", "filesystem"]
        if require_software:
            required_node_strings.append("driver")
        for name in required_node_strings:
            if not isinstance(node.get(name), str) or not node[name].strip():
                missing.append(f"system.node_types[{index}].{name}")
        if require_software:
            for name in ("inference_backend", "other_software_stack"):
                if not isinstance(node.get(name), str) or not node[name].strip():
                    missing.append(f"system.node_types[{index}].{name}")
    if require_software and (
        not isinstance(system.get("serving_framework"), str) or not system["serving_framework"].strip()
    ):
        missing.append("system.serving_framework")
    if missing:
        raise ValueError("Missing required metadata disclosures: " + ", ".join(missing))
    submission_identifier(system["submission_id"])

    settings = point.get("runtime_settings")
    if not isinstance(settings, dict):
        settings = {}
    runtime = settings.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    misplaced = {*_SEEDS, "scheduler_random_seed", "dataloader_random_seed"}
    if misplaced.intersection(point) or misplaced.intersection(settings):
        raise ValueError("Seeds must be nested under point.runtime_settings.runtime")
    missing_seeds = [name for name in _SEEDS if type(runtime.get(name)) is not int]
    if missing_seeds:
        raise ValueError(
            "point.runtime_settings.runtime must contain integer seeds: "
            + ", ".join(missing_seeds)
        )
    warmup = metadata.get("warmup")
    if warmup is not None and (
        not isinstance(warmup, dict) or set(warmup) - _WARMUP_FIELDS
    ):
        raise ValueError("metadata.warmup must use supported warmup disclosure fields")

    try:
        candidate_system = copy.deepcopy(system)
        if not require_software:
            # Only for early structural/power validation; never serialize a
            # placeholder driver before observing container software.
            for node in candidate_system["node_types"]:
                node.setdefault("driver", "")
        system_model = api.SystemDescription.model_validate(candidate_system)
        power_model = api.SystemPower.model_validate(power)
        point_model = api.PointConfig.model_validate({
            **point, "concurrency": 1,
            "shared_src": "src/endpoints_agentic", "shared_docs": "docs",
        })
        # The schema records some missing disclosures instead of raising.
        missing = [
            result.message for result in point_model._check_results
            if result.rule in {"point-disclosure-complete", "target-cohort", "load-pattern"} and not result.passed
        ]
        disclosure_fields = (
            "model_name", "model_precision", "link_to_model", "dataset_name",
            "dataset_type", "dataset_link", "shared_src", "shared_docs", "seed_set", "target_cohort",
        )
        missing.extend(f"point.{name} must be nonblank" for name in disclosure_fields
                       if not isinstance(point.get(name), str) or not point[name].strip())
        if missing:
            raise ValueError("; ".join(missing))
        if point_model.dataset_type.casefold() not in {"accuracy", "performance", "accuracy + performance"}:
            raise ValueError("point.dataset_type must be Accuracy, Performance, or Accuracy + Performance")
        if power_model.system_desc_id != system_identifier(system["system_name"]):
            raise ValueError("system_power.system_desc_id must match the system results ID")
        ensembles = {node.system_node_ensemble_id: node for node in system_model.node_types}
        cores = {identifier: node.host_processor_core_count for identifier, node in ensembles.items()
                 if node.host_processor_core_count is not None}
        node_totals = {}
        for node_set in power_model.node_sets:
            ensemble = ensembles.get(node_set.system_node_ensemble_id)
            if ensemble is None:
                raise ValueError("system_power.node_sets references an unknown system_node_ensemble_id")
            identifier = node_set.system_node_ensemble_id
            node_totals[identifier] = node_totals.get(identifier, 0) + node_set.nodes_provisioned
        # Node sets describe physical machines, never replicas or NUMA groups.
        if any(node_totals.get(identifier) != node.number_of_nodes for identifier, node in ensembles.items()):
            raise ValueError("system_power.node_sets.nodes_provisioned totals must match observed physical node counts")
        computed = power_model.compute(cores)
        if computed.problems:
            raise ValueError("Invalid system_power: " + "; ".join(computed.problems))
        power_kw = computed.provisioned_power_kw
        if power_kw is None or not math.isfinite(power_kw) or power_kw <= 0:
            raise ValueError("system_power must disclose a positive provisioned total")
        system_max = system.get("max_supported_concurrency")
        point_max = point.get("max_supported_concurrency")
        if type(system_max) is not int or system_max < 1:
            raise ValueError("system.max_supported_concurrency must be a positive integer")
        if type(point_max) is not int or point_max < 1:
            raise ValueError("point.max_supported_concurrency must be a positive integer")
        if system_max != point_max:
            raise ValueError("point.max_supported_concurrency must match system")
        division = str(getattr(system_model.division, "value", system_model.division))
        if str(point.get("division", "")).casefold() != division.casefold():
            raise ValueError("point.division must match system.division")
        from submission_checker.seed_sets import bundled_seed_sets_path, load_seed_sets
        published = load_seed_sets(bundled_seed_sets_path()).get(point["seed_set"])
        if published is None or runtime != published.seeds:
            raise ValueError("Runtime seeds must exactly match the declared published seed set")
        if point["target_cohort"] not in published.cohorts:
            raise ValueError("Declared seed set is not adoptable for the selected target cohort")
    except Exception as exc:
        raise ValueError(f"Invalid official metadata: {exc}") from exc


def validate_metadata(metadata: dict, *, require_software: bool = True) -> None:
    """Validate complete disclosures against the selected upstream checker."""
    _validate_metadata(metadata, _api(), require_software=require_software)


def _directory(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        raise SubmissionPackageError(f"{label} does not exist: {value}") from exc
    if not path.is_dir():
        raise SubmissionPackageError(f"{label} is not a directory: {path}")
    return path


def _json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionPackageError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SubmissionPackageError(f"{label} must be an object: {path}")
    return value


def _yaml(path: Path, label: str) -> dict:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SubmissionPackageError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SubmissionPackageError(f"{label} must be a mapping: {path}")
    return value


def _measure_warmup(path: Path) -> dict:
    active = measured = None

    def finish(timestamp: int) -> None:
        nonlocal active, measured
        if active is None:
            return
        measured = {
            "phase_type": "warmup",
            "duration_s": (timestamp - active["start"]) / 1e9,
            "requests_issued": active["issued"],
            "requests_completed": active["completed"],
        }
        active = None

    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event.get("event_type")
                if kind == "session.phase_start":
                    timestamp = event.get("timestamp_ns")
                    data = event.get("data") or {}
                    # Native PhaseData arrays start with the tag, then phase_type.
                    phase = data[1] if isinstance(data, list) else data.get("phase_type")
                    finish(timestamp)
                    if phase == "performance":
                        break
                    if phase == "warmup":
                        active = {
                            "start": timestamp, "issued": 0, "completed": 0,
                        }
                elif kind == "session.ended":
                    timestamp = event.get("timestamp_ns")
                    finish(timestamp)
                elif active:
                    active["issued"] += kind == "sample.issued"
                    active["completed"] += kind == "sample.complete"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionPackageError(f"Cannot read events.jsonl {path}: {exc}") from exc

    return measured or {
        "phase_type": None, "duration_s": 0.0,
        "requests_issued": 0, "requests_completed": 0,
    }


def _warmup(metadata: dict, config: dict, measured: dict, concurrency: int) -> dict:
    declared = metadata.get("warmup") or {}
    enabled = measured["phase_type"] == "warmup"
    issued, completed = measured["requests_issued"], measured["requests_completed"]
    if "enabled" in declared and declared["enabled"] is not enabled:
        raise SubmissionPackageError("metadata.warmup.enabled conflicts with measured events")
    if declared.get("n_requests") is not None and declared["n_requests"] != issued:
        raise SubmissionPackageError("metadata.warmup.n_requests conflicts with measured events")

    datasets = config.get("datasets") or []
    performance_dataset = next(
        (
            dataset for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("type") == "performance"
        ),
        None,
    )
    source = declared.get("data_source")
    if not source and enabled and performance_dataset:
        source = performance_dataset.get("name")
    result = {
        "duration_s": measured["duration_s"],
        "requests_issued": issued,
        "requests_completed": completed,
        "data_source": source or "No warmup; zero verified by events.jsonl",
        "concurrency": concurrency if enabled else 0,
    }
    for field in ("initialization_steps", "logs_retained", "link_logs"):
        if field in declared:
            result[field] = declared[field]
    return result


def _normal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value.casefold()
    return value


def _curve(points: list[dict]) -> dict[str, dict[str, Any]]:
    result = {}
    for point in points:
        serving = point["serving"]
        result[f"r{point['concurrency']}"] = {
            key: serving[key] for key in PARALLEL_KEYS if key in serving
        }
    if any(not values for values in result.values()):
        raise SubmissionPackageError(
            "Every serving manifest must expose at least one explicit parallelism key"
        )
    return result


def _official_model_name(name: str) -> str:
    from submission_checker.checker import _ALLOWED_MODEL_NAMES
    from submission_checker.layout import canonical_model_name

    canonical = canonical_model_name(name)
    if canonical in _ALLOWED_MODEL_NAMES:
        return canonical
    folded = canonical.casefold()
    return folded if folded in _ALLOWED_MODEL_NAMES else canonical


def _system_desc(metadata: dict, curve: dict, api: _API) -> dict:
    data = api.SystemDescription.model_validate(metadata["system"]).model_dump(mode="json")
    config = data.get("config_summary")
    topology_modes = {
        str(point.get("TOPOLOGY_MODE", "replica")).casefold()
        for point in curve.values()
    }
    mixed_topology = len(topology_modes) > 1
    for name, keys in (
        ("tensor_parallel", _TP_KEYS),
        ("data_parallel", _DP_KEYS),
    ):
        values = [
            point[key]
            for point in curve.values()
            for key in keys
            if key in point
        ]
        if not values:
            continue
        varies = mixed_topology or len({_normal(value) for value in values}) > 1
        if data.get(name) is not None and (
            varies or _normal(data[name]) != _normal(values[0])
        ):
            data.pop(name, None)
        if isinstance(config, dict) and name in config and (
            varies or _normal(config[name]) != _normal(values[0])
        ):
            config.pop(name)
    api.SystemDescription.model_validate(data)
    return data


def _point_system_desc(system: dict, parallelism: dict, api: _API) -> dict:
    """Latest checker permits truthful configuration differences between points."""
    data = copy.deepcopy(system)
    summary = data.get("config_summary") or {}
    if parallelism.get("RUNTIME_DEVICE") == "cpu":
        # The checker interprets these scalar fields as accelerator counts.
        summary["cpu_topology"] = {
            key: parallelism[key]
            for key in (*PARALLEL_KEYS, "NUMA_NODES", "WORKER_NUMA_GROUPS")
            if key in parallelism
        }
        for name in ("tensor_parallel", "pipeline_parallel", "expert_parallel", "data_parallel"):
            data.pop(name, None)
            summary.pop(name, None)
    else:
        for name, keys in (("tensor_parallel", _TP_KEYS), ("data_parallel", _DP_KEYS)):
            values = {int(parallelism[key]) for key in keys if key in parallelism}
            if len(values) == 1:
                data[name] = summary[name] = next(iter(values))
            elif values:
                # P/D can have distinct stage widths; the serving disclosure retains
                # them instead of inventing one scalar width.
                data.pop(name, None)
                summary.pop(name, None)
    if summary:
        data["config_summary"] = summary
    return api.SystemDescription.model_validate(data).model_dump(mode="json", exclude_none=True)


def _point(metadata: dict, raw: dict, api: _API) -> dict:
    concurrency, region = raw["concurrency"], raw["region"]
    reports = Path(raw["report_dir"])
    summary_path = reports / "performance/result_summary.json"
    config_path = reports / "config.yaml"
    events_path = reports / "events.jsonl"
    accuracy_path = reports / "accuracy/accuracy_results.json"
    config = _yaml(config_path, "config.yaml")
    performance_dataset, swe_dataset = benchmark_datasets(config)

    settings = copy.deepcopy(metadata["point"]["runtime_settings"])
    settings["stream_all_chunks"] = config.get("settings", {}).get("client", {}).get(
        "stream_all_chunks", False
    )

    warmup = _warmup(
        metadata, config, _measure_warmup(events_path), concurrency
    )
    point_yaml = copy.deepcopy(metadata["point"])
    if point_yaml.get("model_name"):
        point_yaml["model_name"] = _official_model_name(point_yaml["model_name"])
    point_yaml.update(
        concurrency=concurrency,
        region=region,
        shared_src="src/endpoints_agentic",
        shared_docs="docs",
        runtime_settings=settings,
        warmup=warmup,
    )
    api.PointConfig.model_validate(point_yaml)
    return {
        "concurrency": concurrency,
        "summary_path": summary_path,
        "config_path": config_path,
        "config": config,
        "performance_dataset": performance_dataset,
        "swe_dataset_name": swe_dataset["name"],
        "accuracy_path": accuracy_path,
        "serving": raw["serving"],
        "serving_environment": raw.get("serving_environment", {}),
        "mode": raw.get("mode"),
        "budget": raw.get("budget", {}),
        "point_yaml": point_yaml,
    }


def _stage_config(
    source: Path,
    destination: Path,
    config: dict,
    performance_dataset: dict,
) -> None:
    """Normalize builder ordering and redact generated SWE-bench credentials."""
    raw = source.read_bytes()
    datasets = config["datasets"]
    performance_index = next(
        index for index, dataset in enumerate(datasets)
        if dataset is performance_dataset
    )
    if performance_index:
        staged_config = copy.deepcopy(config)
        staged_datasets = staged_config["datasets"]
        staged_config["datasets"] = [
            staged_datasets[performance_index],
            *staged_datasets[:performance_index],
            *staged_datasets[performance_index + 1:],
        ]
        import yaml

        raw = yaml.safe_dump(
            staged_config, sort_keys=False, allow_unicode=True
        ).encode()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        accuracy_config = dataset.get("accuracy_config")
        if (
            not isinstance(accuracy_config, dict)
            or accuracy_config.get("eval_method") != "swe_bench_scorer"
        ):
            continue
        extras = accuracy_config.get("extras")
        token = extras.get("swebench_service_auth_token") if isinstance(extras, dict) else None
        if isinstance(token, str) and token and not token.startswith("__"):
            raw = raw.replace(token.encode(), b"<REDACTED>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    shutil.copystat(source, destination)


def _canonicalize_swebench_accuracy(raw: bytes, swe_dataset_name: str) -> bytes:
    """Rename only the staged SWE score to the label required by the checker."""
    if swe_dataset_name == "swe_bench":
        return raw
    payload = json.loads(raw)
    scores = payload.get("accuracy_scores") if isinstance(payload, dict) else None
    if not isinstance(scores, list):
        return raw
    changed = False
    for entry in scores:
        if isinstance(entry, dict) and entry.get("dataset_name") == swe_dataset_name:
            entry["dataset_name"] = "swe_bench"
            changed = True
    if not changed:
        return raw
    return (json.dumps(payload, indent=2) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_yaml(path: Path, value: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _copy_specifications(source: Path | None, destination: Path) -> None:
    """Copy regular documentation files without following links or accepting odd paths."""
    if source is None:
        return
    source = Path(source)
    if not source.is_absolute():
        raise SubmissionPackageError("docs_dir must be absolute")
    if ".." in source.parts:
        raise SubmissionPackageError(f"docs_dir contains a parent traversal: {source}")
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise SubmissionPackageError(f"Cannot inspect docs_dir {source}: {exc}") from exc
    if stat.S_ISLNK(source_stat.st_mode):
        raise SubmissionPackageError(f"docs_dir must not be a symlink: {source}")
    if not stat.S_ISDIR(source_stat.st_mode):
        raise SubmissionPackageError(f"docs_dir is not a directory: {source}")
    try:
        source_resolved = source.resolve(strict=True)
        destination_resolved = destination.resolve(strict=False)
    except OSError as exc:
        raise SubmissionPackageError(f"Cannot resolve documentation paths: {exc}") from exc
    if (
        source_resolved.is_relative_to(destination_resolved)
        or destination_resolved.is_relative_to(source_resolved)
    ):
        raise SubmissionPackageError(
            "docs_dir must not overlap the package specifications directory"
        )

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(source, directory_flags)
    except OSError as exc:
        raise SubmissionPackageError(f"Cannot safely open docs_dir {source}: {exc}") from exc

    def safe_component(name: str) -> None:
        if (
            name in {"", ".", ".."}
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise SubmissionPackageError(
                f"docs_dir contains a malformed path component: {name!r}"
            )

    def copy_directory(directory_fd: int, target_dir: Path) -> None:
        try:
            with os.scandir(directory_fd) as entries:
                names = sorted(entry.name for entry in entries)
        except OSError as exc:
            raise SubmissionPackageError(
                f"Cannot inspect documentation directory {target_dir}: {exc}"
            ) from exc

        for name in names:
            safe_component(name)
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise SubmissionPackageError(
                    f"Cannot inspect documentation path {name!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SubmissionPackageError(
                    f"docs_dir must not contain symlinks: {name!r}"
                )
            target = target_dir / name
            if stat.S_ISDIR(entry_stat.st_mode):
                target.mkdir()
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise SubmissionPackageError(
                        f"Cannot safely open documentation directory {name!r}: {exc}"
                    ) from exc
                try:
                    opened_stat = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened_stat.st_mode)
                        or (opened_stat.st_dev, opened_stat.st_ino)
                        != (entry_stat.st_dev, entry_stat.st_ino)
                    ):
                        raise SubmissionPackageError(
                            f"Documentation path changed while packaging: {name!r}"
                        )
                    copy_directory(child_fd, target)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry_stat.st_mode):
                try:
                    file_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise SubmissionPackageError(
                        f"Cannot safely open documentation file {name!r}: {exc}"
                    ) from exc
                try:
                    opened_stat = os.fstat(file_fd)
                    if (
                        not stat.S_ISREG(opened_stat.st_mode)
                        or (opened_stat.st_dev, opened_stat.st_ino)
                        != (entry_stat.st_dev, entry_stat.st_ino)
                    ):
                        raise SubmissionPackageError(
                            f"Documentation path changed while packaging: {name!r}"
                        )
                    with os.fdopen(file_fd, "rb", closefd=False) as input_file:
                        with target.open("xb") as output_file:
                            shutil.copyfileobj(input_file, output_file)
                    target.chmod(stat.S_IMODE(entry_stat.st_mode) & 0o777)
                finally:
                    os.close(file_fd)
            else:
                raise SubmissionPackageError(
                    f"docs_dir contains a non-regular file: {name!r}"
                )

    try:
        opened_stat = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
        ):
            raise SubmissionPackageError(f"docs_dir changed while packaging: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        copy_directory(root_fd, destination)
    finally:
        os.close(root_fd)


def _run_checker_command(org: Path, report: Path, seed_registry_path: Path | None = None) -> tuple[int, str]:
    from click.testing import CliRunner
    from endpoints_submission_cli.commands.check_submission import check_submission
    from submission_checker.seed_sets import bundled_seed_sets_path

    result = CliRunner().invoke(
        check_submission, [str(org), "--output", str(report)], prog_name="check-submission",
        env={"MLPERF_ENDPOINTS_SEED_SETS": str(seed_registry_path or bundled_seed_sets_path())},
    )
    return int(result.exit_code), result.output


def build_bundle(
    campaign_dir: Path,
    metadata: dict,
    points: list[dict],
    source_dir: Path,
    docs_dir: Path | None = None,
    seed_registry_path: Path | None = None,
) -> Path:
    """Build from validated attempts and a clean package-time source export."""
    api = _api()
    from endpoints_submission_cli.truncation import truncate_responses

    _validate_metadata(metadata, api)
    campaign = _directory(Path(campaign_dir), "campaign_dir")
    source = _directory(Path(source_dir), "source_dir")
    if source.name != "endpoints_agentic":
        raise SubmissionPackageError(
            "source_dir must be a clean package-time endpoints_agentic export"
        )
    if source == campaign or campaign.is_relative_to(source):
        raise SubmissionPackageError("campaign_dir must not be inside the source export")
    if not isinstance(points, list) or not points:
        raise SubmissionPackageError("points must be a non-empty list")
    measured = [_point(metadata, row, api) for row in points]
    measured.sort(key=lambda row: row["concurrency"])
    curve = _curve(measured)
    system = _system_desc(metadata, curve, api)
    system_id = system_identifier(metadata["system"]["system_name"])

    root = Path(tempfile.mkdtemp(prefix="submission-package-", dir=campaign))
    staging, docs = root / "staging", root / "staging/docs"
    src_root = staging / "src"
    implementation = src_root / source.name
    runs = staging / "runs"
    runs.mkdir(parents=True)
    shutil.copytree(source, implementation, copy_function=shutil.copy2)
    _write_json(docs / "systemdescription.json", {**system, "parallelism_by_concurrency": curve})
    _copy_specifications(docs_dir, docs / "specifications")

    archives = []
    for row in measured:
        concurrency = row["concurrency"]
        run = runs / f"run-r{concurrency}"
        _write_json(run / "system_desc.json", _point_system_desc(system, row["serving"], api))
        _write_yaml(run / "point.yaml", row["point_yaml"])
        performance = run / "performance"
        performance.mkdir(parents=True)
        shutil.copy2(row["summary_path"], performance / "result_summary.json")
        _write_json(run / "system_power.json", metadata["system_power"])
        accuracy_file = run / "accuracy/accuracy_results.json"
        accuracy_file.parent.mkdir()
        accuracy_bytes = row["accuracy_path"].read_bytes()
        staged_accuracy = _canonicalize_swebench_accuracy(
            accuracy_bytes, row["swe_dataset_name"]
        )
        if staged_accuracy == accuracy_bytes:
            shutil.copy2(row["accuracy_path"], accuracy_file)
        else:
            accuracy_file.write_bytes(staged_accuracy)
            shutil.copystat(row["accuracy_path"], accuracy_file)
        _stage_config(
            row["config_path"], run / "config.yaml", row["config"],
            row["performance_dataset"],
        )
        _write_json(
            run / "serving_config.json",
            {
                "serving_manifest": row["serving"],
                "point_parallelism": curve[f"r{concurrency}"],
                "serving_environment": row["serving_environment"],
                "mode": row["mode"],
                "budget": row["budget"],
            },
        )
        archive = runs / f"run-r{concurrency}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(run, arcname=run.name)
        archives.append((f"r{concurrency}", archive))

    try:
        org = Path(api.build(
            run_archives=archives,
            division=str(metadata["system"]["division"]).lower(),
            availability=str(metadata["system"]["publication_status"]).lower(),
            work_dir=root / "official",
            submission_id=submission_identifier(metadata["system"]["submission_id"]),
            shared_src_dirs=[src_root],
            shared_docs_dirs=[docs],
        ))
    except Exception as exc:
        raise SubmissionPackageError(f"Official submission folder build failed: {exc}") from exc
    submission = org / submission_identifier(metadata["system"]["submission_id"])
    final_source = submission / "src" / source.name
    # The builder writes source bytes without modes; restore modes in place.
    for source_path in sorted(
        (implementation, *implementation.rglob("*")),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        destination_path = final_source / source_path.relative_to(implementation)
        if destination_path.exists():
            shutil.copymode(source_path, destination_path)

    model_dir = measured[0]["point_yaml"]["model_name"]
    for row in measured:
        point_dir = (
            submission / "results" / system_id
            / model_dir
            / f"r{row['concurrency']}"
        )
        accuracy_bytes = _canonicalize_swebench_accuracy(
            row["accuracy_path"].read_bytes(), row["swe_dataset_name"]
        )
        (point_dir / "accuracy_results.json").write_bytes(
            truncate_responses(accuracy_bytes)
        )
        shutil.copyfile(row["summary_path"], point_dir / "result_summary.json")
    shutil.rmtree(staging)

    checker_report = root / "checker.json"
    try:
        code, output = _run_checker_command(org, checker_report, seed_registry_path)
    except Exception as exc:
        _write_json(checker_report, {"errors": [str(exc)], "warnings": []})
        raise SubmissionPackageError(f"Checker failed to run; report saved at {checker_report}") from exc
    if not checker_report.is_file():
        _write_json(checker_report, {"errors": ["Checker did not write its report"], "stdout": output})
    report = _json(checker_report, "checker.json")
    errors = report.get("errors")
    if not isinstance(errors, list):
        raise SubmissionPackageError(
            f"Official checker report lacks an errors list; report saved at {checker_report}"
        )
    if errors or code:
        raise SubmissionPackageError(
            f"Official checker rejected the bundle ({len(errors)} error(s), "
            f"exit {code}); report saved at {checker_report}"
        )
    bundle = Path(api.archive(org, dest=root / "submission_bundle.tar.gz"))
    if not bundle.is_file():
        raise SubmissionPackageError(f"Official builder did not create {bundle}")
    return bundle


__all__ = [
    "installed_checker_ref", "PARALLEL_KEYS",
    "SubmissionPackageError", "build_bundle", "validate_metadata",
]
