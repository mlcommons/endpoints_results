"""Requirements for the pinned Qwen Agentic Inference submission."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import yaml

# MLCommons Endpoints 4235a9c, examples/10_Agentic_Inference/README.md.
APPROVED_REVISIONS = {
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "Qwen/Qwen3.6-35B-A3B-FP8": "95a723d08a9490559dae23d0cff1d9466213d989",
}
for _catalog in sorted((Path(__file__).resolve().parents[1] / "platforms").glob("*/approved-revisions.json")):
    _revisions = json.loads(_catalog.read_text())
    if not isinstance(_revisions, dict) or any(
        not isinstance(model, str) or not isinstance(revision, str)
        or not re.fullmatch(r"[a-f0-9]{40}", revision)
        or model in APPROVED_REVISIONS for model, revision in _revisions.items()
    ):
        raise ValueError(f"Invalid or duplicate approved model revisions: {_catalog}")
    APPROVED_REVISIONS.update(_revisions)
SAMPLING = {
    "temperature": 1.0, "top_k": 20, "top_p": 0.95,
    "repetition_penalty": 1.0, "presence_penalty": 1.5, "max_new_tokens": 8192,
}


def benchmark_datasets(config: dict) -> tuple[dict, dict]:
    """Find the required datasets by role, independent of labels and order."""
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or any(not isinstance(d, dict) for d in datasets):
        raise ValueError("Submission datasets must be mappings")
    performance = [d for d in datasets if d.get("type") == "performance"]
    swebench = [
        d for d in datasets
        if d.get("type") == "accuracy"
        and isinstance(d.get("accuracy_config"), dict)
        and d["accuracy_config"].get("eval_method") == "swe_bench_scorer"
    ]
    if len(performance) != 1 or len(swebench) != 1:
        raise ValueError("Submission requires one performance and one SWE-bench dataset")
    return performance[0], swebench[0]


def validate_benchmark(config: dict) -> None:
    """Validate effective settings, independently of the editable baseline."""
    if not isinstance(config, dict):
        raise ValueError("Submission benchmark configuration must be a mapping")
    model = config.get("model_params", {})
    if not isinstance(model, dict):
        raise ValueError("Submission model_params must be a mapping")
    if model.get("name") != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("Submission requires Qwen/Qwen3.6-35B-A3B")
    for name, expected in SAMPLING.items():
        actual = model.get(name)
        if type(actual) not in (int, float) or not math.isfinite(actual) or actual != expected:
            raise ValueError(f"Submission requires model_params.{name}={expected}")
    if model.get("chat_template_kwargs") != {"preserve_thinking": True}:
        raise ValueError("Submission requires only chat_template_kwargs.preserve_thinking=true")
    settings = config.get("settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("load_pattern"), dict):
        raise ValueError("Submission settings.load_pattern must be a mapping")
    if settings["load_pattern"].get("type") != "agentic_inference":
        raise ValueError("Submission requires the agentic_inference load pattern")
    perf, swe = benchmark_datasets(config)
    agentic = perf.get("agentic_inference", {})
    if not isinstance(agentic, dict):
        raise ValueError("Submission agentic_inference must be a mapping")
    count = agentic.get("num_trajectories_to_issue")
    if type(count) is not int or count < 613 or count % 613:
        raise ValueError("Performance trajectories must be whole multiples of 613")
    for name in ("enable_salt", "inject_tool_delay"):
        if agentic.get(name) is not True:
            raise ValueError(f"Submission requires agentic_inference.{name}=true")
    if agentic.get("stop_issuing_on_first_user_complete") is not False:
        raise ValueError("Submission requires stop_issuing_on_first_user_complete=false")
    perf_scorer = perf.get("accuracy_config")
    if not isinstance(perf_scorer, dict) or perf_scorer.get("eval_method") != "agentic_inference_inline":
        raise ValueError("Submission requires agentic_inference_inline scoring")
    scorer = swe.get("accuracy_config", {})
    extras = scorer.get("extras", {})
    if not isinstance(extras, dict):
        raise ValueError("Submission SWE-bench extras must be a mapping")
    if type(extras.get("num_instances")) is not int or extras["num_instances"] != 200:
        raise ValueError("Submission requires exactly 200 SWE-bench instances")
    if extras.get("swebench_template") != "qwen_tools":
        raise ValueError("Submission requires the qwen_tools SWE-bench template")


def effective_benchmark(root: Path, env: dict[str, str]) -> dict:
    """Apply the launcher's scalar overrides without writing rendered files."""
    config = yaml.safe_load((root / "configs/qwen_agentic_benchmark.yaml").read_text())
    config = copy.deepcopy(config)

    def locations(value, name):
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                if key == name:
                    found.append((value, key))
                found.extend(locations(child, name))
        elif isinstance(value, list):
            for child in value:
                found.extend(locations(child, name))
        return found

    specs = re.findall(r'"([A-Z_]+)\|([^|"]+)\|([^"]+)"', (root / "lib/benchmark.sh").read_text())
    for variable, key, kind in specs:
        if variable not in env or variable == "TARGET_CONCURRENCY":
            continue  # Campaign points own concurrency.
        raw = env[variable]
        value = yaml.safe_load(raw)
        number = type(value) in (int, float) and math.isfinite(value)
        valid = {
            "positive_integer": type(value) is int and 0 < value <= 2147483647,
            "positive_number": number and value > 0,
            "number": number,
            "optional_number": raw == "null" or number,
            "optional_integer": raw == "null" or type(value) is int,
            "boolean": raw in {"true", "false"},
            "warmup_connections": type(value) is int and value >= -1,
            "duration_or_null": raw == "null" or bool(re.fullmatch(
                r"[1-9][0-9]*|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ms|s|m)", raw)),
        }.get(kind, False)
        if not valid:
            raise ValueError(f"Invalid {variable} override ({kind})")
        matches = locations(config, key)
        if len(matches) != 1:
            raise ValueError(f"Benchmark template must contain exactly one {key}")
        parent, name = matches[0]
        parent[name] = value
    validate_benchmark(config)
    return config
