#!/usr/bin/env python3
"""Run the local Qwen3.6 calibration recipe or verify a quantized checkpoint.

The calibration path is deliberately data-free for INT8 W8A8 RTN.  A local
tokenized dataset is consumed only when FP8 KV-cache parameters are requested.
The shell wrapper owns publication and creation of the ``COMPLETE`` marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

RECIPE_VERSION = 8
KV_CALIBRATION_SAMPLES = 10
KV_MAX_SEQUENCE_LENGTH = 16_384
CORPUS_MANIFEST = "corpus_manifest.json"

IGNORE = [
    "model.embed_tokens",
    "lm_head",
    r"re:^model\.layers\.\d+\.mlp\.(?:gate|shared_expert_gate)$",
]
FP8_BLOCK_IGNORE = [
    *IGNORE,
    r"re:^model\.layers\.\d+\.linear_attn\.in_proj_(?:a|b)$",
]
SOURCE_LANGUAGE_NORM = re.compile(
    r"^model\.language_model\.(?:norm|layers\.\d+\."
    r"(?:input_layernorm|post_attention_layernorm|self_attn\.(?:q_norm|k_norm)))"
    r"\.weight$"
)
MTP_TENSOR = re.compile(r"^mtp\.")
MTP_IGNORE = [
    r"re:^mtp\.layers\.\d+\.mlp\.(?:gate|shared_expert_gate)$",
    r"re:^mtp\.pre_fc_norm_(?:embedding|hidden)$",
]
MTP_WEIGHT_SCHEMES = {
    "fp8": "fp8_block",
    "int8": "w8a8_int8",
}
MTP_KV_CACHE_DTYPE = "bfloat16"


def progress(message: str) -> None:
    print(f"[calibration] {message}", file=sys.stderr, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is not a local directory: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus_contract(corpus_dir: Path) -> tuple[Path, str]:
    manifest = read_json(corpus_dir / CORPUS_MANIFEST)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Corpus manifest has no dataset object")
    expected_sha256 = dataset.get("arrow_sha256")
    dataset_path = corpus_dir / "calibration_dataset"
    arrow_path = dataset_path / "data-00000-of-00001.arrow"
    actual_sha256 = file_sha256(arrow_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Calibration dataset SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return dataset_path, actual_sha256


def checkpoint_tensor_map(checkpoint: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing weight_map in {index_path}")
        return {
            name: checkpoint / str(filename)
            for name, filename in weight_map.items()
        }, dict(index.get("metadata") or {})

    model_path = checkpoint / "model.safetensors"
    from safetensors import safe_open

    with safe_open(model_path, framework="pt", device="cpu") as file:
        return {name: model_path for name in file.keys()}, {}


def tensor_nbytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def restore_source_norms(source: Path, output: Path) -> None:
    """Restore source BF16 norms that llmcompressor changes during compression."""
    source_map, _ = checkpoint_tensor_map(source)
    selected = {
        f"model.{name.removeprefix('model.language_model.')}": path
        for name, path in source_map.items()
        if SOURCE_LANGUAGE_NORM.match(name)
    }

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    for source_shard in sorted(set(selected.values()), key=str):
        with safe_open(source_shard, framework="pt", device="cpu") as file:
            for output_name, path in selected.items():
                if path == source_shard:
                    source_name = (
                        "model.language_model."
                        f"{output_name.removeprefix('model.')}"
                    )
                    tensors[output_name] = file.get_tensor(source_name)

    output_map, metadata = checkpoint_tensor_map(output)
    shard_name = "model-source-norms.safetensors"
    save_file(tensors, output / shard_name, metadata={"format": "pt"})
    weight_map = {name: path.name for name, path in output_map.items()}
    weight_map.update(dict.fromkeys(tensors, shard_name))
    if "total_size" in metadata:
        metadata["total_size"] = int(metadata["total_size"]) + sum(
            tensor_nbytes(tensor)
            for name, tensor in tensors.items()
            if name not in output_map
        )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": metadata, "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def mtp_sidecar_config(source: Path) -> dict[str, Any]:
    source_config = read_json(source / "config.json")
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Source config has no text_config object")
    config = deepcopy(text_config)
    is_moe = int(config.get("num_experts", 0)) > 0
    config["architectures"] = ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"]
    # vLLM's explicit MTP path converts this text model type to qwen3_5_mtp
    # while retaining the text-only configuration used by the draft.
    config["model_type"] = "qwen3_5_moe_text" if is_moe else "qwen3_5_text"
    config["n_predict"] = int(config.get("mtp_num_hidden_layers", 1))
    quantization = source_config.get("quantization_config")
    if isinstance(quantization, dict):
        config["quantization_config"] = deepcopy(quantization)
    return config


def source_mtp_tensor_map(source: Path) -> dict[str, Path]:
    source_map, _ = checkpoint_tensor_map(source)
    mtp_map = {
        name: path
        for name, path in source_map.items()
        if MTP_TENSOR.match(name)
    }
    if not mtp_map:
        raise ValueError("Source checkpoint has no mtp.* tensors")
    if any("embed_tokens" in name or name.startswith("lm_head.") for name in mtp_map):
        raise ValueError("MTP source unexpectedly contains embeddings or LM head")
    return mtp_map


def prepare_mtp_stub(source: Path, stub: Path) -> tuple[dict[str, Path], list[str]]:
    mtp_map = source_mtp_tensor_map(source)
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "config.json").write_text(
        json.dumps(mtp_sidecar_config(source), indent=2, sort_keys=True) + "\n"
    )

    source_files = sorted(set(mtp_map.values()), key=str)
    linked_names: dict[Path, str] = {}
    for index, source_file in enumerate(source_files, start=1):
        suffix = (
            "mtp.safetensors"
            if len(source_files) == 1
            else f"mtp-{index:05d}-of-{len(source_files):05d}.safetensors"
        )
        linked_names[source_file] = suffix
        os.symlink(source_file, stub / suffix)
    weight_map = {
        name: linked_names[path]
        for name, path in mtp_map.items()
    }
    (stub / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2, sort_keys=True)
        + "\n"
    )
    fp8_modules = sorted(
        name.removesuffix(".weight_scale_inv")
        for name in mtp_map
        if name.endswith(".weight_scale_inv")
    )
    return mtp_map, fp8_modules


def package_mtp(source: Path, output: Path, weight_dtype: str) -> dict[str, Any]:
    destination = output / "mtp"
    with tempfile.TemporaryDirectory(
        prefix=".mtp-input-",
        dir=output.parent,
    ) as temporary:
        stub = Path(temporary)
        _, fp8_modules = prepare_mtp_stub(source, stub)
        from llmcompressor import model_free_ptq

        conversion = {}
        if fp8_modules:
            from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

            conversion["converter"] = FP8BlockDequantizer(targets=fp8_modules)
        model_free_ptq(
            model_stub=str(stub),
            save_directory=str(destination),
            scheme="FP8_BLOCK" if weight_dtype == "fp8" else "W8A8",
            ignore=list(MTP_IGNORE),
            max_workers=1,
            device="cpu",
            **conversion,
        )

    return {
        "mtp_path": "mtp",
        "mtp_weight_scheme": MTP_WEIGHT_SCHEMES[weight_dtype],
        "mtp_kv_cache_dtype": MTP_KV_CACHE_DTYPE,
    }


def build_recipe(weight_dtype: str, kv_cache_dtype: str) -> Any:
    """Build the exact modifier for the requested source/output combination."""
    from compressed_tensors.quantization import QuantizationArgs
    from llmcompressor.modifiers.quantization import QuantizationModifier

    kv_scheme = None
    if kv_cache_dtype == "fp8":
        kv_scheme = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            symmetric=True,
            dynamic=False,
            observer="static_minmax",
        )

    if weight_dtype == "int8":
        # W8A8's static per-channel weights and dynamic token activations are
        # RTN/data-free; the dataset is needed only when kv_scheme is present.
        return QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=list(IGNORE),
            kv_cache_scheme=kv_scheme,
        )

    if kv_cache_dtype == "fp8":
        # Requantize the BF16 CPU fallback explicitly. Transformers otherwise
        # preserves the source checkpoint's private save-time FP8 conversion,
        # which can hide whether the requested recipe was actually applied.
        return QuantizationModifier(
            targets="Linear",
            scheme="FP8_BLOCK",
            ignore=list(FP8_BLOCK_IGNORE),
            kv_cache_scheme=kv_scheme,
        )

    raise ValueError("No calibration recipe is required for this configuration")


def load_calibration_dataset(corpus_dir: Path) -> tuple[Any, str]:
    dataset_path, corpus_sha256 = load_corpus_contract(corpus_dir)
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if len(dataset) != KV_CALIBRATION_SAMPLES:
        raise ValueError(
            f"Calibration dataset must contain exactly "
            f"{KV_CALIBRATION_SAMPLES} rows; got {len(dataset)}"
        )
    return dataset, corpus_sha256


def manifest_for(
    *,
    args: argparse.Namespace,
    corpus_sha256: str | None,
    mtp_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recipe_version": RECIPE_VERSION,
        "model_scope": "language_model",
        "source_model": args.source_model_id,
        "source_revision": args.source_revision,
        "weight_dtype": args.weight_dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "corpus_sha256": corpus_sha256,
        "calibration_image_id": os.environ.get("CALIBRATION_IMAGE_ID"),
        **mtp_metadata,
    }


def calibrate(args: argparse.Namespace) -> None:
    source = require_directory(args.source_model, "source model")
    if args.weight_dtype == "int8":
        config = read_json(source / "config.json")
        if any(
            part.get("quantization_config") is not None
            for part in (config, config.get("text_config") or {})
        ):
            raise ValueError("INT8 calibration requires an unquantized BF16 source")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.weight_dtype == "fp8" and args.kv_cache_dtype == "auto":
        raise ValueError("No calibrated checkpoint is required for fp8/auto")
    progress(
        f"requested weights={args.weight_dtype} kv_cache={args.kv_cache_dtype}; "
        f"source={source} output={output}"
    )
    dataset = None
    corpus_sha256 = None
    if args.kv_cache_dtype == "fp8":
        if args.corpus_dir is None:
            raise ValueError("--corpus-dir is required for FP8 KV calibration")
        corpus = require_directory(args.corpus_dir, "corpus")
        dataset, corpus_sha256 = load_calibration_dataset(corpus)
        progress(
            f"validated calibration corpus: rows={KV_CALIBRATION_SAMPLES} "
            f"arrow_sha256={corpus_sha256}"
        )

    output.mkdir(parents=True, exist_ok=True)
    progress("loading calibration dependencies (torch, llmcompressor, transformers)")
    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress(f"loading tokenizer from {source}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(source),
        local_files_only=True,
    )
    progress(f"loading BF16 source model from {source}")
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        output_loading_info=True,
    )
    progress("BF16 source model loaded")
    if loading_info["missing_keys"]:
        raise ValueError(
            "Language model checkpoint is incomplete: "
            f"{next(iter(loading_info['missing_keys']))}"
        )
    # Transformers attaches the source FP8 loader's reverse converters to
    # the model even after dequantizing every parameter to BF16. If retained,
    # save_pretrained re-applies the source FP8 format after llmcompressor
    # runs and restores the multimodal source key prefix, overriding the
    # requested quantization and text-only output layout.
    model._weight_conversions = None
    recipe = build_recipe(args.weight_dtype, args.kv_cache_dtype)

    calibration_samples = (
        KV_CALIBRATION_SAMPLES if args.kv_cache_dtype == "fp8" else 0
    )
    progress(
        f"running llmcompressor oneshot: weights={args.weight_dtype} "
        f"kv_cache={args.kv_cache_dtype} samples={calibration_samples}"
    )
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        dataset=dataset,
        output_dir=str(output),
        save_compressed=True,
        batch_size=1,
        num_calibration_samples=calibration_samples,
        shuffle_calibration_samples=False,
        max_seq_length=(
            KV_MAX_SEQUENCE_LENGTH if args.kv_cache_dtype == "fp8" else None
        ),
        pad_to_max_length=False,
        dataloader_num_workers=0,
    )
    progress("oneshot completed; restoring source BF16 norms")
    restore_source_norms(source, output)
    progress("source norm restoration complete; packaging MTP sidecar")
    mtp_metadata = package_mtp(source, output, args.weight_dtype)

    manifest = manifest_for(
        args=args,
        corpus_sha256=corpus_sha256,
        mtp_metadata=mtp_metadata,
    )
    manifest_path = output / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    progress(f"calibration complete; manifest={manifest_path}")


def safetensors_headers(output: Path) -> dict[str, dict[str, Any]]:
    index_path = output / "model.safetensors.index.json"
    expected_names_by_file: dict[str, set[str]] | None = None
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing weight_map in {index_path}")
        checkpoint_files = sorted({str(value) for value in weight_map.values()})
        expected_names_by_file = {
            filename: {
                str(name)
                for name, mapped_filename in weight_map.items()
                if str(mapped_filename) == filename
            }
            for filename in checkpoint_files
        }
    elif (output / "model.safetensors").is_file():
        checkpoint_files = ["model.safetensors"]
    else:
        return {}

    headers: dict[str, dict[str, Any]] = {}
    for filename in checkpoint_files:
        path = output / filename
        try:
            with path.open("rb") as checkpoint:
                header_length_bytes = checkpoint.read(8)
                if len(header_length_bytes) != 8:
                    raise ValueError("truncated header length")
                header_length = struct.unpack("<Q", header_length_bytes)[0]
                header = json.loads(checkpoint.read(header_length))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Could not read safetensors header {path}: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"Expected a JSON object in safetensors header {path}")
        headers.update(
            {
                name: value
                for name, value in header.items()
                if name != "__metadata__"
                and isinstance(value, dict)
                and (
                    expected_names_by_file is None
                    or name in expected_names_by_file[filename]
                )
            }
        )
    if expected_names_by_file is not None:
        expected_names = set().union(*expected_names_by_file.values())
        missing_names = sorted(expected_names - set(headers))
        if missing_names:
            raise ValueError(
                f"Safetensors index references missing tensor {missing_names[0]}"
            )
    return headers


def module_is_ignored(module_name: str, weight_dtype: str) -> bool:
    patterns = IGNORE if weight_dtype == "int8" else FP8_BLOCK_IGNORE
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], module_name):
                return True
        elif module_name == pattern:
            return True
    return False


def mtp_module_is_ignored(module_name: str) -> bool:
    if module_name.endswith("norm"):
        return True
    return any(
        re.match(pattern.removeprefix("re:"), module_name)
        for pattern in MTP_IGNORE
    )


def parse_quantization_config(value: Any, label: str) -> Any:
    """Validate checkpoint quantization metadata with the pinned library API."""
    try:
        from compressed_tensors.quantization import QuantizationConfig

        return QuantizationConfig.model_validate(value)
    except Exception as exc:
        raise ValueError(f"{label} quantization_config is invalid: {exc}") from exc


def quantization_args_match(
    value: Any,
    expected: dict[str, Any],
) -> bool:
    if value is None:
        return False
    actual = value.model_dump(mode="json")
    return all(actual.get(key) == expected_value for key, expected_value in expected.items())


def validate_quantized_checkpoint(
    output: Path,
    weight_dtype: str,
    kv_cache_dtype: str,
    require_mtp_sidecar: bool,
    inspect_tensors: bool = True,
) -> list[str]:
    failures: list[str] = []
    try:
        config = read_json(output / "config.json")
    except (OSError, ValueError) as exc:
        return [str(exc)]

    raw_quantization = config.get("quantization_config")
    native_fp8_auto = (
        weight_dtype == "fp8"
        and kv_cache_dtype == "auto"
        and isinstance(raw_quantization, dict)
        and raw_quantization.get("quant_method") == "fp8"
    )
    if native_fp8_auto:
        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config

            quantization = Fp8Config.from_config(raw_quantization)
        except Exception as exc:
            return [f"config.json native FP8 quantization is invalid: {exc}"]
        if not quantization.is_checkpoint_fp8_serialized:
            failures.append("config.json is not a serialized FP8 checkpoint")
        if quantization.weight_block_size != [128, 128]:
            failures.append("native FP8 weight block size is not [128, 128]")
    else:
        try:
            quantization = parse_quantization_config(raw_quantization, "config.json")
        except ValueError as exc:
            return [str(exc)]

        groups = quantization.config_groups
        weight_scheme = (
            {"type": "int", "num_bits": 8, "strategy": "channel"}
            if weight_dtype == "int8" else
            {"type": "float", "num_bits": 8, "strategy": "block", "block_structure": [128, 128]}
        )
        activation_scheme = {"type": "int", "num_bits": 8, "strategy": "token", "dynamic": True}
        if not groups or not all(
            quantization_args_match(group.weights, weight_scheme)
            and (
                weight_dtype != "int8"
                or quantization_args_match(group.input_activations, activation_scheme)
            )
            for group in groups.values()
        ):
            failures.append(
                "all quantization groups must use channelwise INT8 weights "
                "with dynamic tokenwise INT8 activations" if weight_dtype == "int8" else
                "all quantization groups must use 128x128 blocked FP8 weights"
            )

    kv_scheme = getattr(quantization, "kv_cache_scheme", None)
    if kv_cache_dtype == "fp8":
        if not quantization_args_match(
            kv_scheme,
            {"type": "float", "num_bits": 8, "strategy": "tensor", "dynamic": False},
        ):
            failures.append("KV-cache config is not static tensorwise FP8")
    elif kv_scheme is not None:
        failures.append("unexpected KV-cache quantization scheme")
    if not inspect_tensors:
        return failures

    try:
        headers = safetensors_headers(output)
    except (OSError, ValueError) as exc:
        return failures + [str(exc)]
    if require_mtp_sidecar and any(name.startswith("mtp.") for name in headers):
        failures.append("language-model checkpoint contains root MTP tensors")
    if not native_fp8_auto and any(
        "weight_scale_inv" in name
        and not module_is_ignored(name.removesuffix(".weight_scale_inv"), weight_dtype)
        for name in headers
    ):
        failures.append("checkpoint retained source FP8 inverse weight scales")
    expected_dtype = "I8" if weight_dtype == "int8" else "F8_E4M3"
    quantized_weights = [
        name for name, value in headers.items()
        if name.endswith(".weight")
        and not module_is_ignored(name.removesuffix(".weight"), weight_dtype)
        and value.get("dtype") == expected_dtype
    ]
    scale_suffix = "_scale_inv" if native_fp8_auto else "_scale"
    if not quantized_weights:
        failures.append(f"checkpoint has no {expected_dtype} weights")
    elif any(f"{name}{scale_suffix}" not in headers for name in quantized_weights):
        failures.append("quantized weight has no scale")
    if kv_cache_dtype == "fp8":
        for key in ("k", "v"):
            if not any(name.endswith(f".{key}_scale") for name in headers):
                failures.append(f"checkpoint has no calibrated {key.upper()} scales")
    return failures


def validate_mtp_checkpoint(
    output: Path,
    weight_dtype: str,
    inspect_tensors: bool = True,
) -> list[str]:
    failures: list[str] = []
    try:
        config = read_json(output / "config.json")
    except (OSError, ValueError) as exc:
        return [str(exc)]

    if config.get("model_type") not in ("qwen3_5_text", "qwen3_5_moe_text"):
        failures.append("MTP config is not a Qwen3.5/3.6 text config")
    architectures = config.get("architectures")
    if architectures not in (["Qwen3_5MTP"], ["Qwen3_5MoeMTP"]):
        failures.append("MTP config has no supported MTP architecture")
    try:
        quantization = parse_quantization_config(config.get("quantization_config"), "MTP config.json")
    except ValueError as exc:
        return [str(exc)]

    groups = quantization.config_groups
    weight_scheme = (
        {"type": "float", "num_bits": 8, "strategy": "block", "block_structure": [128, 128]}
        if weight_dtype == "fp8" else
        {"type": "int", "num_bits": 8, "strategy": "channel", "symmetric": True, "dynamic": False}
    )
    activation_scheme = {
        "type": "int", "num_bits": 8, "strategy": "token", "symmetric": True, "dynamic": True,
    }
    if not groups or not all(
        quantization_args_match(group.weights, weight_scheme)
        and (weight_dtype == "fp8" or quantization_args_match(group.input_activations, activation_scheme))
        for group in groups.values()
    ):
        failures.append(
            "MTP groups are not all 128x128 blocked FP8" if weight_dtype == "fp8" else
            "MTP config is not channelwise W8A8"
        )
    if not inspect_tensors:
        return failures
    try:
        headers = safetensors_headers(output)
    except (OSError, ValueError) as exc:
        return failures + [str(exc)]
    if not headers:
        return failures + ["MTP checkpoint has no tensors"]
    if any(not name.startswith("mtp.") for name in headers):
        failures.append("MTP checkpoint contains a non-mtp tensor")
    if any("embed_tokens" in name or name.startswith("lm_head.") for name in headers):
        failures.append("MTP checkpoint duplicates embeddings or LM head")
    weights = {
        name: value for name, value in headers.items() if name.endswith(".weight")
    }
    if any(
        value.get("dtype") != "BF16" for name, value in weights.items()
        if mtp_module_is_ignored(name.removesuffix(".weight"))
    ):
        failures.append("MTP norms or router gates are not BF16")
    if any(name.endswith(".weight_scale_inv") for name in headers):
        failures.append("MTP retained source FP8 inverse weight scales")
    quantized_weights = {
        name: value for name, value in weights.items()
        if not mtp_module_is_ignored(name.removesuffix(".weight"))
    }
    expected_dtype = "F8_E4M3" if weight_dtype == "fp8" else "I8"
    if not quantized_weights or any(
        value.get("dtype") != expected_dtype for value in quantized_weights.values()
    ):
        failures.append(f"MTP quantizable weights are not all {expected_dtype}")
    if any(f"{name}_scale" not in headers for name in quantized_weights):
        failures.append("MTP quantized weight has no scale")
    fc = headers.get("mtp.fc.weight")
    if not isinstance(fc, dict) or fc.get("dtype") != expected_dtype:
        failures.append(f"MTP fc is not {expected_dtype}")
    return failures


def verify_existing(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    require_mtp = args.mtp_enabled or args.inspect_tensors
    failures = validate_quantized_checkpoint(
        output, args.weight_dtype, args.kv_cache_dtype, require_mtp,
        inspect_tensors=args.inspect_tensors,
    )
    if require_mtp:
        mtp_path = output / "mtp"
        if not mtp_path.is_dir():
            failures.append("missing mtp sidecar")
        else:
            failures.extend(
                validate_mtp_checkpoint(
                    mtp_path,
                    args.weight_dtype,
                    inspect_tensors=args.inspect_tensors,
                )
            )

    source_model_id = getattr(args, "source_model_id", None)
    source_revision = getattr(args, "source_revision", None)
    if source_model_id is not None or source_revision is not None:
        try:
            manifest = read_json(output / "calibration_manifest.json")
        except (OSError, ValueError) as exc:
            failures.append(f"source provenance requires calibration manifest: {exc}")
        else:
            for field, expected in (
                ("recipe_version", RECIPE_VERSION),
                ("source_model", source_model_id),
                ("source_revision", source_revision),
            ):
                if expected is None or manifest.get(field) != expected:
                    failures.append(f"calibration manifest {field} does not match")

    if failures:
        raise SystemExit("; ".join(failures))
    print(json.dumps({"ok": True, "output_dir": str(output)}, sort_keys=True))


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-model",
        required=True,
        help="Local checkpoint directory",
    )
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-dtype", choices=("fp8", "int8"), required=True)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        required=True,
    )
    parser.add_argument("--corpus-dir", help="Local corpus root; required for FP8 KV")
    parser.add_argument("--source-revision", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "calibrate",
        help="Run local INT8 W8A8 and/or FP8 KV calibration",
    )
    add_calibration_arguments(run)
    run.set_defaults(func=calibrate)

    verify = subparsers.add_parser(
        "verify-existing",
        help="Verify requested quantization metadata and tensors in a checkpoint",
    )
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--weight-dtype", choices=("fp8", "int8"), required=True)
    verify.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        required=True,
    )
    verify.add_argument("--mtp-enabled", action="store_true")
    verify.add_argument(
        "--inspect-tensors",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    verify.add_argument("--source-model-id")
    verify.add_argument("--source-revision")
    verify.set_defaults(func=verify_existing)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    args.func(args)


if __name__ == "__main__":
    main()
