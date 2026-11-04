"""Freeze launch inputs and render the per-job SFlow bootstrap/finalizer."""
import os
from pathlib import Path
import shlex
import shutil
import sys

import yaml


def main():
    run_dir = Path(sys.argv[1])
    version = sys.argv[2]
    point = Path(sys.argv[3])
    workflow = Path(sys.argv[4])
    tools = Path(__file__).resolve().parent
    model_dir = tools.parent

    # The revision goes inside a single-quoted requirement in the template.
    if any(char in version for char in "'\\\n\r"):
        raise ValueError("Invalid SFlow revision")

    inputs = run_dir / "inputs"
    for source, name in (
        (point.parent / "slurm_env.yaml", "slurm_env.yaml"),
        (point / "dsv4.1_flash_config.yaml", "config.yaml"),
        (workflow, "workflow.yaml"),
    ):
        shutil.copy2(source, inputs / name)

    settings = yaml.safe_load((inputs / "slurm_env.yaml").read_text())["variables"]
    scratch = Path(os.environ.get("SCRATCH_DIR") or settings["SCRATCH_DIR"]["value"]).expanduser().resolve()
    setup_script = inputs / "setup_swebench_service.sbatch"
    if point.parent.name.startswith("GB300-"):
        shutil.copy2(model_dir / "setup_swebench_service.sbatch", setup_script)

    values = {
        "@INPUT_SLURM@": inputs / "slurm_env.yaml",
        "@INPUT_WORKFLOW@": inputs / "workflow.yaml",
        "@INPUT_POINT@": inputs / "config.yaml",
        "@SBATCH_OUT@": run_dir / "slurm_%j.out",
        "@SBATCH_ERR@": run_dir / "slurm_%j.err",
        "@OUTPUT_DIR@": run_dir / "output",
        "@WORKSPACE_DIR@": run_dir / "workspace",
        "@MERGED_YAML@": run_dir / "slurm_job.yaml",
        "@RECIPE_DIR@": point.parent,
        "@MODEL_DIR@": model_dir,
        "@MODEL_CHECKPOINT_INDEX@": scratch / "models/DeepSeek-V4.1-Flash/model.safetensors.index.json",
        "@SERVICE_SETUP_SCRIPT@": setup_script,
        "@SERVICE_SETUP_LOG@": run_dir / "swebench_setup.log",
    }
    body = (tools / "reference_batch_runtime.sh.in").read_text()
    for tag, value in values.items():
        body = body.replace(tag, shlex.quote(str(value)))
    body = body.replace("@SFLOW_VERSION@", version)

    script = run_dir / "slurm_job.sh"
    lines = script.read_text().splitlines(keepends=True)
    last = max(index for index, line in enumerate(lines) if line.startswith("#SBATCH"))
    script.write_text("".join(lines[:last + 1]) + "\n" + body)


if __name__ == "__main__":
    main()
