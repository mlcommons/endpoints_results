"""Write the SWE-bench image list for a bench config, using the endpoints client's own code.

Instance ids come from the same path the client uses at run time: the bench yaml's swe_bench dataset is
loaded through the client's dataset loader, then the first `num_instances` rows are taken, as
SWEBenchScorer does. Each id is mapped to its image with mini-swe-agent's own naming function (service venv).

Runs inside the client image, which has both venvs. param_search_multireplica_mlperf.sh calls it as:
  python /gentic/src/e2e_benchmarking/scripts/swebench_image_manifest.py BENCH_YAML OUT_TXT
"""
import json
import subprocess
import sys

from inference_endpoint.config.schema import BenchmarkConfig, DatasetType
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.evaluation.swe_bench_scorer import SWEBenchScorer

SERVICE_PYTHON = "/opt/endpoints/src/inference_endpoint/evaluation/swebench_service/.venv/bin/python"

bench_yaml, out_path = sys.argv[1], sys.argv[2]
config = BenchmarkConfig.from_yaml_file(bench_yaml)
dataset = next(
    d for d in config.datasets
    if d.type == DatasetType.ACCURACY and d.accuracy_config and d.accuracy_config.eval_method.value == "swe_bench_scorer"
)
extras = dataset.accuracy_config.extras or {}
loader = DataLoaderFactory.create_loader(dataset, num_repeats=1, **SWEBenchScorer.dataset_loader_kwargs(extras))
loader.load(api_type=config.endpoint_config.api_type, model_params=dataset.effective_generation_config(config.model_params))
num_instances = SWEBenchScorer.external_sample_count(extras) or len(loader.dataframe)
instance_ids = [str(i) for i in loader.dataframe.iloc[:num_instances]["instance_id"].tolist()]

# Image names from mini-swe-agent itself, so a naming change upstream is picked up automatically.
names = subprocess.run(
    [SERVICE_PYTHON, "-c",
     "import json, sys; from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name as f; "
     "print('\\n'.join(f({'instance_id': i}) for i in json.load(sys.stdin)))"],
    input=json.dumps(instance_ids), capture_output=True, text=True, check=True,
).stdout
# mini-swe-agent prints a startup banner on import; keep only the image refs.
names = [line for line in names.splitlines() if line.startswith("docker.io/")]
if len(names) != len(instance_ids):
    sys.exit(f"expected {len(instance_ids)} image names, got {len(names)}")
with open(out_path, "w") as f:
    f.write("\n".join(names) + "\n")
print(f"{len(instance_ids)} images ({extras.get('subset', 'verified')}/{extras.get('split', 'test')}) -> {out_path}", file=sys.stderr)