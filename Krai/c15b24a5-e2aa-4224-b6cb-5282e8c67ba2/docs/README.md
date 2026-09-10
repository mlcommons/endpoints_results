# MLPerf Endpoints - Llama3.1-8b

## Prerequisites
* Docker must be installed.
* The user running benchmarks must be incuded in the `docker` group.

## Define a work directory
```
export WORK_DIR=...
mkdir -p ${WORK_DIR}
```

## Install KRAI [AXS](https://github.com/krai/axs)

### Clone

Clone the AXS repository under `${WORK_DIR}`:
```
git clone https://github.com/krai/axs ${WORKS_DIR}/axs
```

### Init

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export PATH=${WORK_DIR}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORK_DIR}/work_collection

" >> ~/.bashrc
source ~/.bashrc
```

### Test
```
axs version
```

## Import public AXS repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2kiss
```

## Setup the HuggingFace token
```
export HF_TOKEN=...
```

## Benchmark
To start a benchmark set a target concurrency
```
export TARGET_CONCURRENCY=...
```
and the `max-num-seqs`:
```
export MAX_NUM_SEQS=...
```
then execute the following command:
```
axs byquery endpoints_output,task=llama3_1_8b,inference_server=vllm,backend=rocm,\
dp=8,tp=1,load_balancer=nginx,gpu_memory_utilization=0.95,\
server_docker_image=rocm/vllm,\
server_docker_image_tag=rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1,\
hf_token=${HF_TOKEN},\
max_num_seqs=${MAX_NUM_SEQS},target_concurrency=${TARGET_CONCURRENCY}
```

## Accessing results
The input and output files are stored at the path that can be obtained by running
```
axs byquery endpoints_output,task=llama3_1_8b,inference_server=vllm,backend=rocm,\
dp=8,tp=1,load_balancer=nginx,gpu_memory_utilization=0.95,\
server_docker_image=rocm/vllm,\
server_docker_image_tag=rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1,\
hf_token=${HF_TOKEN},\
max_num_seqs=${TARGET_CONCURRENCY},target_concurrency=${TARGET_CONCURRENCY} , get_path
```
