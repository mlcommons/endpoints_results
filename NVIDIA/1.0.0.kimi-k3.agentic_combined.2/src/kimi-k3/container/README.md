# Kimi K3 server container

## Build from source

On an ARM64 host with Docker, Git, Git LFS and Make installed, run from this
repository's root:

```bash
bash NVIDIA/src/kimi-k3/container/build.sh
```

The script clones public TensorRT-LLM's `feat/k3_26.08` branch at commit
[`45c5ae41`](https://github.com/NVIDIA/TensorRT-LLM/commit/45c5ae4150e51df678b2435457ff39679e8fd8d6),
the published image's (`TRTLLM_COMMIT` overrides it), builds TRT-LLM,
installs the public Dynamo wheels, applies the renderer and TCP patches, builds both frontend
variants and prepares the FlashInfer metadata. The resulting image is
`kimi-k3-dynamo:local`. Source builds print frontend hashes; set
`DYNAMO_FRONTEND_CORE_SHA256` to the matching hash when launching.

## Prebuilt container

NVIDIA's prebuilt image is `kimi-k3-dynamo@sha256:796bbbb499a61221fc04930179763f8be3c9af412782843bc9ec763cf2e4c2c8` (tag
`kimi-k3-dynamo:20261015`): the same Linux ARM64 image with Dynamo, patches and metadata already installed.
