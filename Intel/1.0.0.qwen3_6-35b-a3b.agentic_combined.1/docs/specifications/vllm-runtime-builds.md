# vLLM runtime build provenance

The submission uses an upstream CPU CI image and two locally built CPU test images. Image identifiers and observed vLLM versions are retained exactly as recorded for the measured runs in each point's `serving_config.json` and the system software disclosures.

## Local test builds: PR #61329

Points **16, 64, and 256** used local builds from [vLLM PR #61329, "[CPU] Fix underflow in FP8 AMX attention"](https://github.com/vllm-project/vllm/pull/61329). Both local builds report source revision **7a0d68f29**, corresponding to [PR commit 7a0d68f299ad575e7516033a07d5badfafc9e998](https://github.com/vllm-project/vllm/pull/61329/commits/7a0d68f299ad575e7516033a07d5badfafc9e998).

| Measured points | Observed vLLM version | Measured local Docker image ID |
|---|---|---|
| 64 | `0.31.1rc1.dev366+g7a0d68f29` | `sha256:d45df48f5e737142f62e77d4960a79605af03a32afbec63c4959b2f9d2e9f46d` |
| 16, 256 | `0.32.0rc1.dev8+g7a0d68f29.d20261018` | `sha256:7b1435a0fb63969b12a6509c721e51c8ed1ad767655434fb95c166cd8672f422` |

The recorded image name `vllm/vllm-openai-cpu:test` was a local build tag, not a published release image. The tag was reused for distinct builds; the image IDs above identify the builds used for measurement. Neither local image has a registry digest.

PR #61329 merged on 2026-10-19 as commit `93de77c5ed7ecde1b92649318e95a1e1c45edf5e`. The measured local builds use the earlier PR commit cited above, rather than that final merge commit. They are not disclosed as the official `v0.31.0` release image.

## Upstream CPU CI build

Points **8, 12, 32, and 128** used the upstream CPU CI image at source commit `67ddac3ea8d5b36978187f92908f555ff71190e4`. Its observed vLLM version is `0.30.1rc1.dev57+g67ddac3ea`.

- Image tag: `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:67ddac3ea8d5b36978187f92908f555ff71190e4-cpu`
- Registry reference: `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo@sha256:f27c9b9ec391bfab3f8af4608a137679203d8ad930bcc57fd02a434fc6dbabc1`
- Measured local Docker image ID: `sha256:320822eb5201c388cccb926f76caa94f81ea20fba5579773a0637dc721ee33f6`
