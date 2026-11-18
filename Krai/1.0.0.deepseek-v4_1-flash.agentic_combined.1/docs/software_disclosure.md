# Software disclosure (§8.4)

Every point used this stack. Per-point server flags are in each point's `server_configs/`; build and reproduction
steps are in `src/gentic/README.md`.

| Component | Version |
|---|---|
| Serving framework | vLLM v0.30.0 (commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`), Rust frontend. Image `wongcannan/vllm-v0.30.0-nsys-withsrc:v1.0@sha256:6518e146fd3255a7649e8443f8a4057b8a978292bbc3bc23644ea68cf0dcccb6` (image ID `sha256:d600a4050af197cb65bccbd21ae20e62c667e2386d61edc6b3bda55d9d76e572`), built `FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` by `src/gentic/src/docker/vllm_server.dockerfile` |
| Accelerator libraries (in the vLLM image) | CUDA 13.0 (cudart 13.0.96), PyTorch 2.13.0+cu130, NCCL 2.30.7, cuDNN 9.20.0.48, cuBLAS 13.1.1.3, FlashInfer 0.6.18.post1, Triton 3.7.1; FlashMLA and DeepGEMM bundled with vLLM |
| GPU driver | NVIDIA 595.58.03 (CUDA 13.2) |
| Firmware | BIOS CUXD670_5.32_v2.05; GPU VBIOS 96.00.A5.00.03 |
| Operating system | Red Hat Enterprise Linux 9.4, kernel 5.14.0-427.13.1.el9_4.x86_64 |
| Container runtime | Docker 27.4.1, NVIDIA Container Toolkit 1.17.3 |
| Benchmark client | MLPerf Endpoints reference client, mlcommons/endpoints `d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`, unmodified. Image `wongcannan/mlperf-endpoints:v1.1@sha256:514e6360b981f2f59300a796f84585edaa4c6401dad352b062d4038dd9b32ea4` (image ID `sha256:a036d11620a656febddd995130a70688503dbcffadfa5d255096ad912ed8629a`), built by `src/gentic/src/docker/inference_endpoint.dockerfile` |
| Router | nginx 1.27.5 (`nginx:1.27-alpine`, image ID `sha256:6769dc3a703c719c1d2756bda113659be28ae16cf0da58dd5fd823d6b9a050ea`), reverse proxy only |
