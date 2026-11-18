ARG VLLM_BASE_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_BASE_TAG}

ARG VLLM_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607
# bash so the official nsys install steps' `source` works; pipefail so a failed key download stops the build here.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Nsight Systems CLI, following NVIDIA's official install guide (devtools repo, CLI only).
# Its postinst links /usr/local/bin/nsys.
RUN apt update \
    && apt install -y --no-install-recommends gnupg2 wget ca-certificates git \
    && wget -O- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
       | gpg --dearmor \
       | tee /usr/share/keyrings/nvidia-devtools-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /" \
       | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt update \
    && apt install -y nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/* \
    && nsys --version

# vLLM source at the pinned version's commit.
RUN git clone https://github.com/vllm-project/vllm.git /vllm-workspace/vllm \
    && git -C /vllm-workspace/vllm checkout "${VLLM_COMMIT}"