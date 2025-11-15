# CUDA 11.4 on Ubuntu 20.04
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

ENV http_proxy="http://qtf5078:Saphi_1004Johnny0412@proxy.muc:8080"
ENV https_proxy="http://qtf5078:Saphi_1004Johnny0412@proxy.muc:8080"
ENV HTTP_PROXY="http://qtf5078:Saphi_1004Johnny0412@proxy.muc:8080"
ENV HTTPS_PROXY="http://qtf5078:Saphi_1004Johnny0412@proxy.muc:8080"

ARG DEBIAN_FRONTEND=noninteractive

# ---- base OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl bzip2 ca-certificates bash git build-essential ccache cmake \
      libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# helpful for faster rebuilds of CUDA/C++
ENV CCACHE_DIR=/root/.ccache
RUN ccache --max-size=10G || true

# ---- micromamba (as you had) ----
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

# ---- Python deps ----
# tip: the base already includes torch; avoid pinning torch in requirements.txt to prevent conflicts
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# ---- copy source and build CUDA extension in-place ----
# (Adjust CUDA archs for your machines; +PTX gives a forward-compat PTX fallback.)
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_HOME=/usr/local/cuda

# copy the whole project so the in-place build can emit .so next to sources
WORKDIR /workspace

ENV PYTHONPATH=/workspace/src:${PYTHONPATH}

# Use bash for subsequent RUNs
SHELL ["/bin/bash", "-lc"]

# Default to an interactive shell
CMD ["bash"]