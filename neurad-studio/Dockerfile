ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: Most commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    wget \
    ffmpeg \
    git \
    vim-tiny \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libmetis-dev \
    libqt5opengl5-dev \
    libsuitesparse-dev \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# 设置pip清华源
#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#    pip config set global.trusted-host mirrors.tuna.tsinghua.edu.cn

COPY docker/pins-common.txt /tmp/constraints.txt

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -c /tmp/constraints.txt "setuptools<70.0" pathtools promise pybind11

SHELL ["/bin/bash", "-c"]

RUN python3.10 -m pip install --no-cache-dir -c /tmp/constraints.txt \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

ENV TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
RUN python3.10 -m pip install --no-cache-dir -c /tmp/constraints.txt \
    git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# renderer 依赖（tyro 必备，外加常用组件）
RUN python3.10 -m pip install -c /tmp/constraints.txt \
    "uvicorn[standard]>=0.23,<0.30" "fastapi>=0.103,<0.112" \
    opencv-python-headless==4.9.* imageio==2.* tqdm pyyaml rich==13.* \
    einops==0.7.* typing_extensions==4.12.* jaxtyping==0.2.* beartype==0.17.* \
    tyro==0.7.*

RUN pip install -c /tmp/constraints.txt viser
RUN pip install -c /tmp/constraints.txt matplotlib
RUN pip install -c /tmp/constraints.txt tensorboard==2.12.0
RUN pip install -c /tmp/constraints.txt torchmetrics==1.1.2
RUN pip install -c /tmp/constraints.txt av2==0.2.1
RUN pip install -c /tmp/constraints.txt pyquaternion
RUN pip install -c /tmp/constraints.txt nuscenes-devkit==1.1.11

RUN git clone https://github.com/scaleapi/pandaset-devkit.git /opt/pandaset-devkit
WORKDIR /opt/pandaset-devkit/python
RUN pip install -e .

RUN pip install -c /tmp/constraints.txt zod
RUN pip install -c /tmp/constraints.txt pathos
RUN pip install -c /tmp/constraints.txt mediapy
RUN pip install -c /tmp/constraints.txt plotly
RUN pip install -c /tmp/constraints.txt gsplat
RUN pip install -c /tmp/constraints.txt pytorch_msssim
RUN pip install -c /tmp/constraints.txt splines
RUN pip install -c /tmp/constraints.txt nerfacc




# Install waymo-open-dataset
RUN python3.10 -m pip install --no-cache-dir waymo-open-dataset-tf-2-11-0==1.6.1
RUN pip install -c /tmp/constraints.txt numpy==1.26.*

# Install tzdata
#RUN python3.10 -m pip install --no-cache-dir tzdata

# Change working directory
#WORKDIR /workspace

#RUN git clone https://github.com/georghess/neurad-studio.git
#WORKDIR /workspace/neurad-studio

#RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \ 
#    pip install --no-build-isolation -e .[dev]
#WORKDIR /workspace

#RUN git clone --recurse-submodules https://github.com/carlinds/splatad.git
#WORKDIR /workspace/splatad
#RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
#    BUILD_NO_CUDA=1 python3.10 -m pip install -e .[dev]

# Make sure viser client is built
#RUN python -c "import viser; viser.ViserServer()"

# Install nerfstudio cli auto completion
#RUN ns-install-cli --mode install

# Bash as default entrypoint.
#CMD /bin/bash -l
