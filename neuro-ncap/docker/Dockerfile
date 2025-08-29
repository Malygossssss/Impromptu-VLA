FROM nvcr.io/nvidia/pytorch:23.10-py3
# this contains the base stuff:
# - torch == 2.1
# - cuda == 12.2.2
# - python 3.10

# install the rest of the dependencies
COPY docker/pins-common.txt /tmp/constraints.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -c /tmp/constraints.txt\
    nuscenes-devkit==1.1.11 \
    websocket-client==1.5 \
    httpx==0.25 \
    tyro==0.7.*

# we also want the nuplan devkit, but we need to install it from source git@github.com:motional/nuplan-devkit.git

# install the nuplan devkit
RUN git clone --branch nuplan-devkit-v1.2.1-release 'https://github.com/motional/nuplan-devkit.git' && \
    cd nuplan-devkit && \
    python -m pip install -c /tmp/constraints.txt -e .

# Fix issues with cv2
RUN pip uninstall -y $(pip list --format=freeze | grep opencv)
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -c /tmp/constraints.txt opencv-python
# fix fonts with PIL: _imagingft C module is not installed
RUN pip install -c /tmp/constraints.txt -U --force-reinstall Pillow
