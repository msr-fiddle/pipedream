#
# This example Dockerfile illustrates a method to apply
# patches to the source code in NVIDIA's PyTorch
# container image and to rebuild PyTorch.  The RUN command
# included below will rebuild PyTorch in the same way as
# it was built in the original image.
#
# By applying customizations through a Dockerfile and
# `docker build` in this manner rather than modifying the
# container interactively, it will be straightforward to
# apply the same changes to later versions of the PyTorch
# container image.
#
# https://docs.docker.com/engine/reference/builder/
#
FROM nvcr.io/nvidia/pytorch:19.05-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-extra \
      && \
    rm -rf /var/lib/apt/lists/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Bring in changes from outside container to /tmp
# (assumes pre_hook.patch is in same directory as Dockerfile)
COPY pre_hook.patch /tmp

# Change working directory to PyTorch source path
WORKDIR /opt/pytorch

# Apply modifications and re-build PyTorch
RUN cd pytorch && patch -p1 < /tmp/pre_hook.patch && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    python setup.py install && python setup.py clean

# Reset default working directory
WORKDIR /workspace
