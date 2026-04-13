# 推荐使用 CUDA 12.4 或更高版本，配合 Ubuntu 22.04
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python3-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
# 将原来的那行改为以下命令
RUN pip install --no-cache-dir --upgrade uv

# copy dependency files first
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# copy local files to /cleanrl
COPY . /cleanrl
WORKDIR /cleanrl

# install dependencies and package
RUN uv pip install . --system

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh

# Default to interactive shell, but allow entrypoint.sh to be used when needed
CMD ["/bin/bash"]