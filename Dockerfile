FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip git
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /tmp/requirements.txt
WORKDIR /workspace
CMD ["/bin/bash"]
