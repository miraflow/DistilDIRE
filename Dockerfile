FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 python3-dev libglib2.0-0 libgl1-mesa-glx \
    cmake tmux git curl wget gcc build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace





