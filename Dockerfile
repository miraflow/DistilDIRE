FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 python3-dev libglib2.0-0 libgl1-mesa-glx \
    cmake tmux git curl wget gcc build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app    
ADD networks/ /app/networks
ADD utils/ /app/utils
ADD server.py /app/server.py
ADD celebahq_sdv2.pth /app/celebahq_sdv2.pth
ADD imagenet_adm.pth /app/imagenet_adm.pth
ADD 256x256-adm.pt /app/256x256-adm.pt
ADD custommodel.py /app/custommodel.py



