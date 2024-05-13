# !/bin/bash

docker run --gpus=all --ipc=host -v ~/Projects/yewon/:/workspace/ -ti -e --name=yewon_gpu_all_dire miraflow/yewon-dire:dev
