#! /bin/bash

torchrun --standalone --nproc_per_node 8 -m train --batch 32 --exp_name y1-scaled-training --datasets y1dataset --datasets_test truemedia-political --epoch 100 --lr 1e-4 --only_img True