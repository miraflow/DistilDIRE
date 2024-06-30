#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 8 --exp_name y1-scaled-training --dataset_root /truemedia-eval/y1dataset --datasets_test truemedia-political --epoch 100 --lr 1e-4 --only_img True