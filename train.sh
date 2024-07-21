#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 64 --exp_name 240718-distildire3 --datasets TRAIN --datasets_test TEST --epoch 10 --lr 1e-5
