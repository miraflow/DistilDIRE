#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 8 --exp_name 240708-ext-finetune-from-scratch-fp16-distildire-dnf --datasets truemedia-external --datasets_test truemedia-external --epoch 50 --lr 1e-4 --only_eps True