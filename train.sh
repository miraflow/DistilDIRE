#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 64 --exp_name 240709-tot-finetune-from-scratch-fp16-distildire-w-prp-pos05 --datasets truemedia-total --datasets_test truemedia-total --epoch 50 --lr 1e-4 
