#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 64 --exp_name 240709-tot-finetune-from-scratch-fp16-distildire-w-prp-pos2 --datasets truemedia-total --datasets_test truemedia-external --epoch 50 --lr 1e-4 --pretrained_weights /home/ubuntu/y1/DistilDIRE/experiments/240709-tot-finetune-from-scratch-fp16-distildire-w-prp/ckpt/model_epoch_4.pth 
