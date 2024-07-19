#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 64 --exp_name 240718-distildire3 --datasets truemedia-external --datasets_test truemedia-external --epoch 10 --lr 1e-5 --pretrained_weights /home/ubuntu/y1/DistilDIRE/models/distil-240709-tot-model_epoch_20.pth 
