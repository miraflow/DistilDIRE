#! /bin/bash

torchrun --standalone --nproc_per_node 1 -m train --batch 128 --exp_name 240715-scale100k-half2 --datasets truemedia-total --datasets_test truemedia-external --epoch 100 --lr 1e-4 
