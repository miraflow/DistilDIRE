# !/bin/bash 

# Imagenet
torchrun --standalone --nproc_per_node 8 -m train --batch 128 --exp_name truemedia-distil-dire-imagenet --datasets distil-train-imagenet --datasets_test distil-test-sdv1-imagenet --epoch 100 --lr 1e-4

# Celeba-HQ
torchrun --standalone --nproc_per_node 8 -m train --batch 128 --exp_name truemedia-distil-dire-celebahq --datasets distil-train-celebahq --datasets_test distil-test-midjourney-celebahq --epoch 100 --lr 1e-4
