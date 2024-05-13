#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
#eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="imagenet_distill_naive_mse_ratio_1_1"
DATASETS="imagenet"
PRETRAINED_WEIGHTS="/workspace/DIRE/checkpoints/imagenet/model_epoch_16.pth"
# PRETRAINED_WEIGHTS="/workspace/DIRE/imagenet_adm.pth"
#DATASETS_TEST="imagenet"
python3 test.py --gpus 0 --exp_name $EXP_NAME --datasets $DATASETS --pretrained_weights $PRETRAINED_WEIGHTS #datasets_test $DATASETS_TEST
