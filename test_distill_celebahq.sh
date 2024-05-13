#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
#eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="celebahq_distill_naive_mse_ratio_1_1"
DATASETS="celebahq"
PRETRAINED_WEIGHTS="/workspace/DIRE/checkpoints/celebahq/model_epoch_2.pth"
# PRETRAINED_WEIGHTS="/workspace/DIRE/celebahq_sdv2.pth"
#DATASETS_TEST="celebahq"
torchrun --standalone --nproc-per-node=1 test.py --gpus 0 --exp_name $EXP_NAME --datasets $DATASETS --pretrained_weights $PRETRAINED_WEIGHTS #datasets_test $DATASETS_TEST
