#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
#eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="imagenet_distill_naive_mse_ratio_1_1"
DATASETS="imagenet"
#DATASETS_TEST="celebahq"
python3 train.py --gpus 0,1,2,3 --exp_name $EXP_NAME --datasets $DATASETS #datasets_test $DATASETS_TEST
