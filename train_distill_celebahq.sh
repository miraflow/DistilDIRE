#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
#eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="celebahq_distill_naive_mse_ratio_1_1"
DATASETS="celebahq"
#DATASETS_TEST="celebahq"
python3 train.py --gpus 4,5,6,7 --exp_name $EXP_NAME --datasets $DATASETS #datasets_test $DATASETS_TEST
