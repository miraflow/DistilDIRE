EXP_NAME="y1_celebahq_distill_naive_mse_ratio_1_1_augfixed"
DATASETS="celebahq"


torchrun --standalone --nproc-per-node=4 --master_port=25678 train.py --gpus 4,5,6,7 --exp_name $EXP_NAME --datasets $DATASETS 