EXP_NAME="y1_imagenet_distill_naive_mse_ratio_1_1"
DATASETS="imagenet"


torchrun --standalone --nproc-per-node=4 train.py --gpus 0,1,2,3 --exp_name $EXP_NAME --datasets $DATASETS