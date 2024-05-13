EXP_NAME="y1_distilDIRE"

torchrun --standalone --nproc-per-node=8 test.py --gpus 0,1,2,3,4,5,6,7 --exp_name $EXP_NAME