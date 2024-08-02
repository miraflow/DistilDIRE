DATA_ROOT=("SRC")
SAVE_ROOT=("DST")

MODEL_PATH="models/256x256-adm.pt" # imagenet pretrained adm (unconditional, 512x512)
SAMPLE_FLAGS="--batch_size 1" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"

for i in 0 
do
    SAVE_FLAGS="--data_root ${DATA_ROOT[$i]} --save_root ${SAVE_ROOT[$i]}"
    echo "Running on ${DATA_ROOT[$i]} with save root ${SAVE_ROOT[$i]}"
    torchrun --standalone --nproc_per_node 1 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
done
