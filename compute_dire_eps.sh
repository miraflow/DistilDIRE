## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT=("/workspace/DistilDIRE/datasets/imagenet-train" "/workspace/DistilDIRE/datasets/imagenet-test" "/workspace/DistilDIRE/datasets/celebahq-train")
SAVE_ROOT=("/workspace/DistilDIRE/datasets/distil-train-imagenet" "/workspace/DistilDIRE/datasets/distil-test-imagenet" "/workspace/DistilDIRE/datasets/distil-train-celebahq")

MODEL_PATH="models/256x256-adm.pt" # imagenet pretrained adm (unconditional, 256x256)
SAMPLE_FLAGS="--batch_size 16" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"

for i in 0 1 2
do
    SAVE_FLAGS="--data_root ${DATA_ROOT[$i]} --save_root ${SAVE_ROOT[$i]}"
    echo "Running on ${DATA_ROOT[$i]} with save root ${SAVE_ROOT[$i]}"
    torchrun --standalone --nproc_per_node 8 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
done

