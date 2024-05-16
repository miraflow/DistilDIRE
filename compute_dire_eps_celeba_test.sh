## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT=("/workspace/DistilDIRE/datasets/celebahq-test-dalle2" "/workspace/DistilDIRE/datasets/celebahq-test-if" "/workspace/DistilDIRE/datasets/celebahq-test-midjourney" "/workspace/DistilDIRE/datasets/celebahq-test-sdv2")
SAVE_ROOT=("/workspace/DistilDIRE/datasets/distil-test-dalle2-celebahq" "/workspace/DistilDIRE/datasets/distil-test-if-celebahq" "/workspace/DistilDIRE/datasets/distil-test-midjourney-celebahq" "/workspace/DistilDIRE/datasets/distil-test-sdv2-celebahq")

MODEL_PATH="models/256x256-adm.pt" # imagenet pretrained adm (unconditional, 256x256)
SAMPLE_FLAGS="--batch_size 16" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"

for i in 0 1 2 3
do
    SAVE_FLAGS="--data_root ${DATA_ROOT[$i]} --save_root ${SAVE_ROOT[$i]}"
    echo "Running on ${DATA_ROOT[$i]} with save root ${SAVE_ROOT[$i]}"
    torchrun --standalone --nproc_per_node 8 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
done

