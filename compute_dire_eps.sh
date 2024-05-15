## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT="/workspace/DistilDIRE/datasets/imagenet-train"
SAVE_ROOT="/workspace/DistilDIRE/datasets/distil-train-imagenet"

MODEL_PATH="models/256x256-adm.pt" # imagenet pretrained adm (unconditional, 256x256)
SAMPLE_FLAGS="--batch_size 16" # ddim20 is forced
SAVE_FLAGS="--data_root $DATA_ROOT --save_root $SAVE_ROOT"
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"
python3 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $SAVE_FLAGS $PREPROCESS_FLAGS $SAMPLE_FLAGS