## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT=("/truemedia-eval/crawled-fakes")
SAVE_ROOT=("/truemedia-eval/y1dataset")

MODEL_PATH="models/512x512-adm.pt" # imagenet pretrained adm (unconditional, 512x512)
SAMPLE_FLAGS="--batch_size 4" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire False --compute_eps False"

for i in 0 
do
    SAVE_FLAGS="--data_root ${DATA_ROOT[$i]} --save_root ${SAVE_ROOT[$i]}"
    echo "Running on ${DATA_ROOT[$i]} with save root ${SAVE_ROOT[$i]}"
    torchrun --standalone --nproc_per_node 1 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
done

# torchrun --standalone --nproc_per_node 8 -m train --batch 128 --exp_name tm-global-scale --datasets y1-global-truemedia --epoch 40 --lr 1e-4