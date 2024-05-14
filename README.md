# Distil-DIRE
Distil-DIRE is a lightweight version of DIRE, which can be used for real-time applications. Instead of calculating DIRE image directly, Distl-DIRE aims to reconstruct the features of corresponding DIRE image forwared by a image-net pretrained classifier with one-step noise of DDIM inversion. 
![overview](distil.png)


## Pretrained ADM diffusion model
We use image-net pretrained unconditional ADM diffusion model for feature reconstruction. You can download the pretrained model from the following link:
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

or you can use the following script to download the model:
```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt -O models/256x256-adm.pt
```

## Data Preparation
Before training the model on your own dataset, you need to prepare the dataset in the following format:
```bash
mydataset/train|val|test
└── images
    ├── fakes
    │   └──img1.png...
    ├── reals
        └──rimg1.png...
```

After preparing the dataset, you can calculate the epsilons and dire images for the dataset using the following script:
```bash
bash compute_dire_eps.sh
```

After running the script, you will have the following directory structure:
```bash
mydataset/train|val|test
└── images
    ├── fakes
    │   └──img1.png...
    ├── reals
        └──rimg1.png...
└── eps
    ├── fakes
    │   └──img1.pt...
    ├── reals
        └──rimg1.pt...
└── dire
    ├── fakes
    │   └──img1.png...
    ├── reals
        └──rimg1.png...
``` 
Note that we currently provide single-gpu preprocessing script. But you can modify the script to run on multiple gpus. For eps and dire calculation we set the DDIM steps to 20. This should be same when inference.

### Train
For training Distil-DIRE, we only provide single gpu training script. Be sure to have `datasets` directory in the root of the project and your dataset inside the `datasets` directory. 
```
python3 -m train --batch 128 --exp_name truemedia-distil-dire --datasets mydataset --epoch 100 --lr 1e-4
```

#### Fine-tuning
You can also fine-tune the model on your own dataset. For fine-tuning, you need to provide the path to the pretrained model. 
```bash
python3 -m train --batch 128 --exp_name truemedia-distil-dire --datasets mydataset --epoch 100 --lr 1e-4 --pretrained_weights YOUR_PRETRAINED_MODEL_PATH
```
 

### Test
For testing the model, you can use the following script:
```bash
python3 -m test --test True --datasets mydataset --pretrained_weights YOUR_PRETRAINED_MODEL_PATH
```


### with Docker 
```
export DOCKER_REGISTRY="YOUR_NAME" # Put your Docker Hub username here  
export DATE=`date +%Y%m%d` # Get the current date

# Build the Docker image for development
docker build -t "$DOCKER_REGISTRY/distil-dire:dev-$DATE" -f Dockerfile .


# Push your docker image to docker hub
docker login
docker push "$DOCKER_REGISTRY/distil-dire:dev-$DATE"

```


# Devl env 
```
export WORKDIR="YOUR_WORKDIR" # Put your working directory here
docker run --gpus=all --name=truemedia_gpu_all_distildire -v "$WORKDIR:/workspace/" -ti -e  "$DOCKER_REGISTRY/distil-dire:dev-$DATE"

# work inside the container (/workspace)
```

### Note
* This repo runs on ADM diffusion model (256x256, unconditional) trained on ImageNet 1k dataset and ResNet-50 classifier trained on ImageNet 1k dataset. 
* Minimum requirements: 1 GPU, 10GB VRAM


## Acknowledgments
Our code is developed based on [DIRE](https://github.com/ZhendongWang6/DIRE), [guided-diffusion](https://github.com/openai/guided-diffusion) and [CNNDetection](https://github.com/peterwang512/CNNDetection). Thanks for their sharing codes and models.

## Citation
If you find this work useful for your research, please cite our paper:
```
TODO: arxiv the paper
```
