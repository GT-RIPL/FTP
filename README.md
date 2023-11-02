# Fast Trainable Projection (FTP)
 This repo implements the DomainNet robust finetuning experiments in the paper [``Fast Trainable Projection for Robust Fine-Tuning (NeurIPS23)``](https://arxiv.org/abs/2310.19182).

## Overview
FTP learns per-layer projection constraints to encourage a fine-tuned model to stay close to its pre-trained initialization. It can be integrated into existing optimizers such as Adam and SGD, and used as drop-in replacement of them for better robust fine-tuning. In this repo, we provide the implementation of AdamP (Adam + FTP) and SGDP (SGD + FTP) in  `util/FTP.py`.

## Create conda environment
- The environment uses Ubuntu 18.04, Pytorch 1.7 supported on CUDA 11.x and python 3.8. 
```
cd FTP
conda env create -f environment.yml
conda activate ftp
```

## Download DomainNet
 - The script downloads the two pre-trained models under the `/datasets/domainnet` directory. Please change `DATA_DIR` in `download.sh` if you wish to download the data to a different folder. 
```
. ./datasets/download.sh
```

## Download Pre-trained Models (CLIP-ResNet50 and MoCoV3-ResNet50)
- The script downloads the two pre-trained models under the `"./pre_trained/"` directory.  Please change `MODEL_DIR` in `download_models.sh` if you wish to download the models to a different folder.
```
. ./datasets/download_models.sh
```

## Resources
- We used 4 RTX2080Ti gpus with 11G VRAM each. To fit the training script on smaller number of gpus, you can modify the `--gpu_per_node` flag in the launch script. 

## Launch Script
- Fine-tuning CLIP ResNet50 with SGDP (SGD + FTP) (100% data)
```
python main_finetune.py --arch clip_resnet50 --id FTP_clip --opt sgdp --lr 1e-2 --data_dir /datasets/domainnet --percent 100 --epoch 50 --gpu_per_node 4 --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt --batch_size 64 
```

- Fine-tuning CLIP ResNet50 with AdamP (Adam + FTP) (100% data)
```
python main_finetune.py --arch clip_resnet50 --id FTP_clip --opt adamp --lr 1e-4 --data_dir /datasets/domainnet --percent 100 --epoch 50 --gpu_per_node 4 --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt --batch_size 64 
```

- Fine-tuning CLIP ResNet50 with SGDP (10% data)
```
python main_finetune.py --arch clip_resnet50 --id FTP_clip_10 --opt sgdp --lr 1e-1 --data_dir /datasets/domainnet --percent 10 --epoch 150 --gpu_per_node 4 --load_pretrained ./pre_trained/clip_resnet50_pretrain.pt --batch_size 64 
```

- Fine-tuning MoCoV3 ResNet50 with SGDP (100% data)
```
python main_finetune.py --arch resnet50 --id FTP_moco --opt sgdp --lr 1e-2 --data_dir /datasets/domainnet --percent 100 --epoch 50 --gpu_per_node 4 --load_pretrained ./pre_trained/mocov3_resnet50_pretrain.tar --batch_size 64 
```

## Use Adamp/SGDP in Your Project
- AdamP (SGDP) is the Adam (SGD) variant with built-in FTP. It can easily intergrated into you project for robust fine-tuning of a pre-trained model. Make sure you have copied `util/FTP.py` into your own directory. Here is an example how you would incoroprate the `AdamP` optimizer into your project. 
 ```
from FTP import AdamP
# Initalize optimizer parameters
optimizer_params = {
    "lr": args.lr,
    "weight_decay": 0.0,
    "k": 1, 
    "exclude_set": {'module.head.weight','module.head.bias'}
} 

# Cache pre-trained model weights 
params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
params_anchor = copy.deepcopy(params_to_opt)
param_group = [{'params':params_to_opt,
                'pre': params_anchor, 
                'name': params_to_opt_name}]
optimizer = AdamP(param_group,**optimizer_params)
```
- The only special parameters in AdamP are `k` and `exclude_set`. `k` (a scalar between 0 and 1) controls the strength of regularization with 1 being the default and the strongest. `exclude_set` specifies which layers to exclude from projection constraints. Normally, it is recommened to exlcude layers with no corresponding pre-trained inialization such as the last linear layer. 

 