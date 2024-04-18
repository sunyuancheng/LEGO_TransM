# 3D Molecular Pretraining via Localized Geometric Generation


## Overview

![arch](docs/model_overview.jpg)


## Installation

- Clone this repository

- Install the dependencies (Using [Anaconda](https://www.anaconda.com/), tested with CUDA version 11.0)

```shell
cd ./Transformer-M
conda env create -f requirement.yaml
conda activate Transformer-M
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==1.6.3
pip install torch_scatter==2.0.7
pip install torch_sparse==0.6.9
pip install azureml-defaults
pip install rdkit-pypi cython
python setup.py build_ext --inplace
python setup_cython.py build_ext --inplace
pip install -e .
pip install --upgrade protobuf==3.20.1
pip install --upgrade tensorboard==2.9.1
pip install --upgrade tensorboardX==2.5.1
```

## Pretraining on PCQM4Mv2
```shell
export data_path='./datasets/pcq-pos'               # path to data
export save_path='./uniform_noise/'                          # path to logs

export lr=2e-4                                      # peak learning rate
export warmup_steps=150000                          # warmup steps
export total_steps=1500000                          # total steps
export layers=12                                    # set layers=18 for 18-layer model
export hidden_size=768                              # dimension of hidden layers
export ffn_size=768                                 # dimension of feed-forward layers
export num_head=32                                  # number of attention heads
export batch_size=256                               # batch size for a single gpu
export dropout=0.0
export act_dropout=0.1
export attn_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.1                            # probability of stochastic depth
export noise_scale=0.2                              # noise scale
export mode_prob="1.0,0.0,0.0"                      # mode distribution for {2D+3D, 2D, 3D}
export dataset_name="PCQM4M-LSC-V2-3D"
export add_3d="true"
export num_3d_bias_kernel=128                       # number of Gaussian Basis kernels

bassh lego_pretrain.sh
```

We use "uniform_noise/lr-2e-4-end_lr-1e-9-tsteps-1500000-wsteps-150000-L12-D768-F768-H32-SLN-false-BS2048-SEED1-CLIP5-dp0.0-attn_dp0.1-wd0.0-dpp0.1-noise1.0-mr0.50-strategylego-lossfncos/checkpoint_best.pt" as the pretrained checkpoint.

## Evaluation on MoleculeNet
```shell
task_name="bbbp"  # bbbp as example, more supported task names can be found in the script
bash finetune_molnet.sh $task_name
```

## Evaluation on QM9
```shell
task_idx=0   # task idx ranges from 0 to 11
bash finetune_qm9_lego_noisynode.sh $task_idx
```
