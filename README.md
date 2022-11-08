# BT-2
Research Code for "BT^2: Backward-compatible Training with Basis Transformation".

Code adapted from https://github.com/apple/ml-fct.

![dimension_reduction](https://user-images.githubusercontent.com/83000332/200150995-6e64bdd9-7e9b-45c2-8917-422c30b9263f.png)


## Requirements
We suggest using Conda virtual environments, please run:

```bash
conda env create -f environment.yml
conda activate sm86
```

## Dataset Preparation
Make dataset and checkpoint directories.
```bash
mkdir data_store
mkdir checkpoints
```
### Cifar 100
Please refer to https://www.cs.toronto.edu/~kriz/cifar.html for downloading Cifar 100.

### Imagenet 1k
Please refer to https://www.image-net.org/challenges/LSVRC/2012/index.php for downloading the Imagenet 1k.

## Sample Experiments on Cifar 100
We provide training and evaluation experiment configurations for Cifar 100 in <code>./configs</code>. The following commands are backward compatible training experiments from ResNet50 to VitB16.

### Train Old Backbone Model
<code>python train_backbone.py --config configs/cifar100_backbone_old.yaml</code>

### Train Independent New Backbone Model
<code>python train_backbone.py --config configs/cifar100_backbone_new.yaml</code>

### Train Backward-compatible New Model with Basis Transformation
<code>python train_feature_transfer.py --config configs/cifar100_transfer_vit.yaml</code>

### Train Backward-compatible New Model with BCT (https://arxiv.org/abs/2003.11942)
<code>python train_BCT.py --config configs/cifar100_BCT.yaml</code>

### Evaluation of Old/Old (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_old_old.yaml</code>

### Evaluation of New/New (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_new_new.yaml</code>

### Evaluation of New/Old (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_old_new.yaml</code>

## Sample Experiments on Imagenet
We provide training and evaluation experiment configurations for Imagenet 1k in <code>./configs</code>. The provided configurations are for backward compatible training experiments from ResNet50 to VitB16. Commands are similar to commands used for experiments in Cifar 100
