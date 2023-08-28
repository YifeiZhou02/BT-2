# BT-2
Research Code for "BT^2: Backward-compatible Training with Basis Transformation" (https://arxiv.org/abs/2211.03989).

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

## Example Experiments on Cifar 100
We provide training and evaluation experiment configurations for Cifar 100 in <code>./configs</code>. The following commands are backward compatible training experiments from ResNet50 to ResNet50 (with data change of 50 classes to 100 classes).

### Train Old Backbone Model
<code>python train_backbone.py --config configs/cifar100_backbone_old.yaml</code>

### Train Independent New Backbone Model
<code>python train_backbone.py --config configs/cifar100_backbone_new.yaml</code>

### Train Backward-compatible New Model with Basis Transformation
<code>python train_feature_transfer.py --config configs/cifar100_transfer.yaml</code>

### Train Backward-compatible New Model with BCT (https://arxiv.org/abs/2003.11942)
<code>python train_BCT.py --config configs/cifar100_BCT.yaml</code>

### Evaluation of Old/Old (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_old_old.yaml</code>

### Evaluation of New/New (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_new_new.yaml</code>

### Evaluation of New/Old (query feature/ gallery feature)
<code>python eval.py --config configs/cifar100_eval_old_new.yaml</code>

## Example Experiments on Imagenet
We provide training and evaluation experiment configurations for Imagenet 1k in <code>./configs</code>. Commands are similar to commands used for experiments in Cifar 100.

## Checkpoints and results:
We provide trained checkpoints using config files in example configurations and in the paper [here](https://drive.google.com/drive/folders/1-Rl5DyneZqyc5o_zWT4Zikq4hPiKTimD?usp=sharing). 

Example experiment results on Cifar100.
| Method     | Setting         | TOP1    | TOP5 | meanAP |
| :---        |    :----:   |          ---: |  ---: |  ---: |
| Independent    |    $\phi_{old}/\phi_{old}$ <br> $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 33.6<br>0.8<br>62.7   | 55.4<br>4.9<br>74.6 | 24.4<br>1.5<br>49.9 |
| BCT    |    $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 25.0<br>60.0   | 62.1<br>71.9 | 24.7<br>47.3 |
| $BT^2$(ours)    |    $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 38.7<br>62.4   | 64.6<br>75.1 | 27.7<br>50.5 |

Example experiment results on Imagenet1k.
| Method     | Setting         | TOP1    | TOP5 | meanAP |
| :---        |    :----:   |          ---: |  ---: |  ---: |
| Independent    |    $\phi_{old}/\phi_{old}$ <br> $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 40.9<br>0.1<br>67.9   | 55.8<br>0.5<br>81.4 | 33.6<br>0.2<br>52.3 |
| BCT    |    $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 44.3<br>65.3   | 66.4<br>80.0 | 34.6<br>54.0 |
| $BT^2$(ours)    |    $\phi_{new}/\phi_{old}$ <br> $\phi_{new}/\phi_{new}$     | 44.4<br>66.6   | 65.7<br>81.1 | 35.0<br>54.6 |

\* Some results are different from the paper due to sensitivity to hyperparameters and random seeds.

Experiment results of sequence updates on Imagenet1k.
| Method     | Setting         | TOP1    | TOP5 | meanAP |
| :---        |    :----:   |          ---: |  ---: |  ---: |
| Independent | $\phi_{alex}/\phi_{alex}$<br> $\phi_{vgg}/\phi_{vgg}$<br> $\phi_{res}/\phi_{res}$ <br> $\phi_{vit}/\phi_{vit}$ | 46.6<br>63.2<br>67.9<br>78.0 | 66.3<br>79.0<br>81.4<br>87.5 | 29.1<br>49.6<br>52.3<br>72.4 |
| BCT | $\phi_{vgg}/\phi_{alex}$<br>$\phi_{vgg}/\phi_{vgg}$<br>$\phi_{res}/\phi_{alex}$<br>$\phi_{res}/\phi_{vgg}$<br>$\phi_{res}/\phi_{res}$<br>$\phi_{vit}/\phi_{alex}$<br>$\phi_{vit}/\phi_{vgg}$<br>$\phi_{vit}/\phi_{res}$<br>$\phi_{vit}/\phi_{vit}$ | 54.4<br>58.4<br>46.0<br>48.9<br>64.3<br>54.9<br>57.5<br>70.3<br>73.9 | 74.1<br>75.4<br>71.9<br>75.2<br>79.1<br>82.0<br>84.1<br>85.1<br>86.0 | 36.2<br>47.0<br>30.6<br>44.4<br>52.7<br>36.3<br>50.5<br>57.0<br>65.8 |
| $BT^2$(ours) | $\phi_{vgg}/\phi_{alex}$<br>$\phi_{vgg}/\phi_{vgg}$<br>$\phi_{res}/\phi_{alex}$<br>$\phi_{res}/\phi_{vgg}$<br>$\phi_{res}/\phi_{res}$<br>$\phi_{vit}/\phi_{alex}$<br>$\phi_{vit}/\phi_{vgg}$<br>$\phi_{vit}/\phi_{res}$<br>$\phi_{vit}/\phi_{vit}$ | 56.5<br>61.0<br>56.7<br>61.5<br>66.6<br>57.9<br>62.5<br>72.0<br>75.6 | 75.6<br>77.2<br>78.5<br>80.8<br>80.8<br>83.5<br>86.5<br>87.0<br>87.4 | 37.1<br>47.5<br>37.2<br>50.6<br>56.8<br>37.6<br>52.7<br>60.6<br>68.0 |

## Cite our paper
```bibtex
@misc{zhou2023bt2,
      title={$BT^2$: Backward-compatible Training with Basis Transformation}, 
      author={Yifei Zhou and Zilu Li and Abhinav Shrivastava and Hengshuang Zhao and Antonio Torralba and Taipeng Tian and Ser-Nam Lim},
      year={2023},
      eprint={2211.03989},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
