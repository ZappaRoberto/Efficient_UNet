<a href="https://pytorch.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="80" />
</a>

# Efficent Unet

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Efficent Unet (EUnet) is a improved version of the original U-Net architectures to approach [Semantic Segmentation](https://en.wikipedia.org/wiki/Image_segmentation) in real time (>= 60 FPS) on low computing power hardware with hight fedelity.


## Table Of Content

- [Architecture Analysis](#Architecture-Analysis)
- [Dataset](#Dataset)
    - [COCO Semantic Segmentation](#COCO-Semantic-Segmentation)
    - [FSCOCO](#FSCOCO)
- [Training](#Training)
- [Result Analysis](#Result-Analysis)
- [Limitation and further improvement](#Limitation-and-further-improvement)


## Architecture Analysis

The overall architecture of this network is shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/Efficent_Unet/blob/main/img/architecture.png" />
</p>

The Architecture consists of two building blocks: Downblock and Upblock.
- The Downblock uses an [SPD-Conv](https://github.com/LabSAINT/SPD-Conv) pooling layer, followed by two depthwise separable convolutions and an [LKA](https://arxiv.org/abs/2202.09741) attention layer.<br/>
- The Upblock is similar, except that a ConvTranspose is used to increase the resolution instead of a pooling layer. In addition, the in_channels are managed to enable concatenation of the skip connection without the need for cropping.


## Dataset

The dataset used for the training part are the [COCO Semantic Segmentation](https://cocodataset.org/#download) and [FSCOCO](https://www.fsoco-dataset.com/). 


### COCO Semantic Segmentation

COCO 2017 dataset is made by 123k images already splitted in 118K for training and 5K for testing


### FSCOCO

FSCOCO segmentation dataset consist in 1516 images with the relative mask in Supervisely format. I use 20% of the dataset for testing and 80% for training. 

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Training

The training of the network could be splitted in two part: the first one a pre-training phase and the second one the finethuning of the model.
for all the phase I use the following hyperparameters and settings:</br>
- learning_rate: 1e-4,
- batch_size: 64,
- optimizer: [Lion](https://arxiv.org/abs/2302.06675),
- weight_decay: 1e-2,
- scheduler: One Cycle Learning with a max learning rate of 1e-4,
- num_epochs: 1000,
- patience: 20,
</br>
I run all my experiments on my personal computer equipped by an RTX 4090 and i7-13700k with 32gb of RAM

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Result Analysis

For computational limitation I trained the models only with depth 9. the result showed below are the test error of my implementation and paper implementation.

### Text Classification

|     Pool Type     |  My Result  | Paper Result |
| :---------------: | :---------: | :----------: |
| Convolution       |    32.57    |     28.10    |
| KMaxPooling       |    28.92    |     28.24    |
| MaxPooling        |    28.40    |     27.60    |


### Sentiment Analysis

|     Pool Type     |  My Result  | Paper Result |
| :---------------: | :---------: | :----------: |
| Convolution       |    40.35    |     38.52    |
| KMaxPooling       |    38.58    |     39.19    |
| MaxPooling        |    38.45    |     37.95    |

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Limitation and further improvement

After training run main.py file changing variable **`WEIGHT_DIR`** with the local directory where the weight are saved

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>
