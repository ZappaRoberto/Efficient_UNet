<a href="https://pytorch.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="80" />
</a>

# Efficent Unet

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Efficent Unet (EUnet) is a improved version of the original [U-Net](https://arxiv.org/abs/1505.04597) architectures to approach [Semantic Segmentation](https://en.wikipedia.org/wiki/Image_segmentation) in real time (>= 60 FPS) on low computing power hardware with hight fedelity.


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

The network training process can be divided into two parts: the first phase involves pre-training, while the second phase involves fine-tuning of the model. For all phases, the following hyperparameters and settings were utilized:
</br>
- learning_rate: 1e-4,
- batch_size: 64,
- optimizer: [Lion](https://arxiv.org/abs/2302.06675),
- weight_decay: 1e-2,
- scheduler: [One Cycle Learning](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) with a max learning rate of 1e-4,
- num_epochs: 1000,
- patience: 20,

I utilize the [dice score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and [intersect over union](https://en.wikipedia.org/wiki/Jaccard_index) as the primary metrics to measure the improvement of the networks. All the metrics employed in this study were obtained from [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest), as they are extensively tested and reliable.
</br>
I run all the experiments on my personal computer equipped by an RTX 4090 and i7-13700k with 32gb of RAM

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Result Analysis


### COCO 2017 Segmentation dataset

I compare my result with the SOTA networks that can be founded [here](https://paperswithcode.com/sota/semantic-segmentation-on-coco-1)

<p align="center">
  <img src="https://github.com/ZappaRoberto/Efficent_Unet/blob/main/img/result.png" />
</p>

|     Networks    |  Year  |  N Parameters  |   IoU   |
|  :------------: | :----: | :------------: |  :----: |
|    OneFormer    |  2022  |      223M      |   **68.1**  |
|    OneFormer    |  2022  |      219M      |   67.4  |
|    Mask2Former  |  2021  |      216M      |   67.4  |
|    MaskFormer   |  2021  |      212M      |   64.8  |
|    SegCLIP      |  2022  |       ?        |   26.5  |
|    EUnet        |  2023  |      **16M**       |   64.9  |


### FSCOCO


<p align="center">
  <img src="https://github.com/ZappaRoberto/Efficent_Unet/blob/main/img/result2.png" />
</p>

|     Networks    |  Year  |  N Parameters  |   IoU   |   Dice Score   |
|  :------------: | :----: | :------------: |  :----: |  :----------:  |
|    **EUnet**    |**2023**|    **16M**     | **?**|    **90.7**    |


<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Limitation and further improvement

Try different normalization layer like groupnorm or layernorm <br/>
Try newer verion of LKA <br/>


<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>
