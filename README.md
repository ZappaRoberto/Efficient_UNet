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
    - [FSACOCO 🤫](#FSACOCO-🤫)
- [Training](#Training)
- [Result Analysis](#Result-Analysis)


## Architecture Analysis

The overall architecture of this network is shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/Efficent_Unet/blob/main/img/architecture.png" />
</p>

The Architecture consists of two building blocks: Downblock and Upblock.
- The Downblock uses an [SPD-Conv](https://github.com/LabSAINT/SPD-Conv) pooling layer, followed by two depth-wise convolutions and an [LKA](https://arxiv.org/abs/2202.09741) attention layer.<br/>
- The Upblock is similar, except that a ConvTranspose is used to increase the resolution instead of a pooling layer. In addition, the in_channels are managed to enable concatenation of the skip connection without the need for cropping.


## Dataset

The dataset used for the training part are the [Yahoo! Answers Topic Classification](https://www.kaggle.com/datasets/b78db332b73c8b0caf9bd02e2f390bdffc75460ea6aaaee90d9c4bd6af30cad2) and a subset of [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) that can be downloaded [here](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=share_link&resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg). The vocabolary used is the same used in the paper: **"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%^&*~‘+=<>()[]{} "**. I choose to use 0 as a value for padding and 69 as a value for unknown token. All this datasets are maneged by **`Dataset class`** inside dataset.py file. 


### Yahoo! Answer topic classification

The Yahoo! Answers topic classification dataset is constructed using the 10 largest main categories. Each class contains 140000 training samples and 6000 testing samples. Therefore, the total number of training samples is 1400000, and testing samples are 60000. The categories are:

* Society & Culture
* Science & Mathematics
* Health
* Education & Reference
* Computers & Internet
* Sports
* Business & Finance
* Entertainment & Music
* Family & Relationships
* Politics & Government


### Amazon Reviews

The Amazon Reviews dataset is constructed using 5 categories (star ratings).

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Training

> **Warning**
> 
> Even if it can be choosen the device between cpu or GPU, I used and tested the training part only with GPU.

First things first, at the beginning of train.py file there are a some useful global variable that manage the key settings of the training.

```python

LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MAX_LENGTH = 1024
NUM_EPOCHS = 1
PATIENCE = 40
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "dataset/amazon/train.csv"
TEST_DIR = "dataset/amazon/test.csv"
```

> **Note**
> 
> Change **`TRAIN_DIR`** and **`TEST_DIR`** with your datasets local position.

The train_fn function is build to run one epoch and return the average loss and accuracy of the epoch.

```python

def train_fn(epoch, loader, model, optimizer, loss_fn, scaler):
    # a bunch of code
    return train_loss, train_accuracy
```

The main function is build to inizialize and manage the training part until the end.

```python

def main():
    model = VDCNN(depth=9, n_classes=5, want_shortcut=True, pool_type='vgg').to(DEVICE)
    # training settings
    for epoch in range(NUM_EPOCHS):
        # run 1 epoch
        # check accuracy
        # save model
        # manage patience for early stopping
    # save plot
    sys.exit()
```

> **Note**
> 
> Remember to change **`n_classes`** from 5 to 10 if you use Amazon dataset or Yahoo! Answer dataset.

**`get_loaders`**, **`save_checkpoint`**, **`load_checkpoint`**, **`check_accuracy`** and **`save_plot`**  are function used inside tran.py that can be finded inside utils.py.

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

## How to use

After training run main.py file changing variable **`WEIGHT_DIR`** with the local directory where the weight are saved

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>
