import os
import sys
from random import sample
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from matplotlib import pyplot as plt
import albumentations as A
from tempfile import TemporaryFile

# TODO: Augmentation

"""
Faccio l'augmentation sul training set e salvo tutto in un unico file numpy
"""


def splitTrainingTest():
    images = os.listdir("FsCocoDataset/images")
    print(len(images))
    n = int(len(images) * 20 / 100)
    print(n)
    samp = sample(images, n)
    for sa in tqdm(samp):
        shutil.move("FsCocoDataset/images/" + sa, "Dataset/Test/images")
        shutil.move("FsCocoDataset/segmask/" + sa, "Dataset/Test/segmask")


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[0].show()
        ax[1].imshow(mask)
        ax[1].show()
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        plt.show()


def augmentation(n):
    images = os.listdir("FsCocoDataset/images")
    images = [ele for ele in images if ele != '.DS_Store']  # Mac bullshit
    aug = A.Resize(height=256, width=256)
    for iteration, img in enumerate(tqdm(images)):
        image = np.asarray(Image.open('FsCocoDataset/images/{}'.format(img)))
        mask = np.asarray(Image.open('FsCocoDataset/segmask/{}'.format(img)))
        augmented = aug(image=image, mask=mask)
        image_padded = augmented['image']
        mask_padded = augmented['mask']
        mask_padded[mask_padded != 0.0] = 1.0
        visualize(image_padded, mask_padded, original_image=image, original_mask=mask)
        #np.savez('Dataset/Taining/{}'.format(iteration), augmented['image'], augmented['mask'])
    aug = A.Compose([
        A.Resize(height=256, width=256),
        A.CropNonEmptyMaskIfExists(random.randint(30, 150), random.randint(30, 150), p=0.5),
        A.OneOf([
            A.CoarseDropout(max_holes=random.randint(1, 15), max_height=random.randint(1, 8),
                            max_width=random.randint(1, 8), p=0.5),
            A.MaskDropout(p=0.5),
            #A.GridDropout(p=0.5),
            A.RandomShadow(p=0.5)], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Perspective(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.Resize(height=256, width=256),
        A.CLAHE(p=0.5),
        A.FancyPCA(p=0.5),
        A.OneOf([
            #A.ColorJitter(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.HueSaturationValue(p=0.5),
            #A.RandomGamma(p=0.5),
            #A.Solarize(p=0.5)
            ], p=0.5),
        A.OneOf([
            A.ISONoise(),
            A.GaussNoise(),
            A.MultiplicativeNoise()
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(p=0.5),
            A.GlassBlur(p=0.5),
            A.GaussianBlur(p=0.5),
            A.Blur(p=0.5)], p=0.5),
        A.Sharpen(p=0.5),
        A.OneOf([
            A.RandomFog(p=0.5),
            A.RandomRain(p=0.5),
        ], p=0.5)
    ], p=1)
    for i in range(n):
        for iteration, img in enumerate(tqdm(images)):
            image = np.asarray(Image.open('FsCocoDataset/images/{}'.format(img)))  # 00016 00810
            mask = np.asarray(Image.open('FsCocoDataset/segmask/{}'.format(img)))
            augmented = aug(image=image, mask=mask)
            image_padded = augmented['image']
            mask_padded = augmented['mask']
            mask_padded[mask_padded != 0.0] = 1.0
            #np.savez('Dataset/Taining/aug_{}'.format(iteration), augmented['image'], augmented['mask'])
            #print(image_padded.shape, mask_padded.shape)
            visualize(image_padded, mask_padded, original_image=image, original_mask=mask)


if __name__ == "__main__":
    # splitTrainingTest()
    augmentation(1)
    pass
