from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from coco_assistant import COCO_Assistant


def deleteborderfromimage():
    images = os.listdir("FRT_MERGED/images")
    for image in tqdm(images):
        img = Image.open("FRT_MERGED/images/" + image)
        w, h = img.size
        im1 = img.crop((140, 140, w - 140, h - 140))
        im1.save('FRT_MERGED/images2/{}'.format(image))


def deleteborderfromsegmentation():
    images = os.listdir("FRT_MERGED/segmasks")
    for image in tqdm(images):
        img = Image.open("FRT_MERGED/segmasks/" + image)
        w, h = img.size
        im1 = img.crop((140, 140, w - 140, h - 140))
        im1.save('FRT_MERGED/segmask2/{}'.format(image))


def convertFileExtension():
    images = os.listdir("FsCocoDataset/images")
    for image in tqdm(images):
        if image.endswith(".jpg"):
            name = image[:-4] + '.png'
            img = Image.open("FsCocoDataset/images/" + image)
            img.save('FsCocoDataset/images/{}'.format(name))
            os.remove('FsCocoDataset/images/' + image)


if __name__ == "__main__":
    #deleteborderfromsegmentation()
    #deleteborderfromimage()
    convertFileExtension()
