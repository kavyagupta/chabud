import os
import glob
import pandas as pd
import json
from torchvision.io import read_image
import torch.utils.data as data
import rasterio as rio
import base64
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io
import cv2
from skimage import transform


class ChabudDataset(data.Dataset):
    def __init__(self, json_dir, data_list=[], transform=True, target_transform=None):
        # self.json_paths = [x for x in glob.glob(json_dir + '/*') if 'metadata' not in x]
        self.json_paths = json_dir
        self.train_list = data_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        # json_path = self.json_paths[idx]
        json_path = os.path.join(self.json_paths, self.train_list[idx])
        f = open(json_path)
        data = json.load(f)
        img_pre = rio.open(data["images"][0]["file_name"]).read()
        img_post = rio.open(data["images"][1]["file_name"]).read()
        mask_string = data["properties"][0]["labels"][0]
        img_mask = np.array(Image.open(io.BytesIO(base64.b64decode(mask_string))))
        if self.transform:
            img_pre = self.transform(img_pre)
            img_post = self.transform(img_post)
        if self.target_transform:
            img_mask = self.target_transform(img_mask)
        return img_pre.astype("float32"), img_post.astype("float32"), img_mask


class Rescale_train(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        channels = image.shape[0]
        img = transform.resize(image, (channels, self.output_size, self.output_size))

        return img


class Rescale_target(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        img = transform.resize(image, (self.output_size, self.output_size))

        return img


f = open("../CHABUD/vectors/Original_Split-20230524T135331/MASK/metadata.json")
data = json.load(f)
train_list = data["dataset"]["train"]
val_list = data["dataset"]["val"]

# chabud = ChabudDataset(json_dir='../CHABUD/vectors/Original_Split-20230524T135331/MASK',list = train_list,transform = Rescale_train(256),target_transform = Rescale_target(256))

chabud_train = ChabudDataset(
    json_dir="../CHABUD/vectors/Original_Split-20230524T135331/MASK",
    data_list=train_list,
    transform=Rescale_train(256),
    target_transform=Rescale_target(256),
)
chabud_val = ChabudDataset(
    json_dir="../CHABUD/vectors/Original_Split-20230524T135331/MASK",
    data_list=val_list,
    transform=Rescale_train(256),
    target_transform=Rescale_target(256),
)

train_dataloader = DataLoader(chabud_train, batch_size=16, shuffle=True)
# print(next(iter(train_dataloader))[0].shape)
# examples = enumerate(train_dataloader)
# batch_idx, (images_pre, images_post,images_mask) = next(examples)
# print(images_pre.shape)
# print(images_post.shape)
# print(images_mask[5,:,:])
