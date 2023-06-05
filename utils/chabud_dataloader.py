import os
import io
import json
import base64

import cv2
import numpy as np
from PIL import Image
import rasterio as rio
from skimage import transform

import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader


class ChabudDataset(data.Dataset):
    def __init__(self, data_root, json_dir, data_list, window=512):
        self.data_root = data_root
        self.json_dir = json_dir
        self.data_list = data_list
        self.window = (window, window)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fin = open(os.path.join(self.data_root, self.json_dir, 
                                self.data_list[idx]))
        data = json.load(fin)
        fin.close()

        img_pre = rio.open(os.path.join(self.data_root,
                                        data["images"][0]["file_name"])).read()
        img_post = rio.open(os.path.join(self.data_root,
                                         data["images"][1]["file_name"])).read()
        mask_string = data["properties"][0]["labels"][0]
        img_mask = np.array(Image.open(io.BytesIO(base64.b64decode(mask_string))))
        
        img_pre_resize = []
        img_post_resize = []
        for i in range(img_pre.shape[0]):
            img_pre_resize.append(cv2.resize(img_pre[i], self.window))
            img_post_resize.append(cv2.resize(img_post[i], self.window))
        
        img_pre_resize = np.asarray(img_pre_resize, dtype=np.float32)
        img_post_resize = np.asarray(img_post_resize, dtype=np.float32)
        img_mask = cv2.resize(img_mask, self.window, cv2.INTER_NEAREST)

        return img_pre_resize, img_post_resize, img_mask
    

def batch_mean_and_sd(loader):
    
    cnt = 0
    n_channels = 12
    fst_moment = torch.empty(n_channels)
    snd_moment = torch.empty(n_channels)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, range(2,13)])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, range(2,13)])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean, std


f = open(f"{'../data'}/{'vectors/Original_Split-20230524T135331/MASK'}/metadata.json")
data = json.load(f)
train_list = data["dataset"]["train"]
val_list = data["dataset"]["val"]

chabud_val = ChabudDataset(
        data_root='./data',
        json_dir="vectors/Original_Split-20230524T135331/MASK",
        data_list=val_list,
        window=512
    )

val_loader = DataLoader(chabud_val, batch_size=4, 
                            shuffle=False)
  
mean, std = batch_mean_and_sd(val_loader)
print("mean and std: \n", mean, std)