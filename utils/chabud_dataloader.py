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


class ChabudDataset(data.Dataset):
    def __init__(self, data_root, json_dir, data_list, window=512,transform = None):
        self.data_root = data_root
        self.json_dir = json_dir
        self.data_list = data_list
        self.window = (window, window)
        self.transform = transform

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

        
        if self.transform:
            transformed = self.transform(image = img_pre.transpose(2, 1, 0), image1 = img_post.transpose(2, 1, 0), 
                                         mask= img_mask)
            img_pre = transformed['image'].transpose(0,1,2)
            img_post = transformed['imag1'].transpose(0,1,2)
            img_mask = transformed['mask']
        
        print(img_pre.shape)

        img_pre_resize = []
        img_post_resize = []
        for i in range(img_pre.shape[0]):
            img_pre_resize.append(cv2.resize(img_pre[i], self.window))
            img_post_resize.append(cv2.resize(img_post[i], self.window))
        
        img_pre_resize = np.asarray(img_pre_resize, dtype=np.float32)
        img_post_resize = np.asarray(img_post_resize, dtype=np.float32)
        img_mask = cv2.resize(img_mask, self.window, cv2.INTER_NEAREST)

        return img_pre_resize, img_post_resize, img_mask
