import os
import io
import json
import base64

import cv2
import numpy as np
from PIL import Image
import rasterio as rio

import torch.utils.data as data
from torch.utils.data import DataLoader

import albumentations as A


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
            img_pre = transformed['image']
            img_pre = img_pre.transpose(2,1,0)
            img_post = transformed['image1']
            img_post = img_post.transpose(2,1,0)
            img_mask = transformed['mask']
        
        # img_pre_resize = []
        # img_post_resize = []
        # for i in range(img_pre.shape[0]):
        #     img_pre_resize.append(cv2.resize(img_pre[i], self.window))
        #     img_post_resize.append(cv2.resize(img_post[i], self.window))
        
        # img_pre_resize = np.asarray(img_pre_resize, dtype=np.float32)
        # img_post_resize = np.asarray(img_post_resize, dtype=np.float32)
        # img_mask = cv2.resize(img_mask, self.window, cv2.INTER_NEAREST)

        return img_pre, img_post, img_mask


def get_dataloader(args):

    f = open(f"{args.data_root}/{args.vector_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]


    transform_train = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2), 
                                A.OneOf([
                                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    A.GridDistortion(p=0.5),
                                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                                    ], p=0.8),
                                A.Resize(512, 512)
                              ])
    
    transform_val = A.Compose([A.Resize(512, 512)])

    chabud_train = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=train_list,
        window=args.window, transform = transform_train
    )

    chabud_val = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=val_list,
        window=args.window, transform=transform_val
    )

    train_loader = DataLoader(chabud_train, batch_size=args.batch_size, 
                              shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=args.batch_size, 
                            shuffle=False)
    
    return train_loader, val_loader