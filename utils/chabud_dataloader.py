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
    def __init__(self, data_root, json_dir, data_list, transform = None):
        self.data_root = data_root
        self.json_dir = json_dir
        self.data_list = data_list
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
            transformed = self.transform(image = img_pre.transpose(1, 2, 0), 
                                         post = img_post.transpose(1, 2, 0), 
                                         mask= img_mask)
            img_pre = transformed['image']
            img_pre = img_pre.transpose(2, 0, 1)
            img_post = transformed['post']
            img_post = img_post.transpose(2, 0, 1)
            img_mask = transformed['mask']

        return img_pre.astype(np.float32), img_post.astype(np.float32), img_mask


def get_dataloader(args):

    f = open(f"{args.data_root}/{args.vector_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]

    mean=[1353.72692573, 1117.20229235, 1041.88472484,  946.55425487,
        1199.1886645 , 2003.00679994, 2374.00844442, 2301.22043839,
        732.18195008,   12.09952762, 1118.20272293, 2599.78293726]
    std=[ 72.41170098, 146.47166895, 158.20546468, 217.42332058,
            168.33411967, 230.56343772, 296.15066586, 307.65398036,
            85.71403735,   0.8560447, 221.18654082, 329.1786173 ]

    pipeline = []
    if args.normalize:
        pipeline.append(A.Normalize(mean=mean, std=std))

    transform_train = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2), 
                                A.OneOf([
                                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    A.GridDistortion(p=0.5),
                                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                                    ], p=0.8),
                                A.MaskDropout((1,2), p=1),
                                A.Cutout(p=1),
                                A.Resize(args.window, args.window)
                              ] + pipeline,
                              additional_targets={'post': 'image'})
    
    transform_val = A.Compose([A.Resize(args.window, args.window)] + pipeline,
                              additional_targets={'post': 'image'})

    chabud_train = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=train_list,
        transform = transform_train
    )

    chabud_val = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=val_list,
        transform=transform_val
    )

    train_loader = DataLoader(chabud_train, batch_size=args.batch_size, 
                              num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=False)
    
    return train_loader, val_loader