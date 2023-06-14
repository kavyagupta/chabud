import os
import io
import json
import base64
import random

import cv2
import numpy as np
from PIL import Image
import rasterio as rio

import torch.utils.data as data
from torch.utils.data import DataLoader

import albumentations as A


def _stretch_8bit(band, lower_percent=0, higher_percent=98):
    """Serves `stretch_bands` by clipping extreme numbers and stretching 1 band.
    Parameters
    ----------
    band : np.ndarray
        A single band (h,w)
    lower_percent : integer
        Lower percentile to clip from image
    higher_percent : type
        Higher percentile to clip from image
    Returns
    -------
    np.ndarray
        (H, W) stretched band with same dimensions as input band
    """
    a = 0
    b = 255
    real_values = band.flatten()
    # real_values = real_values[real_values > 0]

    c = np.percentile(real_values, lower_percent)
    d = np.percentile(real_values, higher_percent)
    if (d - c) == 0:
        d += 1
    t = a + (band - c) * ((b - a) / (d - c))
    t[t < a] = a
    t[t > b] = b
    return t.astype(np.uint8)


class ChabudDataset(data.Dataset):
    def __init__(self, data_root, json_dir, data_list, bands, 
                 bit8=False, swap=False, transform = None):
        self.data_root = data_root
        self.json_dir = json_dir
        self.data_list = data_list
        self.bands = bands
        self.bit8 = bit8
        self.swap = swap
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

        pre = []
        post = []
        for band_idx in self.bands:
            img = cv2.resize(img_pre[band_idx], (512, 512), interpolation=cv2.INTER_CUBIC)
            if self.bit8:
                img = _stretch_8bit(img) / 255.
            pre.append(img)
            im = cv2.resize(img_post[band_idx], (512, 512), interpolation=cv2.INTER_CUBIC)
            if self.bit8:
                img = _stretch_8bit(img) / 255.
            post.append(img)

        img_pre = np.asarray(pre)
        img_post = np.asarray(post)

        if self.swap and random.random() > 0.5:
            # swap pre post as a form of augmentation
            img_pre, img_post = img_post, img_pre
            img_mask = np.zeros(img_mask.shape, dtype=np.uint8)
        
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
    args.bands = list(map(int, args.bands.split(',')))

    f = open(f"{args.data_root}/{args.vector_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]


    if args.bands == [1, 2, 3]:
        mean_bands = [0.406, 0.456, 0.485]
        std_bands = [0.225, 0.224, 0.229]
        bit8 = True
    else:
        mean=[1353.72692573, 1117.20229235, 1041.88472484,  946.55425487,
            1199.1886645 , 2003.00679994, 2374.00844442, 2301.22043839,
            732.18195008,   12.09952762, 1118.20272293, 2599.78293726]
        std=[ 72.41170098, 146.47166895, 158.20546468, 217.42332058,
                168.33411967, 230.56343772, 296.15066586, 307.65398036,
                85.71403735,   0.8560447, 221.18654082, 329.1786173 ]

        mean_bands = []
        std_bands = []
        for i in args.bands:
            mean_bands.append(mean[i])
            std_bands.append(std[i])
        
        bit8 = False

    pipeline = []
    if args.normalize:
        pipeline.append(A.Normalize(mean=mean_bands, std=std_bands))

    transform_train = A.Compose([A.OneOf([
                                    A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                                    A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
                                ], p=1),
                                A.HorizontalFlip(p=0.5), 
                                A.VerticalFlip(p=0.5),
                                # A.RandomBrightnessContrast(p=0.2), 
                                A.OneOf([
                                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    A.GridDistortion(p=0.5),
                                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                                    ], p=0.8),
                                A.Resize(args.window, args.window)
                              ] + pipeline,
                              additional_targets={'post': 'image'})
    
    transform_val = A.Compose([A.Resize(args.window, args.window)] + pipeline,
                              additional_targets={'post': 'image'})

    chabud_train = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=train_list,
        bands=args.bands,
        bit8=bit8,
        swap=args.swap,
        transform=transform_train
    )

    chabud_val = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=val_list,
        bands=args.bands,
        bit8=bit8,
        transform=transform_val
    )

    train_loader = DataLoader(chabud_train, batch_size=args.batch_size, 
                              num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=False)
    
    return train_loader, val_loader