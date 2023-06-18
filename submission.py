import os 
import json 
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Union, Dict, Literal

from trimesh.voxel.runlength import dense_to_brle

import h5py
import pandas as pd
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import albumentations as A

from models import get_model
from utils.engine_hub import weight_and_experiment
    

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

def retrieve_validation_fold(args) -> Dict[str, NDArray]:
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

    result = defaultdict(dict)
    with h5py.File(args.path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue

            if "pre_fire" not in values:
                continue
            
            img_pre = values['post_fire'][...]
            img_post = values['pre_fire'][...]

            pre = []
            post = []
            for band_idx in args.bands:
                band_pre = img_pre[band_idx]
                if bit8:
                    band_pre = _stretch_8bit(band_pre) / 255.
                pre.append(band_pre)

                band_post = img_post[band_idx]
                if bit8:
                    band_post = _stretch_8bit(band_post) / 255.
                post.append(band_post)

            img_pre = np.asarray(pre)
            img_post = np.asarray(post)

            if args.normalize:
                transform = A.Compose([A.Normalize(mean=mean_bands, std=std_bands)])
                transformed = transform(image = img_pre.transpose(1, 2, 0), 
                                         post = img_post.transpose(1, 2, 0))
                img_pre = transformed['image']
                img_pre = img_pre.transpose(2, 0, 1)
                img_post = transformed['post']
                img_post = img_post.transpose(2, 0, 1)

            result[uuid]['post'] = img_pre.astype(np.float32)
            result[uuid]['pre'] = img_post.astype(np.float32)

    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.astype(bool).flatten())
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model Testing")

    parser.add_argument(
        "--experiment-url", type=str, required=True, help="url of the model")

    parser.add_argument(
        "--data-path", type=str, required=True, help="data-path")

    parser.add_argument(
        "--csv-name", type=str, required=True, help="prediction csv name")


    args = parser.parse_args()

    device = torch.device("cuda:0")
    dst_path, _ = weight_and_experiment(args.experiment_url, best=True)
    fin = open('/'.join(dst_path.split('/')[:-1]) + '/epxeriment_config.json', 'r')
    metadata = json.load(fin)
    args.__dict__.update(metadata)
    fin.close()

    validation_fold = retrieve_validation_fold(args)

    # use a list to accumulate results
    result = []
    # instantiate the model
    model = get_model(args)
    model.to(device)
    weight = torch.load(dst_path)
    model.load_state_dict(weight)
    _ = model.eval()

    out_path = f"predictions/{args.arch}-{args.experiment_url.split('/')[-1]}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for uuid in validation_fold:
        input_images = validation_fold[uuid]

        # perform the prediction
        pre = torch.from_numpy(input_images['pre']).to(device).float().unsqueeze(0)
        post = torch.from_numpy(input_images['post']).to(device).float().unsqueeze(0)
        predicted = model(pre, post)
        predicted = torch.argmax(predicted, axis=1)
        predicted = predicted.data.cpu().numpy()

        cv2.imwrite(f"{out_path}/{uuid}.png", predicted.astype(np.uint8))

        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv(f"{out_path}.csv", 
                         index=False)