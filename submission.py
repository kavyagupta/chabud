import json 
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Union, Dict, Literal

from trimesh.voxel.runlength import dense_to_brle

import h5py
import pandas as pd

import numpy as np
from numpy.typing import NDArray
import torch

from models import get_model
from utils.engine_hub import weight_and_experiment
    

def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue

            if "pre_fire" not in values:
                continue
            
            result[uuid]['post'] = values['post_fire'][...].astype(np.float32).transpose(2, 0, 1)
            result[uuid]['pre'] = values['pre_fire'][...].astype(np.float32).transpose(2, 0, 1)

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

    validation_fold = retrieve_validation_fold(args.data_path)

    # use a list to accumulate results
    result = []
    # instantiate the model
    model = get_model(args)
    model.to(device)
    weight = torch.load(dst_path)
    model.load_state_dict(weight)
    _ = model.eval()

    for uuid in validation_fold:
        input_images = validation_fold[uuid]

        # perform the prediction
        pre = torch.from_numpy(input_images['pre']).to(device).float().unsqueeze(0)
        post = torch.from_numpy(input_images['post']).to(device).float().unsqueeze(0)
        predicted = model(pre, post)
        predicted = torch.argmax(predicted, axis=1)
        predicted = predicted.data.cpu().numpy()

        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv(f'predictions/{args.csv_name}.csv', index=False)