import numpy as np
import pandas as pd
import h5py

from trimesh.voxel.runlength import dense_to_brle
from pathlib import Path
from collections import defaultdict

import argparse

import torch

from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray

from models import get_model
    

def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue
            
            result[uuid]['post'] = values['post_fire'][...]
            result[uuid]['pre'] = values['pre_fire'][...]

    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.astype(bool).flatten())
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model Testing")

    parser.add_argument(
        "--weight-file", type=str, required=True, help="weight file")

    parser.add_argument(
        "--data-path", type=str, required=True, help="data-path")

    parser.add_argument(
        "--model-name", type=str, required=True, help="model arch")

    parser.add_argument(
        "--save-path", type=str, required=True, help="saved csv path")


    args = parser.parse_args()

    device = torch.device("cuda:0")

    validation_fold = retrieve_validation_fold(args.eval_path)

    # use a list to accumulate results
    result = []
    # instantiate the model
    model = get_model(args)
    weight = torch.load(args.weight_file_path)
    model.load_dict(weight)
    _ = model.eval()

    for uuid in validation_fold:
        input_images = validation_fold[uuid]

        # perform the prediction
        pre = torch.from_numpy(input_images['pre']).to(device).float()
        post = torch.from_numpy(input_images['post']).to(device).float()
        predicted = model(pre, post)
        predicted = torch.argmax(predicted, axis=1)
        predicted = predicted.data.cpu().numpy()

        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv(f'"predictions/{args.prediction_name}.csv', index=False)