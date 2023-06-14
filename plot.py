import argparse

import torch
import os

from models import get_model

try:
    from engine.libs.hub import get_last_weight, get_best_weight
except ImportError:
    raise ImportError(
        'Please run "pip install granular-engine" to install engine')

def weight_and_experiment(url, best=False):
    if best:
        checkpoint, experiment_id = get_best_weight(url)
    else:
        checkpoint, experiment_id = get_last_weight(url)

    dst_path = 'pretrain/' + '/'.join(checkpoint.replace('gs://', '').replace('s3://', '').split('/')[2:])
    os.system(f"gsutil -m cp -n -r {checkpoint} {dst_path} 2> /dev/null")

    weight = torch.load(dst_path)
    if 'state_dict' in weight:
        weight = weight['state_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in weight.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
        
            
    return net.load_state_dict(model_dict)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting images")

    parser.add_argument(
        "--weight-file", type=str, required=True, help="weight file")

    parser.add_argument(
        "--data-path", type=str, required=True, help="folder to save the images")

    parser.add_argument(
        "--arch", type=str, required=True, help="model arch")

    # parser.add_argument(
    #     "--csv-name", type=str, required=True, help="prediction csv name")


    args = parser.parse_args()

    device = torch.device("cuda:0")

    model = get_model(args)
    model.to(device)