import os
import tqdm 
import argparse
import operator

import cv2
import numpy as np
import torch
from torchmetrics.functional import dice
from torchmetrics.functional.classification import multiclass_jaccard_index

from models import get_model

from utils.chabud_dataloader import get_dataloader, _stretch_8bit
from utils.loss import get_loss
from utils.engine_hub import weight_and_experiment


def save_img(sample_post, sample_pre):
    post_r = sample_post[1, :, :]
    post_g = sample_post[2, :, :]
    post_b = sample_post[3, :, :]

    pre_r = sample_pre[1, : ,:]
    pre_g = sample_pre[2, :, :]
    pre_b = sample_pre[3, :, :]

    post_r = _stretch_8bit(post_r)
    post_g = _stretch_8bit(post_g)
    post_b = _stretch_8bit(post_b)

    pre_r = _stretch_8bit(pre_r)
    pre_g = _stretch_8bit(pre_g)
    pre_b = _stretch_8bit(pre_b)

    post_bgr = np.asarray([post_b, post_g, post_r])
    pre_bgr = np.asarray([pre_b, pre_g, pre_r])
    
    cv2.imwrite('pre_img.png', pre_bgr)
    cv2.imwrite('post_img.png', post_bgr)

    return


def val(val_loader, net, device):
    # net.eval()
    results = []

    idx = 0
    for pre, post, mask in tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        outputs = net(pre, post)
     
        outputs = torch.argmax(outputs, axis=1)
        ious = multiclass_jaccard_index(outputs, mask, num_classes=2, average=None)
        outputs = outputs.data.cpu().numpy()
        ious = ious.data.cpu().numpy()

        for i in range(pre.shape[0]):
            results.append(val_loader.data_list[idx], 
                           outputs[i].astype(np.uint8),
                           ious[i])
            idx += 1

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting images")

    parser.add_argument(
        "--experiment-url", type=str, required=True, help="url of the model")

    # parser.add_argument(
    #     "--data-path", type=str, required=True, help="folder to save the images")

    parser.add_argument(
        "--arch", type=str, required=True, help="model arch")
    
    parser.add_argument(
        "--config-path", type=str, required=True, help="config path",
    )
    parser.add_argument(
        "--data-root", default="./data", type=str, help="directory to save results",
    )
    parser.add_argument(
        "--vector-dir", required=True, type=str, help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0")
    dst_path, _ = weight_and_experiment(args.experiment_url)
    net = get_model(args)
    net.to(device)
    weight = torch.load(dst_path)
    net.load_state_dict(weight)
    net.eval()

    _, val_loader = get_dataloader(args)
    criterion = get_loss(args, device)

    results = val(val_loader=val_loader, net=net, device=device)
    
    results = sorted(results, key=operator.itemgetter(2))
    worst5 = results[:5]
    best5 = results[:5]

    print(worst5)
    print(best5)
    



    
        