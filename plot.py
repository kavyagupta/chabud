import os
import json
import tqdm 
import argparse
import operator

import cv2
import numpy as np
import torch
from torchmetrics.functional import dice
from torchmetrics import JaccardIndex

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
    jaccard_index = JaccardIndex(task="binary").to(device)

    idx = 0
    for pre, post, mask in tqdm.tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        outputs = net(pre, post)
     
        outputs = torch.argmax(outputs, axis=1)

        for i in range(pre.shape[0]):
            iou = jaccard_index(outputs[i], mask[i])
            results.append([val_loader.dataset.data_list[idx], 
                           outputs[i].data.cpu().numpy().astype(np.uint8),
                           iou.item()])
            idx += 1

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting images")

    parser.add_argument(
        "--experiment-url", type=str, required=True, help="url of the model")
    parser.add_argument("--normalize", action='store_true', help="normalize")
    parser.add_argument("--bands", default="0,1,2,3,4,5,6,7,8,9,10,11", 
                        help="bands to use")
    parser.add_argument("--swap", action='store_true', help="swap pre and post images")

    args = parser.parse_args()

    device = torch.device("cuda:0")
    dst_path, _ = weight_and_experiment(args.experiment_url, best=True)
    fin = open('/'.join(dst_path.split('/')[:-1]) + '/epxeriment_config.json', 'r')
    metadata = json.load(fin)
    args.__dict__.update(metadata)
    fin.close()

    net = get_model(args)
    net.to(device)
    weight = torch.load(dst_path)
    net.load_state_dict(weight)
    net.eval()

    _, val_loader = get_dataloader(args)

    results = val(val_loader=val_loader, net=net, device=device)
    
    results = sorted(results, key=operator.itemgetter(2))
    worst5 = results[:5]
    best5 = results[-5:]

    print(worst5)
    print(best5)
    



    
        