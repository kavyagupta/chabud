import argparse
import numpy as np
import cv2

import torch
import os

from utils.chabud_dataloader import get_dataloader, _stretch_8bit
from utils.loss import get_loss

from models import get_model
from torchmetrics.functional import dice
from torchmetrics.functional.classification import multiclass_jaccard_index


try:
    from engine.libs.hub import get_last_weight, get_best_weight
except ImportError:
    raise ImportError(
        'Please run "pip install granular-engine" to install engine')

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


def weight_and_experiment(args):
    net = get_model(args)

    if best:
        checkpoint, experiment_id = get_best_weight(args.cpt_url)
    else:
        checkpoint, experiment_id = get_last_weight(args.cpt_url)

    dst_path = 'pretrain/' + '/'.join(checkpoint.replace('gs://', '').replace('s3://', '').split('/')[2:])
    os.system(f"gsutil -m cp -n -r {checkpoint} {dst_path} 2> /dev/null")

    weight = torch.load(dst_path)
    if 'state_dict' in weight:
        weight = weight['state_dict']
    model_dict = net.state_dict()
               
    return net.load_state_dict(model_dict)


def val(val_loader, net, criterion, device):
    # net.eval()

    running_loss = 0.0
    running_score = 0.0
    running_iou = 0.0

    for pre, post, mask in tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        outputs = net(pre, post)
        loss = criterion(outputs, mask.long())
     
        outputs = torch.argmax(outputs, axis=1)
        score = dice(outputs, mask)
        iou = multiclass_jaccard_index(outputs, mask, num_classes=2)
        
        running_loss += loss.item()
        running_score += score.item()
        running_iou += iou.item()

    return outputs, running_loss, running_score, running_iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting images")

    parser.add_argument(
        "--cpt-url", type=str, required=True, help="url of the model")

    parser.add_argument(
        "--data-path", type=str, required=True, help="folder to save the images")

    parser.add_argument(
        "--arch", type=str, required=True, help="model arch")
    
    parser.add_argument(
        "--loss", required=True, type=str, help="cross entropy/focal")

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
    net = weight_and_experiment(args)
    net.to(device)
    net.eval()

    _, val_loader = get_dataloader(args)
    criterion = get_loss(args, device)

    outputs, vloss, vscore, viou = val(val_loader=val_loader, net=net, 
                                        criterion=criterion, device=device)
    
    sorted, indices = torch.sort(viou)
        