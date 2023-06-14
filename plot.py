import argparse
import numpy as np

import torch
import os

from utils.chabud_dataloader import get_dataloader
from utils.loss import get_loss

from models import get_model


try:
    from engine.libs.hub import get_last_weight, get_best_weight
except ImportError:
    raise ImportError(
        'Please run "pip install granular-engine" to install engine')


def _stretch_8bit(band, lower_percent=0, higher_percent=98):
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

    return running_loss, running_score, running_iou


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

    # parser.add_argument(
    #     "--csv-name", type=str, required=True, help="prediction csv name")


    args = parser.parse_args()

    device = torch.device("cuda:0")
    net = weight_and_experiment(args)
    net.to(device)
    net.eval()

    _, val_loader = get_dataloader(args)

    criterion = get_loss(args, device)

    
    vloss, vscore, viou = val(val_loader=val_loader, net=net, 
                                        criterion=criterion, device=device)
    
    sorted, indices = torch.sort(viou)

    for 

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

        post_rgb = np.asarray([post_r, post_g, post_b])
        pre_rgb = np.asarray([pre_r, pre_g, pre_b])