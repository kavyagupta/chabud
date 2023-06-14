import argparse

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

