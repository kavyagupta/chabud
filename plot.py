import os
import io
import json
import tqdm 
import argparse
import operator
import base64

import cv2
from PIL import Image
import numpy as np
import rasterio as rio
import torch
from torchmetrics.functional import dice
from torchmetrics import JaccardIndex

from models import get_model

from utils.chabud_dataloader import get_dataloader, _stretch_8bit
from utils.loss import get_loss
from utils.engine_hub import weight_and_experiment


def get_8bit(sample_pre, sample_post):
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
    
    return pre_bgr.tranpose(1, 2, 0), post_bgr.tranpose(1, 2, 0)


def val(val_loader, net, device):
    # net.eval()
    results = []
    jaccard_index = JaccardIndex(task="binary").to(device)

    idx = 0
    for pre, post, mask in tqdm.tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        outputs = net(pre, post)

        print (outputs.min(), outputs.max(), outputs.shape)
        outputs = torch.argmax(outputs, axis=1)
        print (outputs.min(), outputs.max(), outputs.shape)

        for i in range(pre.shape[0]):
            iou = jaccard_index(outputs[i], mask[i])
            print (iou.item())
            results.append([val_loader.dataset.data_list[idx], 
                           outputs[i].data.cpu().numpy().astype(np.uint8),
                           iou.item()])
            idx += 1

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plotting images")

    parser.add_argument(
        "--experiment-url", type=str, required=True, help="url of the model")
    parser.add_argument("--plot-dir", type=str, required=True)

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

    for best in best5:
        fin = open(os.path.join(args.data_root, args.vector_dir, 
                                best[0]))
        data = json.load(fin)
        fin.close()

        img_pre = rio.open(os.path.join(args.data_root,
                                        data["images"][0]["file_name"])).read()
        img_post = rio.open(os.path.join(args.data_root,
                                         data["images"][1]["file_name"])).read()
        mask_string = data["properties"][0]["labels"][0]
        img_mask = np.array(Image.open(io.BytesIO(base64.b64decode(mask_string))))

        img_mask = np.stack([img_mask, img_mask, img_mask]).transpose(1, 2, 0)
        pred_mask = np.stack([best[1], best[1], best[1]]).transpose(1, 2, 0)

        img_pre, img_post = get_8bit(img_pre, img_post)

        print (img_mask.shape, pred_mask.shape, img_pre.shape, img_post.shape)
    



    
        