import torch 

from .bidate_model import BiDateNet
from .siamnet_diff import SiamUnet_diff
from .bidate_concat import BiDateConcatNet
from .bidate_deeplab import (bidate_deeplab_resnet50,
                             bidate_deeplab_resnet101,
                             bidate_deeplab_mobilenet_v3_large)
from utils.engine_hub import weight_and_experiment

__all__ = ["get_model"]


def get_model(args):
    ############# model #####################
    if args.arch == "bidate_unet":
        net = BiDateNet(n_channels=12, n_classes=2)
    elif args.arch == "siamunet_diff":
        net = SiamUnet_diff(n_channels=12, n_classes=2)
    elif args.arch == "bidate_concat":
        net = BiDateConcatNet(n_channels=12, n_classes=2)
    elif args.arch == "bidate_deeplab_resnet50":
        net = bidate_deeplab_resnet50(n_channels=12, n_classes=2)
    elif args.arch == "bidate_deeplab_resnet101":
        net = bidate_deeplab_resnet101(n_channels=12, n_classes=2)
    elif args.arch == "bidate_deeplab_mobilenet_v3_large":
        net = bidate_deeplab_mobilenet_v3_large(n_channels=12, n_classes=2)
    else:
        print ("Proper architecture name not passed")
        return 
    
    if args.finetune_from:
        dst_path, _ = weight_and_experiment(args.finetune_from)
        weight = torch.load(dst_path)
        net.load_state_dict(weight, strict=False)

    return net