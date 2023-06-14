import torch 

from .bidate_model import BiDateNet
from .siamnet_diff import SiamUnet_diff
from .bidate_concat import BiDateConcatNet
from .bidate_deeplab import (bidate_deeplab_resnet50,
                             bidate_deeplab_resnet101,
                             bidate_deeplab_mobilenet_v3_large)
from .a2net import BaseNet
from utils.engine_hub import weight_and_experiment

__all__ = ["get_model"]


def get_model(args, n_channels=12, n_classes=2):
    ############# model #####################
    if args.arch == "bidate_unet":
        net = BiDateNet(n_channels=n_channels, 
                        n_classes=n_classes)
    elif args.arch == "siamunet_diff":
        net = SiamUnet_diff(n_channels=n_channels, 
                            n_classes=n_classes)
    elif args.arch == "bidate_concat":
        net = BiDateConcatNet(n_channels=n_channels, 
                              n_classes=n_classes)
    elif args.arch == "bidate_deeplab_resnet50":
        net = bidate_deeplab_resnet50(n_channels=n_channels, 
                                      n_classes=n_classes)
    elif args.arch == "bidate_deeplab_resnet101":
        net = bidate_deeplab_resnet101(n_channels=n_channels, 
                                       n_classes=n_classes)
    elif args.arch == "bidate_deeplab_mobilenet_v3_large":
        net = bidate_deeplab_mobilenet_v3_large(n_channels=n_channels, 
                                                n_classes=n_classes)
    elif args.arch == "a2net":
        net = BaseNet(n_channels=n_channels, n_classes=n_classes)
    else:
        print ("Proper architecture name not passed")
        return 

    return net