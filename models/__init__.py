from .bidate_model import BiDateNet
from .siamnet_diff import SiamUnet_diff
from .bidate_concat import BiDateConcatNet

__all__ = ["BiDateNet", "SiamUnet_diff", "BiDateConcatNet", "get_model"]


def get_model(args):
    ############# model #####################
    if args.arch == "bidate_unet":
        net = BiDateNet(n_channels=12, n_classes=2)
    elif args.arch == "siamunet_diff":
        net = SiamUnet_diff(n_channels=12, n_classes=2)
    elif args.arch == "bidate_concat":
        net = BiDateConcatNet(n_channels=12, n_classes=2)
    else:
        print ("Proper architecture name not passed")
        return 