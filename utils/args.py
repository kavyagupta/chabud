import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--config-path", type=str, required=True, help="config path",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--vector-dir",
        required=True,
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="GeoEngine experiment URL to resume an experiment."
    )

    parser.add_argument(
        "--finetune-from",
        default=None, 
        help="Engine epxeriment URL for pretrained weights"
    )


    # Model
    parser.add_argument("--arch", required=True, type=str, help="Model achitecture (bidate_unet, simaunet_diff)")
    parser.add_argument("--loss", requried=True, type=str, help="cross entropy/focal")
    parser.add_argument("--alpha", defaut="0.99,0.01", type=str, help="For focal loss" )
    parser.add_argument("--gamma", defaut=2, type=float, help="For focal loss" )
    parser.add_argument("--optim", default="sgd", help="optimizer sgd/adam")
    
    # Data
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="N",
    )
    
    parser.add_argument(
        "--window", type=int, default=512, help="Image size: dim x dim x 3"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
   
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    
    return parser.parse_args()
