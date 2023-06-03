import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--config-path", type=str, require=True, help="config path",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--vector-dir",
        default="vectors/Original_Split-20230524T135331/MASK",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )


    # Model
    parser.add_argument("--arch", required=True, type=str, help="Model achitecture")
    
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
