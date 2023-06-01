import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default=None, help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./trained_models",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )


    # Model
    parser.add_argument("--arch", type=str, help="Model achitecture")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of output classes in the model",
    )
    parser.add_argument(
        "--layer-type", type=str, choices=("dense", "unstructured", "channel", "filter"), help="dense | unstructured | channel | filter"
    )

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
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="path to datasets"
    )

    parser.add_argument(
        "--image-dim", type=int, default=32, help="Image size: dim x dim x 3"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
   
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    

    # Evaluate
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model"
    )

    parser.add_argument(
        "--val-method",
        type=str,
        default="base",
        choices=["base"],
        help="base: evaluation on unmodified inputs",
    )

    # Restart
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to latest checkpoint (default:None)",
    )

    # Additional
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Number of batches to wait before printing training logs",
    )


    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use PFTT to accelerate",
    )

    return parser.parse_args()
