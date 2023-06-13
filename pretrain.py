import os
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.nn.functional as F

from torchmetrics.functional import dice
from torchmetrics.functional.classification import multiclass_jaccard_index


from engine import Engine

from models import get_model
from utils.chabud_dataloader import get_dataloader
from utils.args import parse_args
from utils.loss import get_loss


def train_one_epoch(train_loader, net, criterion, 
                    optimizer, device):
    running_loss = 0.0

    for pre, post, _ in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post = pre.to(device), post.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(pre, post)
        mask = pre - post
        loss = criterion(outputs, mask)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def val(val_loader, net, criterion, device):
    # net.eval()

    running_loss = 0.0

    for pre, post, _ in tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post = pre.to(device), post.to(device) 
        outputs = net(pre, post)
        mask = pre - post
        loss = criterion(outputs, mask)
             
        running_loss += loss.item()
     
    return running_loss / len(val_loader)



def main():
    args = parse_args()

    fin = open(args.config_path)
    metadata = json.load(fin)
    fin.close()

    device = torch.device("cuda:0")
    ########Dataloaders #################
    train_loader, val_loader = get_dataloader(args)

    keep = 5
    track_ckpts = []
    ckpt_path = f"checkpoints/{args.arch}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        fout = open(os.path.join(ckpt_path, "epxeriment_config.json"), "w")
        json.dump(args.__dict__, fout)
        fout.close()

    net = get_model(args, n_classes=12)    
    net = net.to(device)
    criterion = get_loss(args, device)
    
    if args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
    elif args.optim == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)
        

    engine = Engine(**metadata)

    best_vloss = 10**9

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss = train_one_epoch(train_loader=train_loader, net=net, 
                                                criterion=criterion, optimizer=optimizer,
                                                device=device)

        print("Train loss {}".format(avg_loss))
        
        with torch.no_grad():
            avg_vloss = val(val_loader=val_loader, net=net, 
                                        criterion=criterion, device=device)
        if args.optim == "sgd":
            scheduler.step()
        elif args.optim == "adam":
            scheduler.step(avg_vloss)
        
        print("Val loss {}".format(avg_vloss))

        engine.log(step=epoch, train_loss=avg_loss, val_loss=avg_vloss)
        
        # Track best performance, and save the model's state
        if avg_vloss >= best_vloss:
            best_vloss = avg_vloss
            model_path = f"{ckpt_path}/epoch_{epoch}.pt" 
            torch.save(net.state_dict(), model_path)
            track_ckpts.append(model_path)

            if len(track_ckpts) > 5:
                remove_ckpt = track_ckpts.pop(0)
                os.remove(remove_ckpt)
                print (f"Checkpoint {remove_ckpt} removed")

            dst_path = engine.meta['experimentUrl']
            os.system(f"gsutil -m rsync -r -d {ckpt_path}/ {dst_path} 2> /dev/null")

            engine.log(step=epoch, best=True, checkpoint_path=model_path)
        
    engine.done()

if __name__ == "__main__":
    main()

