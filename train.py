import os
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import multiclass_jaccard_index

from engine import Engine

from models.bidate_model import BiDateNet
from utils.chabud_dataloader import ChabudDataset
from utils.args import parse_args


def train_one_epoch(train_loader, net, criterion, 
                    optimizer, device):
    running_loss = 0.0
    running_score = 0.0
    running_iou = 0.0

    for pre, post, mask in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(pre, post)
        loss = criterion(outputs, mask.long())

        outputs = torch.argmax(outputs, axis=1)
        score = dice(outputs, mask)
        iou = multiclass_jaccard_index(outputs, mask, num_classes=2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_score += score.item()
        running_iou += iou.item()

    return running_loss / len(train_loader), running_score / len(train_loader), running_iou / len(train_loader)

def val(val_loader, net, criterion, device):
    net.eval()

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

    return running_loss / len(val_loader), running_score / len(val_loader), running_iou / len(train_loader)



def main():
    args = parse_args()

    fin = open(args.config_path)
    metadata = json.load(fin)
    fin.close()
    engine = Engine(**metadata)

    device = torch.device("cuda:0")
    ########Dataloaders #################
    
    f = open(f"{args.data_root}/{args.vector_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]

    chabud_train = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=train_list,
        window=args.window
    )

    chabud_val = ChabudDataset(
        data_root=args.data_root,
        json_dir=args.vector_dir,
        data_list=val_list,
        window=args.window
    )

    train_loader = DataLoader(chabud_train, batch_size=args.batch_size, 
                              shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=args.batch_size, 
                            shuffle=False)
    
    keep = 5
    track_ckpts = []
    ckpt_path = f"checkpoints/{args.arch}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        fout = open(os.path.join(ckpt_path, "epxeriment_config.json"), "w")
        json.dump(args.__dict__, fout)
        fout.close()


    ############# model #####################
    if args.arch == "bidate_unet":
        net = BiDateNet(n_channels=12, n_classes=2)
    else:
        print ("Proper architecture name not passed")
        return 
    
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    best_vscore = -1

    for epoch in range(args.epochs):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss, avg_score, avg_iou = train_one_epoch(train_loader=train_loader, net=net, 
                                                criterion=criterion, optimizer=optimizer,
                                                device=device)

        print("Train loss {} dice {} iou {}".format(avg_loss, avg_score, avg_iou))
        
        avg_vloss, avg_vscore, avg_viou = val(val_loader=val_loader, net=net, 
                                     criterion=criterion, device=device)
        
        print("Val loss {} dice {} iou {}".format(avg_vloss, avg_vscore, avg_viou))

        engine.log(step=epoch, train_loss=avg_loss, train_score=avg_score, train_iou=avg_iou
                   val_loss=avg_vloss, val_score=avg_vscore, val_iou=avg_viou)

        # Track best performance, and save the model's state
        if avg_vscore > best_vscore:
            best_vscore = avg_vscore
            model_path = f"{ckpt_path}/epoch_{epoch}.pt" 
            torch.save(net.state_dict(), model_path)
            track_ckpts.append(model_path)

            if len(track_ckpts) > 5:
                remove_ckpt = track_ckpts.pop(0)
                os.remove(remove_ckpt)
                print ("Checkpoint {remove_ckpt} removed")

            dst_path = engine.meta['experimentUrl']
            os.system(f"gsutil -m rsync -r -d {ckpt_path}/ {dst_path} 2> /dev/null")

            engine.log(step=epoch, best=True, checkpoint_path=model_path)
        
        engine.done()

if __name__ == "__main__":
    main()

