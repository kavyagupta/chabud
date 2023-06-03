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

from engine import Engine

from models.bidate_model import BiDateNet
from utils.chabud_dataloader import ChabudDataset
from utils.args import parse_args


def train_one_epoch(train_loader, net, criterion, 
                    optimizer, device):
    running_loss = 0.0
    running_score = 0.0

    for pre, post, mask in tqdm(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(pre, post)
        loss = criterion(outputs, mask.long())

        outputs = torch.argmax(outputs, axis=1)
        score = dice(outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_score += score.item()

    return running_loss / len(train_loader), running_score / len(train_loader)

def val(val_loader, net, criterion, device):
    net.eval()

    running_loss = 0.0
    running_score = 0.0

    for pre, post, mask in tqdm(val_loader):
        # get the inputs; data is a list of [inputs, labels]
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)

        outputs = net(pre, post)
        loss = criterion(outputs, mask.long())

        outputs = torch.argmax(outputs, axis=1)
        score = dice(outputs, mask)
        
        running_loss += loss.item()
        running_score += score.item()

    return running_loss / len(val_loader), running_score / len(val_loader)



def main():
    fin = open("engine_config.json")
    metadata = json.load(fin)
    fin.close()
    engine = Engine(**metadata)

    device = torch.device("cuda:0")
    ########Dataloaders #################
    data_root = "data"
    vector_dir = f"vectors/Original_Split-20230524T135331/MASK"
    f = open(f"{data_root}/{vector_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]

    chabud_train = ChabudDataset(
        data_root=data_root,
        json_dir=vector_dir[:2],
        data_list=train_list,
        window=512
    )

    chabud_val = ChabudDataset(
        data_root=data_root,
        json_dir=vector_dir[:2],
        data_list=val_list,
        window=512
    )

    train_loader = DataLoader(chabud_train, batch_size=4, shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=4, shuffle=False)
    
    keep = 5
    track_ckpts = []
    ckpt_path = f"checkpoints/BiDate_UNet_{datetime.now()}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)


    ############# model #####################
    net = BiDateNet(n_channels=12, n_classes=2)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EPOCHS = 20

    best_vloss = 1_000_000.0

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss, avg_score = train_one_epoch(train_loader=train_loader, net=net, 
                                                criterion=criterion, optimizer=optimizer,
                                                device=device)

        print("Train loss {} dice {}".format(avg_loss, avg_score))
        
        avg_vloss, avg_vscore = val(val_loader=val_loader, net=net, 
                                     criterion=criterion, device=device)
        
        print("Val loss {} dice {}".format(avg_vloss, avg_vscore))

        engine.log(step=epoch, train_loss=avg_loss, train_score=avg_score,
                   val_loss=avg_vloss, val_score=avg_vscore)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
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

