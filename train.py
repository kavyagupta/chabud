import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import dice

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
        mask = torch.unsqueeze(mask, 1)
        loss = criterion(outputs, mask.float())

        print (outputs.min(), outputs.max())
        outputs = torch.sigmoid(outputs)
        print (outputs.min(), outputs.max())
        score = dice(outputs, mask.long())
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
        mask = torch.unsqueeze(mask, 1)
        loss = criterion(outputs, mask.float())

        outputs = torch.sigmoid(outputs)
        score = dice(outputs, mask.long())
        
        running_loss += loss.item()
        running_score += score.item()

    return running_loss / len(val_loader), running_score / len(val_loader)



def main():

    device = torch.device("cuda:0")
    ########Dataloaders #################
    json_dir = "../CHABUD/vectors/Original_Split-20230524T135331/MASK"
    f = open(f"{json_dir}/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]

    chabud_train = ChabudDataset(
        json_dir=json_dir,
        data_list=train_list,
        window=512
    )

    chabud_val = ChabudDataset(
        json_dir=json_dir,
        data_list=val_list,
        window=512
    )

    train_loader = DataLoader(chabud_train, batch_size=4, shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=4, shuffle=False)


    ############# model #####################
    net = BiDateNet(n_channels=12, n_classes=1)
    net = net.to(device)
    criterion = nn.BCEWithLogitsLoss()
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

        
        
        avg_vloss, avg_vscore = val(val_loader=val_loader, net=net, 
                                     criterion=criterion, device=device)
        
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        print("score train {} valid {}".format(avg_score, avg_vscore))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}".format(timestamp, epoch)
            torch.save(net.state_dict(), model_path)

if __name__ == "__main__":
    main()

