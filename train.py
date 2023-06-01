import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import dice

from models.bidate_model import BiDateNet
from utils.chabud_dataloader import ChabudDataset, Rescale_train, Rescale_target
from utils.args import parse_args


def train_one_epoch(train_loader, net, criterion, 
                    optimizer, epoch_index, tb_writer, device):
    running_loss = 0.0
    last_loss = 0.0
    running_score = 0.0
    last_score = 0.0

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs_pre, inputs_post, mask = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
        )

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs_pre, inputs_post)
        # outputs = torch.squeeze(outputs)
        mask = torch.unsqueeze(mask, 1)
        loss = criterion(outputs, mask.float())

        outputs = torch.sigmoid(outputs)
        # outputs[outputs>=0.5] = 1
        # outputs[outputs<0.5] = 0
        score = dice(outputs, mask.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_score += score.item()

        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            last_score = running_score / 10  # score per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            print("  batch {} score: {}".format(i + 1, last_score))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            tb_writer.add_scalar("Loss/score", last_score, tb_x)
            running_loss = 0.0
            running_score = 0.0

    return last_loss, last_score

def val(val_loader, net, criterion, tb_writer, device):
    net.eval()

    running_vloss = 0.0
    running_vscore = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs_pre, v_inputs_post, vmask = (
                vdata[0].to(device),
                vdata[1].to(device),
                vdata[2].to(device),
            )
            voutputs = net(vinputs_pre, v_inputs_post)
            vmask = torch.unsqueeze(vmask, 1)
            vloss = criterion(voutputs, vmask.float())
            running_vloss += vloss

            voutputs = torch.sigmoid(voutputs)
            # voutputs[voutputs>=0.5] = 1
            # voutputs[voutputs<0.5] = 0
            running_vscore += dice(voutputs, vmask.int())

    avg_vloss = running_vloss / (i + 1)
    avg_vscore = running_vscore / (i + 1)
   

    # Log the running loss averaged per batch
    # for both training and validation

    return avg_vloss, avg_vscore



def main():

    device = torch.device("cuda:0")
    ########Dataloaders #################
    f = open("../CHABUD/vectors/Original_Split-20230524T135331/MASK/metadata.json")
    data = json.load(f)
    train_list = data["dataset"]["train"]
    val_list = data["dataset"]["val"]

    chabud_train = ChabudDataset(
        json_dir="../CHABUD/vectors/Original_Split-20230524T135331/MASK",
        data_list=train_list,
        transform=Rescale_train(512),
        target_transform=Rescale_target(512),
    )

    chabud_val = ChabudDataset(
        json_dir="../CHABUD/vectors/Original_Split-20230524T135331/MASK",
        data_list=val_list,
        transform=Rescale_train(512),
        target_transform=Rescale_target(512),
    )

    train_loader = DataLoader(chabud_train, batch_size=4, shuffle=True)
    val_loader = DataLoader(chabud_val, batch_size=4, shuffle=True)


    ############# model #####################
    net = BiDateNet(n_channels=12, n_classes=1)
    net = net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/chabud_trainer_{}".format(timestamp))
    epoch_number = 0

    EPOCHS = 20

    best_vloss = 1_000_000.0

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss, avg_score = train_one_epoch(train_loader=train_loader, net=net, 
                                                criterion=criterion, optimizer=optimizer,
                                                epoch_index=epoch_number, tb_writer=writer, 
                                                device=device)

        
        
        avg_vloss, avg_vscore = val(val_loader=val_loader, net=net, 
                                     criterion=criterion, tb_writer=writer, 
                                     device=device)
        
        writer.add_scalars(
            "Training vs. Validation Loss",
            {
                "Training": avg_loss,
                "Validation": avg_vloss,
            },
            epoch_number + 1,
        )
        writer.flush()
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        print("score train {} valid {}".format(avg_score, avg_vscore))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(net.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    main()

