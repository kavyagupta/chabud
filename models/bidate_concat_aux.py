import torch
import torch.nn as nn

from models.unet_parts import down, outconv, up, inconv


class BiDateConcatNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BiDateConcatNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024*2, 256)
        self.up2 = up(512+256, 128)
        self.up3 = up(256+128, 64)
        self.up4 = up(128+64, 64)
        self.outc = outconv(64, n_classes)
        self.out_bands = outconv(64, n_channels)

    def forward(self, x_d1, x_d2):
        x1_d1 = self.inc(x_d1)
        x2_d1 = self.down1(x1_d1)
        x3_d1 = self.down2(x2_d1)
        x4_d1 = self.down3(x3_d1)
        x5_d1 = self.down4(x4_d1)

        x1_d2 = self.inc(x_d2)
        x2_d2 = self.down1(x1_d2)
        x3_d2 = self.down2(x2_d2)
        x4_d2 = self.down3(x3_d2)
        x5_d2 = self.down4(x4_d2)

        x = self.up1(torch.cat([x5_d2, x5_d1], dim=1), torch.cat([x4_d2, x4_d1], dim=1))
        x = self.up2(x, torch.cat([x3_d2, x3_d1], dim=1))
        x = self.up3(x, torch.cat([x2_d2 , x2_d1], dim=1))
        x = self.up4(x, torch.cat([x1_d2 , x1_d1], dim=1))
        out_mask = self.outc(x)
        out_bands = self.out_bands(x)
        return out_mask, out_bands
    
    

