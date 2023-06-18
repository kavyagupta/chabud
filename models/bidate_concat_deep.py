import torch
import torch.nn as nn

from models.unet_parts import down, outconv, up, inconv


class BiDateConcatDeepNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BiDateConcatDeepNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)

        self.up1 = up(2048*2, 512)
        self.up2 = up(1024+512, 256)
        self.up3 = up(512+256, 128)
        self.up4 = up(256+128, 64)
        self.up5 = up(128+64, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x_d1, x_d2):
        x1_d1 = self.inc(x_d1)
        x2_d1 = self.down1(x1_d1)
        x3_d1 = self.down2(x2_d1)
        x4_d1 = self.down3(x3_d1)
        x5_d1 = self.down4(x4_d1)
        x6_d1 = self.down5(x5_d1)

        x1_d2 = self.inc(x_d2)
        x2_d2 = self.down1(x1_d2)
        x3_d2 = self.down2(x2_d2)
        x4_d2 = self.down3(x3_d2)
        x5_d2 = self.down4(x4_d2)
        x6_d2 = self.down5(x5_d2)

        x = self.up1(torch.cat([x6_d2, x6_d1], dim=1), torch.cat([x5_d2, x5_d1], dim=1))
        x = self.up2(x, torch.cat([x4_d2, x4_d1], dim=1))
        x = self.up3(x, torch.cat([x3_d2 , x3_d1], dim=1))
        x = self.up4(x, torch.cat([x2_d2 , x2_d1], dim=1))
        x = self.up5(x, torch.cat([x1_d2, x1_d1], dim=1))
        x = self.outc(x)
        return x
    


    
    

