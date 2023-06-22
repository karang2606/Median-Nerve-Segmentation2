""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, num_filter = 32, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_filter))
        self.down1 = (Down(num_filter, num_filter*2))
        self.down2 = (Down(num_filter*2, num_filter*4))
        self.down3 = (Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_filter*8, num_filter*16 // factor))
        self.up1 = (Up(num_filter*16, num_filter*8 // factor, bilinear))
        self.up2 = (Up(num_filter*8, num_filter*4 // factor, bilinear))
        self.up3 = (Up(num_filter*4, num_filter*2 // factor, bilinear))
        self.up4 = (Up(num_filter*2, num_filter, bilinear))
        self.outc = (OutConv(num_filter, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits