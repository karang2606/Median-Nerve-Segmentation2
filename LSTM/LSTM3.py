#!/usr/bin/env python
# coding: utf-8

# In[1]:


from .TimeDistributed import TimeDistributed
from .ConvLSTM import ConvLSTM
from .unet_parts import *

class LSTM_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filter = 32, bilinear=False):
        super(LSTM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TimeDistributed(DoubleConv(n_channels, num_filter))
        self.down1 = TimeDistributed(Down(num_filter, num_filter*2))
        self.down2 = TimeDistributed(Down(num_filter*2, num_filter*4))
        self.down3 = TimeDistributed(Down(num_filter*4, num_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = TimeDistributed(Down(num_filter*8, num_filter*16 // factor))
        
        self.lstm = ConvLSTM(input_dim = num_filter*16 // factor,
                    hidden_dim=num_filter*16 // factor,
                    kernel_size=(3,3),
                    num_layers=2,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
        
        self.up1 = TimeDistributed(Up(num_filter*16, num_filter*8 // factor, bilinear))
        self.up2 = TimeDistributed(Up(num_filter*8, num_filter*4 // factor, bilinear))
        self.up3 = TimeDistributed(Up(num_filter*4, num_filter*2 // factor, bilinear))
        self.up4 = TimeDistributed(Up(num_filter*2, num_filter, bilinear))
        self.up5 = TimeDistributed(OutConv(num_filter, n_classes))
        self.outc = (OutConv(10, n_classes))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        [x], _ = self.lstm(x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        logits = self.outc(torch.squeeze(x, dim=2))
        return logits