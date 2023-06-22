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
        
        _ , [(_, c)] = self.lstm(x5)
        
        x = self.up1(c, x4[:,-1,...])
        x = self.up2(x, x3[:,-1,...])
        x = self.up3(x, x2[:,-1,...])
        x = self.up4(x, x1[:,-1,...])
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

