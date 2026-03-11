import torch
import torch.nn as nn

class ConvLSTM(nn.Module):

    def __init__(self, input_channels=1, hidden_channels=16):

        super().__init__()

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1
        )

        self.output = nn.Conv2d(hidden_channels,1,kernel_size=1)

        self.hidden_channels = hidden_channels


    def forward(self, x):

        batch, seq, ch, h, w = x.size()

        h_state = torch.zeros(batch,self.hidden_channels,h,w).to(x.device)

        for t in range(seq):

            inp = torch.cat([x[:,t],h_state],dim=1)

            h_state = torch.tanh(self.conv(inp))

        out = self.output(h_state)

        return out