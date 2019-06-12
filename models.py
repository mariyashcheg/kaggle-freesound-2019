import torch
from torch import nn
from torch.nn import functional as F
import math
import time


class ConvGatedBlock(nn.Module):
    def __init__(self, input_channels, mid_channels, factor):
        super(ConvGatedBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, 2 * mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * mid_channels),
            nn.GLU(dim=1),
            nn.Conv2d(mid_channels, 2 * mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * mid_channels),
            nn.GLU(dim=1),
            nn.MaxPool2d((1, factor))
        )

    def forward(self, x):
        return self.block(x)


class ConvGRNN(nn.Module):
    def __init__(self, block, mel_frame):
        super(ConvGRNN, self).__init__()
        self.num_channels = 64
        self.hidden_size = 128
        self.mel_frame = mel_frame

        self.block1 = block(1, self.num_channels, 2)
        self.block2 = block(self.num_channels, self.num_channels, 2)
        self.block3 = block(self.num_channels, self.num_channels, 2)
        self.block4 = block(self.num_channels, self.num_channels, 2)

        self.cnn = nn.Conv2d(self.num_channels, 2 * self.num_channels, 3, padding=1)
        self.pool = nn.MaxPool2d((1, 4))

        self.bigru = nn.GRU(input_size=2 * self.num_channels, hidden_size=self.hidden_size, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.act = nn.GLU(dim=-1)

        self.fc_att = nn.Linear(self.num_channels, 80)
        self.fc_loc = nn.Linear(self.num_channels, 80)

        self.fc = nn.Linear(2 * self.num_channels, 80)

    def forward(self, x, hidden):

        x_att = torch.clamp(torch.sigmoid(self.fc_att(torch.squeeze(x))), 1e-7, 1.)
        x_loc = F.softmax(self.fc_loc(torch.squeeze(x)), dim=-1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(self.cnn(x))

        x = torch.transpose(torch.squeeze(x), 1, 2)
        x, _ = self.bigru(x, hidden)
        x = self.act(x)

        #         cls, att = torch.sigmoid(self.fc1(x)), F.softmax(self.fc2(x), dim=-1)
        #         att = torch.clamp(att, 1e-7)
        #         x = torch.div(torch.sum(torch.mul(cls, att), dim=1), torch.sum(att, dim=1))

        x = self.fc(x)
        x = torch.mul(torch.mul(x_att, x), x_loc)
        x = torch.div(torch.sum(x, dim=1), torch.sum(x_loc, dim=1)+1e-5)

        return x

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, requires_grad=True)


def cgrnn(mel_frame, **kwargs):
    print('ConvGRNN')
    return ConvGRNN(ConvGatedBlock, mel_frame, **kwargs)


class ConvGRNN_simple(nn.Module):
    def __init__(self):
        super(ConvGRNN_simple, self).__init__()
        self.input_size = 20
        self.hidden_size = 32
        self.n_layers = 3
        self.cnn1layer = nn.Sequential(nn.Conv1d(1,1,50),
                                       nn.MaxPool1d(10),
                                       nn.ReLU())
        self.bigru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bidirectional=True, batch_first=True)
        self.fc_att = nn.Linear(256, 80)
        self.fc_loc = nn.Linear(256, 80)
        self.fnn = nn.Sequential(nn.Linear(2*self.hidden_size, 4*self.hidden_size), nn.ReLU(), nn.Dropout2d(p=0.2))
        self.fc = nn.Linear(4*self.hidden_size, 80)

    def forward(self, x, hidden):
        x = torch.squeeze(x)
        x_att = torch.sigmoid(self.fc_att(x))
        x_loc = F.softmax(self.fc_loc(x), dim=-1)
        for t in range(x.size(1)):
            if t == 0:
                features = self.cnn1layer(x[:, t, :].view(x.size(0), 1, -1))
            else:
                features = torch.cat((features, self.cnn1layer(x[:, t, :].view(x.size(0), 1, -1))), dim=1)
        features, _ = self.bigru(features, hidden)
        features = self.fnn(features)
        features = self.fc(features)
        out = torch.mul(torch.mul(x_att, features), x_loc)
        out = torch.div(torch.sum(out, dim=1), torch.sum(x_loc, dim=1))
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(2*self.n_layers, batch_size, self.hidden_size, requires_grad=True)


    def compute_loss(self, entry, hidden, device=None):
        logmel = entry['logmel'].to(device)
        labels = entry['labels'].to(device)
        logits = self.forward(logmel, hidden)
        # print('nn out', logits.size())
        probs = F.sigmoid(logits).cpu().data.numpy()
        # print('bce', probs.size(), labels.size())
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        return dict(
            loss=bce.item(),
            probs=probs,
        )

def cgrnn_simple(**kwargs):
    print('simple CGRNN with att&loc')
    return ConvGRNN_simple(**kwargs)
