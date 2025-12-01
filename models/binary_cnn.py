import torch.nn as nn
from torchvision.ops import DropBlock2d


class AudioBinaryCNN(nn.Module):
    def __init__(self, drop_prob=0.0, block_size=3):
        super(AudioBinaryCNN, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(p=0.1),
            DropBlock2d(p=drop_prob, block_size=block_size),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(p=0.1),
            DropBlock2d(p=drop_prob, block_size=block_size),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(p=0.1),
            DropBlock2d(p=drop_prob, block_size=block_size),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(p=0.1),
            DropBlock2d(p=drop_prob, block_size=block_size),

        )

        """self.proj = nn.Linear(4096, 256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )"""

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.features(x)

        #b, c, f, t = out.size()
        #out = out.permute(0, 3, 1, 2)
        #out = out.contiguous().view(b, t, -1)

        #out = self.proj(out)

        #rnn_out, _ = self.gru(out)
        #rnn_out = rnn_out.mean(dim=1)

        out = self.global_pool(out).squeeze(-1).squeeze(-1)

        out = self.dropout(out)
        out = self.fc(out)
        return out