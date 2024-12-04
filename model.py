import torch
import torch.nn as nn
import torch.nn.functional as F


# DropConnect Layer
class DropConnect(nn.Module):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.rand_like(x) > self.p
        mask = mask.to(x.device)
        scale = 1.0 / (1.0 - self.p)
        return mask * x * scale


# LRFBlock
class LRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=0.1):
        super(LRFBlock, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), stride=stride, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), stride=stride, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_linear = nn.Conv2d(4 * out_channels, out_channels, 1, stride)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        self.scale = scale

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.conv_linear(out)

        shortcut = self.bn_shortcut(self.shortcut(x))

        out = out * self.scale + shortcut
        out = F.relu(out)

        return out


# LRF-CNN Model
class LRFCNN(nn.Module):
    def __init__(self, num_classes=5, dropconnect_prob=0.5):
        super(LRFCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LRFBlock(64, 64),
            LRFBlock(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LRFBlock(128, 192),
            LRFBlock(192, 224),
            LRFBlock(224, 256),
            LRFBlock(256, 288),
            LRFBlock(288, 320),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LRFBlock(320, 320),
            LRFBlock(320, 384),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropconnect = DropConnect(p=dropconnect_prob)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.attention_fc = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropconnect(x)

        gap_out = self.gap(x)
        gap_out = gap_out.view(gap_out.size(0), -1)

        attention_weights = self.attention_fc(gap_out)
        x = x * attention_weights.view(-1, 1, 1, 1)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LRFCNN(num_classes=5, dropconnect_prob=0.5).to(device)
