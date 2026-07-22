"""AlexNet variant for 32x32 CIFAR-10 (TF-port style with BatchNorm).

Teacher (full width):  conv 48/128/192/192/128, FC 1152->512->256->C.
Student (half width):  conv 24/64/96/96/64,    FC  576->256->128->C.

Spatial flow (3x3 stride-2 pools): 32 -> 15 -> 7 -> 3, so the flattened
feature size is 128*3*3 = 1152 (64*3*3 = 576 for the student).
"""

import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes=10, channels=(48, 128, 192, 192, 128),
                 fc_hidden=(512, 256)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.features = nn.Sequential(
            _conv_bn_relu(3, c1, 5, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 32 -> 15
            _conv_bn_relu(c1, c2, 5, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 15 -> 7
            _conv_bn_relu(c2, c3, 3, 1),
            _conv_bn_relu(c3, c4, 3, 1),
            _conv_bn_relu(c4, c5, 3, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 7 -> 3
        )
        fc_in = c5 * 3 * 3
        h1, h2 = fc_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def alexnet(num_classes=10):
    return AlexNetCifar(num_classes, channels=(48, 128, 192, 192, 128),
                        fc_hidden=(512, 256))


def alexnet_half(num_classes=10):
    return AlexNetCifar(num_classes, channels=(24, 64, 96, 96, 64),
                        fc_hidden=(256, 128))
