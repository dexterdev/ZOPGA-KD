"""LeNet-5 for 32x32 single-channel inputs, plus a half-width student.

Teacher: conv 1->6->16->120 (all 5x5), FC 120->84->C.  Params: 61,706.
Student: only conv1/conv2 filters halved (1->3->8->120), FCs unchanged: 35,820.

Spatial: 32 -> 28 -> 14 -> 10 -> 5 -> 1, so conv3 outputs 120x1x1.
"""

import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, conv1_filters=6, conv2_filters=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.conv3 = nn.Conv2d(conv2_filters, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def lenet5(num_classes=10):
    return LeNet5(num_classes, conv1_filters=6, conv2_filters=16)


def lenet5_half(num_classes=10):
    return LeNet5(num_classes, conv1_filters=3, conv2_filters=8)
