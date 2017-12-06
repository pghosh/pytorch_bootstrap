import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import models

IMG_HEIGHT = 480
IMG_WIDTH = 640

class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=False):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = nn.Sequential(
            nn.Conv2d(insize, outsize, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(outsize),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(pool, pool)
        self.avgpool = nn.AvgPool2d(pool, pool)
        self.avg = avg

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        else:
            x = self.maxpool(x)
        return x


class FineTuneImageNet(nn.Module):
    def __init__(self, num_classes, is_trainable=False):
        super(FineTuneImageNet, self).__init__()
        # vgg16
        self.features = models.vgg16(pretrained=True).features
        # self.features = models.resnet50(pretrained=True)
        # num_ftrs = self.features.fc.in_features
        # freeze layers before any change
        if not is_trainable:
            # Freeze those weights
            for p in self.features.parameters():
                p.requires_grad = False

        # Now add new layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if (num_classes == 1):
            self.sig = nn.Sigmoid()
        else:
            self.sig = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.sig is not None:
            x = self.sig(x)
        return x


def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


if __name__ == '__main__':
    model = CustomIcebergNet(1)
    print(model)
    # for parameter in model.parameters():
    #     print(parameter)
