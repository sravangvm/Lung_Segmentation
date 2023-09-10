import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv11 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv13 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv15 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.pool1(x1)

        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x2 = self.pool2(x2)

        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x3 = self.pool3(x3)

        x4 = self.conv8(x3)
        x4 = self.conv9(x4)
        x4 = self.conv10(x4)
        x4 = self.upsample1(x4)

        x5 = torch.cat((x4, x3), 1)
        x5 = self.conv11(x5)
        x5 = self.conv12(x5)
        x5 = self.upsample2(x5)

        x6 = torch.cat((x5, x2), 1)
        x6 = self.conv13(x6)
        x6 = self.conv14(x6)
        x6 = self.upsample3(x6)

        x7 = torch.cat((x6, x1), 1)
        x7 = self.conv15(x7)
        x7 = self.conv16(x7)
        x7 = self.conv17(x7)
        x7 = self.sigmoid(x7)

        return x7