from typing import List
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T

from DoubleConv import DoubleConv


class VGG11(nn.Module):
    def __init__(self,in_channels=3, out_channels=1, features:List[int] = [64, 128, 256, 256], n_classes: int = 32):
        super(VGG11, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)        

        # Down part of VGG11
        self.pretrained_encoder = models.vgg11(pretrained=True).features
        self.relu = self.pretrained_encoder[1]

        self.conv1 = self.pretrained_encoder[0]

        self.conv2 = self.pretrained_encoder[3]

        self.conv3s = self.pretrained_encoder[6]
        self.conv3 = self.pretrained_encoder[8]

        self.conv4s = self.pretrained_encoder[11]
        self.conv4 = self.pretrained_encoder[13]

        self.conv5s = self.pretrained_encoder[16]
        self.conv5 = self.pretrained_encoder[18]

        # Up part
        self.ups = nn.ModuleList()

        self.ups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(768, 512))
        
        self.ups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(768, 512))

        self.ups.append(nn.ConvTranspose2d(512,128, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(384,256))

        self.ups.append(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(192, 128))

        self.ups.append(nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(96, 64))

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        conv1 = self.relu(self.conv1(x))
        skip_connections.append(conv1) #1st skip_connection

        conv2 = self.relu(self.conv2(self.pool(conv1)))
        skip_connections.append(conv2) #2nd skip_connection

        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s)) #3rd
        skip_connections.append(conv3)

        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s)) #4th
        skip_connections.append(conv4)

        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s)) #5th
        skip_connections.append(conv5)

        skip_connections = skip_connections[::-1]
        
        x = conv5

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
