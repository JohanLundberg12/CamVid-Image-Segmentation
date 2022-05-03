from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader

from AEModel import AEModelTrainer
from camvid_dataloader import CamVidDataset
from config import CAMVID_DIR, MODEL_DIR



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

    
class UNETVGG11(nn.Module):
    def __init__(self,in_channels=3, out_channels=1, features:List[int] = [64, 128, 256, 256], n_classes: int = 32):
        super(UNETVGG11, self).__init__()
        
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
        
        #self.bottleneck = DoubleConv(512, 1024)

        # Up part
        self.ups = nn.ModuleList()

        #self.first_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        #self.first_up_cov = DoubleConv(1024, 512)

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

        #for feature in reversed(features):
        #    self.ups.append(
        #        nn.ConvTranspose2d(
        #            feature*2, feature, kernel_size=2, stride=2,
        #        )
        #    )
        #    self.ups.append(DoubleConv(feature*2 + feature, feature*2))

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
        
        # Up part
        #x = nn.ConvTranspose2d(x.shape[1], x.shape[1]/2, kernel_size=2, stride=2)
        #x = torch.cat((skip_connections[0], x), dim=1)
        #x = DoubleConv(x.shape[1]*2, x.shape[1])

        #x = torch.cat((skip_connections[1], x), dim=1)

        #x = self.first_up(conv5) 
        #x = self.first_up_cov(x)

        x = conv5

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], n_classes: int = 32, 
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    
    def encoder(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        z = self.bottleneck(x)

        return z, skip_connections

    
    def decoder(self, x, skip_connections):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return x
        
    
    def forward(self, x):
        z, skip_connections = self.encoder(x)
        skip_connections = skip_connections[::-1]
        x = self.decoder(z, skip_connections)

        return self.final_conv(x)  
    

if __name__ == "__main__":

    # Specify paths
    train_imgs_path = CAMVID_DIR / 'train'
    val_imgs_path = CAMVID_DIR / 'val'
    test_imgs_path = CAMVID_DIR / 'test'
    train_labels_path = CAMVID_DIR / 'train_labels'
    val_labels_path = CAMVID_DIR / 'val_labels'
    test_labels_path = CAMVID_DIR / 'test_labels'

    # Define input size and transformations
    input_size = (128, 128)
    transformation = T.Compose([T.Resize(input_size, 0)])

    # Define training and validation datasets
    camvid_train = CamVidDataset(train_imgs_path, train_labels_path, transformation)
    camvid_val = CamVidDataset(val_imgs_path, val_labels_path, transformation)
    #camvid_test = CamVidDataset(test_imgs_path, test_labels_path, transformation)

    train_loader = DataLoader(
        camvid_train,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    val_loader = DataLoader(
        camvid_val,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    #test_loader = DataLoader(
    #    camvid_test,
    #    batch_size=2,
    #    num_workers=4,
    #    pin_memory=True,
    #    shuffle=False,
    #)

    model = UNETVGG11(in_channels=3, out_channels=3)

    params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=0.00001)
    scaler = torch.cuda.amp.GradScaler()

    AEModel_Unet = AEModelTrainer(model)

    model_name = 'unet'

    train_losses, valid_losses = AEModel_Unet.train(train_loader, val_loader, epochs=50, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler, log_name=model_name)

    #new_model = AEModelTrainer(model)

    #preds, avg_test_iou, test_loss = new_model.predict(test_loader, MODEL_DIR / model_name, loss_fn)

    a = 1