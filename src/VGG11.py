from argparse import ArgumentParser
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

from AEModel import AEModelTrainer
from camvid_dataloader import CamVidDataset
from config import CAMVID_DIR, MODEL_DIR
from DoubleConv import DoubleConv


class VGG11(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features: List[int] = [64, 128, 256, 256], n_classes: int = 32, pretrained: bool = True):
        super(VGG11, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Down part of VGG11
        self.pretrained_encoder = models.vgg11(pretrained=pretrained).features
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

        self.ups.append(nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(384, 256))

        self.ups.append(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(192, 128))

        self.ups.append(nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(96, 64))

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        conv1 = self.relu(self.conv1(x))
        skip_connections.append(conv1)  # 1st skip_connection

        conv2 = self.relu(self.conv2(self.pool(conv1)))
        skip_connections.append(conv2)  # 2nd skip_connection

        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))  # 3rd
        skip_connections.append(conv3)

        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))  # 4th
        skip_connections.append(conv4)

        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))  # 5th
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


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--augmentation', type=str)
    parser.add_argument('--pretraining', type=bool, default=True)

    args = parser.parse_args()

    # Specify paths
    train_imgs_path = CAMVID_DIR / 'train_augment'
    val_imgs_path = CAMVID_DIR / 'val'
    test_imgs_path = CAMVID_DIR / 'test'
    train_labels_path = CAMVID_DIR / 'train_labels_augment'
    val_labels_path = CAMVID_DIR / 'val_labels'
    test_labels_path = CAMVID_DIR / 'test_labels'

    # Define input size and transformations
    input_size = (128, 128)
    transformation = T.Compose([T.Resize(input_size, 0)])

    augmentations = ['00']
    if args.augmentation == 'all':
        augmentations = ['00', '01', '02', '03', '04', '05', '06']
    elif args.augmentation == 'none':
        augmentations = ['00']
    else:
        augmentations.append(args.augmentation)

    # Define training and validation datasets
    camvid_train = CamVidDataset(
        train_imgs_path,
        train_labels_path,
        transformation,
        train=True,
        augmentations=augmentations)
    camvid_val = CamVidDataset(
        val_imgs_path,
        val_labels_path,
        transformation,
        train=False)
    camvid_test = CamVidDataset(
        test_imgs_path,
        test_labels_path,
        transformation,
        train=False)

    train_loader = DataLoader(
        camvid_train,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    val_loader = DataLoader(
        camvid_val,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = DataLoader(
        camvid_test,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    model = VGG11(in_channels=3, out_channels=3, pretrained=args.pretraining)

    params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=0.00001)
    optimizer = optim.AdamW(params, lr=0.00001)
    scaler = torch.cuda.amp.GradScaler()

    AEModel = AEModelTrainer(model)

    if args.pretraining == True:
        model_name = f'vgg_p_40_00_{args.augmentation}'
    else:
        model_name = f'vgg_40_00_{args.augmentation}'

    train_losses, valid_losses = AEModel.train(
        train_loader, val_loader, epochs=40, optimizer=optimizer,
        loss_fn=loss_fn, scaler=scaler, log_name=model_name)

    preds, avg_test_iou, test_loss = AEModel.predict(
        test_loader, MODEL_DIR / model_name, loss_fn)

    print(f'Model name: {model_name}')
    print(f'Test loss: {test_loss}, Test IoU: {avg_test_iou}')
