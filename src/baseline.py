import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms as T
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


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], n_classes: int = 32
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
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
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

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
    camvid_train = CamVidDataset(
        train_imgs_path,
        train_labels_path,
        transformation)
    camvid_val = CamVidDataset(val_imgs_path, val_labels_path, transformation)
    camvid_test = CamVidDataset(
        test_imgs_path,
        test_labels_path,
        transformation)

    train_loader = DataLoader(
        camvid_train,
        batch_size=12,
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

    model = UNET(in_channels=3, out_channels=3)

    params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=0.00001)
    optimizer = optim.AdamW(params, lr=0.00001)
    scaler = torch.cuda.amp.GradScaler()

    AEModel_Unet = AEModelTrainer(model)

    model_name = 'unet_100_adamw'

    train_losses, valid_losses = AEModel_Unet.train(
        train_loader, val_loader, epochs=100, optimizer=optimizer,
        loss_fn=loss_fn, scaler=scaler, log_name=model_name)

    new_model = AEModelTrainer(model)

    preds, avg_test_iou, test_loss = new_model.predict(
        test_loader, MODEL_DIR / model_name, loss_fn)

    print(f'Test loss: {test_loss}, Test IoU: {avg_test_iou}')

    a = 1
