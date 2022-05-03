import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import CAMVID_DIR
from utils import Color_map
from utils import rgb_to_mask


# https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/tree/master/U-Net/src

class CamVidDataset():

    def __init__(self, img_pth, label_pth, transform):
        self.img_pth = img_pth
        self.label_pth = label_pth
        self.transform = transform

        self.total_imgs = os.listdir(self.img_pth)
        self.total_labels = [img_name[:-4] + '_L' + img_name[-4:] for img_name in all_imgs]
        code2id, id2code, name2id, id2name = Color_map(CAMVID_DIR/'class_dict.csv')
        self.id2code = id2code

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_pth, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label_loc = os.path.join(self.label_pth, self.total_labels[idx])
        label = Image.open(label_loc).convert("RGB")
        out_image, rgb_label = self.transform(image), self.transform(label)
        out_image = transforms.Compose([transforms.ToTensor()])(out_image)
        rgb_label = transforms.Compose([transforms.PILToTensor()])(rgb_label)
        out_label = rgb_to_mask(
            torch.from_numpy(
                np.array(rgb_label)).permute(
                1, 2, 0), self.id2code)

        return out_image, out_label, rgb_label.permute(0, 1, 2)


if __name__ == '__main__':
    input_size = (128, 128)
    transformation = transforms.Compose([transforms.Resize(input_size, 0)])

    train_imgs_path = CAMVID_DIR / 'train'
    train_labels_path = CAMVID_DIR / 'train_labels'

    camvid = CamVidDataset(train_imgs_path, train_labels_path, transformation)

    a = 1
