import os
from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms
from config import CAMVID_DIR

class CamVidDataSet():
    def __init__(self, imgs_path, labels_path, transform):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.transform = transform
        
        self.imgs = os.listdir(self.imgs_path)
        self.labels = list(map(lambda x: x[:-4] + '_L.png', self.imgs)) #see train_labels
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img_loc = os.path.join(self.imgs_path, self.imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        
        label_loc = os.path.join(self.labels_path, self.labels[idx])
        label = Image.open(label_loc).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        
        image_tensor = transforms.Compose([transforms.ToTensor()])(image) 
        label_tensor = transforms.Compose([transforms.PILToTensor()])(label)
        
        return image_tensor, label_tensor

if __name__ == '__main__':
    input_size = (128, 128)
    transformation = transforms.Compose([transforms.Resize(input_size, 0)])

    train_imgs_path = CAMVID_DIR / 'train'
    train_labels_path = CAMVID_DIR / 'train_labels'

    camvid = CamVidDataSet(train_imgs_path, train_labels_path, transformation)
