#Code inspiration
#https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

import os
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import cv2
from config import CAMVID_DIR

#from camvid_dataloader import CamVidDataset

#Loading in manually
test_imgs_path = CAMVID_DIR / 'test'
test_labels_path = CAMVID_DIR / 'test_labels'
preds_labels_path = CAMVID_DIR / 'preds_labels'

#Using predictions obtained from the run of the model
preds = np.load(preds_labels_path / 'preds.npy')

#Class taken from dataloader
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
        image_out = self.transform(image)
        
        label_loc = os.path.join(self.labels_path, self.labels[idx])
        label = Image.open(label_loc).convert("RGB")
        label_out = self.transform(label)
        
        image_tensor = transforms.Compose([transforms.ToTensor()])(image_out) 
        label_tensor = transforms.Compose([transforms.PILToTensor()])(label_out)
        
        return image_out, label_out

if __name__ == "__main__":
    #Accesing the test images
    input_size = (128, 128)
    transformation = transforms.Compose([transforms.Resize(input_size, 0)])
    camvid = CamVidDataSet("data/CamVid/test/", "data/CamVid/test_labels/", transformation)

    #Example of image from testset and the predicted output image
    test_image = cv2.cvtColor(np.array(camvid[10][1]), cv2.COLOR_RGB2BGR)
    pred_image = cv2.cvtColor(preds[10], cv2.COLOR_RGB2BGR) 


    #Convert to grayscale instead of color - used in the similarity calculation
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)

    #Calculate similarity and the difference image
    (similarity, image_diff) = structural_similarity(test_gray, pred_gray, full=True)
    print("Image similarity", similarity)

    #Converting the difference image back to RGB color scheme
    image_diff = (image_diff * 255).astype("uint8")
    rgb=cv2.cvtColor(image_diff,cv2.COLOR_GRAY2RGB)

    #Finding the incorrect prediction pixels (i.e. everything other 
    #than (255,255,255)) and converting them to red (i.e. (255,0,0)).
    #The correcly predicted pixels are converted to green, (0,255,0).
    rgb2=np.where(rgb != [255,255,255], [255,0,0], [0,255,0])

    #Visualising the test image, prediction image, correct/incorrect image
    #and an image showing correctness as a spectrum.
    fig, ax = plt.subplots(2,2)

    ax[0,0].imshow(test_image)
    ax[0,0].axis('off')
    ax[0,0].set_title("Test image")

    ax[0,1].imshow(pred_image)
    ax[0,1].axis('off')
    ax[0,1].set_title("Predicted image")

    ax[1,0].imshow(rgb2)
    ax[1,0].axis('off')
    ax[1,0].set_title("Correct/incorrect plot")

    ax[1,1].imshow(image_diff, cmap="RdYlGn")
    ax[1,1].axis('off')
    ax[1,1].set_title("Correctness Spectrum")

    plt.show()





