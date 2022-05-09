# CamVid-Image-Segmentation
Image segmentation with deep encoder-decoder networks using the Cambridge-driving Labeled Video Database (CamVid). 


## The Data

CamVid (Cambridge-driving Labeled Video Database) is a road/driving scene understanding database which was originally captured as five video sequences with a 960×720 resolution camera mounted on the dashboard of a car. Those sequences were sampled (four of them at 1 fps and one at 15 fps) adding up to 701 frames. Those stills were manually annotated with 32 classes: void, building, wall, tree, vegetation, fence, sidewalk, parking block, column/pole, traffic cone, bridge, sign, miscellaneous text, traffic light, sky, tunnel, archway, road, road shoulder, lane markings (driving), lane markings (non-driving), animal, pedestrian, child, cart luggage, bicyclist, motorcycle, car, SUV/pickup/truck, truck/bus, train, and other moving object. 

## Central Problem

The challenge is to segment road images into one of the 32 classes. In this classification task, some classes are only present in a limited amount of the pixels. For example pedestrians on such images will naturally take up a smaller amount of the pixels than the sky, but the pedestrians are important to classify correctly.

## Methods

**Data augmentation methods:**

We have used some simple data augmentation methods in order to increase the amount of data.  We have used the ‘transform’ module from torchvision for all transformation except the gaussian noise, where we have created our own function, ‘AddGaussianNoise’, which was inspired by the method used in the following blogpost: 
https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745.

We used six ways of augmenting data. 

1: horizontally flipping the data. 
2: resizing and cropping the data
3: adjusting brightness
4: adjusting contrast
5: adding gaussian noise
6: combining all previously mentioned methods into one. 

We, of course, transformed labels along with the data when necessary.

## Experiments and results

We used two different model, UNET and VGG11. Each of these models were tested on the raw data, each of the six augmentations of the data with the raw data, and all the augmentations of the data combined with the raw data. We tested both with pretraining and without pretraining on the VGG11 model.
For each experiment, we obtained the following best results:

**UNET originial data:** 

- Parameters: 
  - Augmentation: None
  - Learning rate: 0.0001
  - Epochs: 100
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU: 60.6
- Test loss: 10.40

**UNET original data + one augmented data method:**

- Parameters: 
  - Augmentation: 01
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU: 61.93
- Test loss: 11.61

**UNET all data:**

- Parameters: 
  - Augmentation: All
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU: 66.81
- Test loss: **??**

**VGG11 not pretrained originial data:**

- Parameters:
  - Augmentation: None
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU: **??**
- Test loss: **??**

**VGG11 not pretrained original data + one augmented data method:**

- Parameters: 
  - Augmentation:
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU:
- Test loss:

**vGG11 not pretrained all data:**

- Parameters: 
  - Augmentation:
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU:
- Test loss:

**VGG11 pretrained originial data:**

- Parameters:
  - Augmentation: None
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU:
- Test loss:

**VGG11 pretrained original data + one augmented data method:**

- Parameters: 
  - Augmentation:
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU:
- Test loss:

**VGG11 pretrained all data:**

- Parameters: 
  - Augmentation:
  - Learning rate: 0.0001
  - Epochs: 40
  - Optimization: AdamW
  - Loss function: CE
- Test mIoU:
- Test loss:

## Discussion

## Install environment and packages

- Create new virtual environment: `python -m venv .venv`
- Activate environment: `source .venv/bin/activate`
- Install packages from file: `pip install -r requirements.txt`

