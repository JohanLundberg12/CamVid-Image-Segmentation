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

**Utility functions and evaluation metrics:** 

Several of the utility functions and evaluation metrics are taken from the following Git Repository: https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/tree/master/U-Net/src 

## Experiments and results

We used two different model, UNET and VGG11. Each of these models were tested on the raw data, each of the six augmentations of the data with the raw data, and all the augmentations of the data combined with the raw data. We tested both with pretraining and without pretraining on the VGG11 model.
For each experiment, we obtained the following best results:

| *Model* 	| *Data* 	| *Test mIoU* 	| *Test loss* 	|
|:---------:	|:--------:	|:-------------:	|:-------------:	|
|    UNET   	| Original 	|      50.81         	|   14.63            	|
|    UNET   	|   O + 1  	|     61.93     	|     11.61     	|
|    UNET   	|    All   	|     66.81     	|      11.28         	|
|   VGG11   	| Original 	|     61.29      	|      13.81            	|
|   VGG11   	|   0 + 5  	|     66.12          	|       11.35        	|
|   VGG11   	|    All   	|     71.16            	|       6.38        	|
| VGG11 + P 	| Original 	|     58.18          	|      13.49         	|
| VGG11 + P 	|   0 + 3  	|     67.11          	|      11.42         	|
| VGG11 + P 	|    All   	|     71.03           	|      6.30         	|

## Discussion

- We observed that the pretraining of VGG11 did not improve model performance.
- We observed that data augmentation methods improved model performance.

**Points for future work:**

- Hyperparameter tuning could be optimized. 
- Classifying on related domains. 


## Install environment and packages

- Create new virtual environment: `python -m venv .venv`
- Activate environment: `source .venv/bin/activate`
- Install packages from file: `pip install -r requirements.txt`

