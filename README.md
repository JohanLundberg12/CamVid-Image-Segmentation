# CamVid-Image-Segmentation
Image segmentation with deep encoder-decoder networks using the Cambridge-driving Labeled Video Database (CamVid). 


## The Data

CamVid (Cambridge-driving Labeled Video Database) is a road/driving scene understanding database which was originally captured as five video sequences with a 960×720 resolution camera mounted on the dashboard of a car. Those sequences were sampled (four of them at 1 fps and one at 15 fps) adding up to 701 frames. Those stills were manually annotated with 32 classes: void, building, wall, tree, vegetation, fence, sidewalk, parking block, column/pole, traffic cone, bridge, sign, miscellaneous text, traffic light, sky, tunnel, archway, road, road shoulder, lane markings (driving), lane markings (non-driving), animal, pedestrian, child, cart luggage, bicyclist, motorcycle, car, SUV/pickup/truck, truck/bus, train, and other moving object. 

## Central Problem

The challenge is to segment road images into several classes, i.e. pedestrians, roads, cars etc. In this classification, some classes arw only present in a limited amount of the pixels. For example pedestrians on such images will naturally take up a smaller amount of the pixels than the sky, but the pedestrians are important to classify correctly.

## Methods

## Experiments and results

## Discussion

## Install environment and packages

- Create new virtual environment: `python -m venv .venv`
- Activate environment: `source .venv/bin/activate`
- Install packages from file: `pip install -r requirements.txt`

