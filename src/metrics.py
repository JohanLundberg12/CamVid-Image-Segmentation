import torch

SMOOTH = 1e-6

def iou(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Kindly borrowed from:
    # https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/U-Net/src/IoU.py    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds
    
    return thresholded