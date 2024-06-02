from albumentations import Crop, CenterCrop
from albumentations import GaussianBlur
from albumentations import OpticalDistortion, GridDistortion, ElasticTransform
from albumentations import RandomBrightnessContrast 
import numpy as np
import random

"""
Returns one list of augmented scans and another list of the corresponding segmentations.
"""
def augment_data(scans, segmentations):
    
    new_scans = []
    new_segmentations = []
    for scan, segmentation in zip(scans, segmentations):
        # augmented_scan, augmented_segmentation = add_crop(scan, segmentation)
        # new_scans += augmented_scan
        # new_segmentations += augmented_segmentation
        # augmented_scan = add_contrast(scan)
        # new_scans.append(augmented_scan)
        # new_segmentations.append(segmentation)
        augmented_scan = add_blur(scan)
        new_scans.append(augmented_scan)
        new_segmentations.append(segmentation)
        augmented_scans, augmented_segmentations = add_distortion(scan, segmentation)
        new_scans.extend(augmented_scans)
        new_segmentations.extend(augmented_segmentations)
    return new_scans, new_segmentations

"""
Adds random cropping and center cropping to the scan/segmentation pair.
"""
def add_crop(scan, segmentation, IMGSIZE=(128,128), GENERATE=1):
    augmented_scans = [[] for _ in range(2*GENERATE)]
    augmented_segmentations = [[] for _ in range(2*GENERATE)]
    for i in range(GENERATE):
        ## random crop
        x_min = random.randint(0, IMGSIZE[0]-2)
        y_min = random.randint(0, IMGSIZE[1]-2)
        x_max = random.randint(IMGSIZE[0]-1, IMGSIZE[0])
        y_max = random.randint(IMGSIZE[1]-1, IMGSIZE[1])
        begin_x = int((IMGSIZE[0] - (x_max - x_min)) / 2)
        begin_y = int((IMGSIZE[1] - (y_max - y_min)) / 2)
        aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        for image, mask in zip(scan, segmentation):
            augmented = aug(image=image, mask=mask)
            new_mask = np.zeros(IMGSIZE)
            new_image = np.zeros(IMGSIZE)
            new_mask[begin_y:begin_y+y_max-y_min, begin_x:begin_x+x_max-x_min] = augmented['mask']
            new_image[begin_y:begin_y+y_max-y_min, begin_x:begin_x+x_max-x_min] = augmented['image']
            augmented_scans[i].append(new_image)
            augmented_segmentations[i].append(new_mask)
        ## random center
        h = IMGSIZE[0]//(i+2)
        w = IMGSIZE[1]//(i+2)
        aug = CenterCrop(p=1, height=h, width=w)
        for image, mask in zip(scan, segmentation):
            augmented = aug(image=image, mask=mask)
            new_mask = np.zeros(IMGSIZE)
            new_image = np.zeros(IMGSIZE)
            begin_x = (IMGSIZE[0]-h) // 2
            begin_y = (IMGSIZE[1]-w) // 2
            new_mask[begin_y:begin_y+h, begin_x:begin_x+w] = augmented['mask']
            new_image[begin_y:begin_y+h, begin_x:begin_x+w] = augmented['image']
            augmented_scans[i+1].append(new_image)
            augmented_segmentations[i+1].append(new_mask)
    return augmented_scans, augmented_segmentations

"""
Add random brightness and contrast to the scan.
"""
def add_contrast(scan):
    augmented_scan = []
    aug = RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2)
    for image in scan:
        augmented = aug(image=image)
        augmented_scan.append(augmented['image'])
    return augmented_scan

"""
Add Gaussian blur to the scan.
"""
def add_blur(scan):
    augmented_scan = []
    for image in scan:
        aug = GaussianBlur(p=1, blur_limit=(1, 3), sigma_limit=(0.1, 0.5))
        augmented = aug(image=image)
        augmented_scan.append(augmented['image'])
    return augmented_scan

"""
Apply optical distortion, grid distortion, and elsatic transform
independently to the scan/segmentation pair.
"""
def add_distortion(scan, segmentation, IMGSIZE=(128,128)):
    augmented_scans = [[], [], []]
    augmented_segmentations = [[], [], []]
    aug1 = OpticalDistortion(p=1)
    aug2 = GridDistortion(p=1)
    aug3 = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    for image, mask in zip(scan, segmentation):
        augmented = aug1(image=image, mask=mask)
        augmented_scans[0].append(augmented['image'])
        augmented_segmentations[0].append(augmented['mask'])
        augmented = aug2(image=image, mask=mask)
        augmented_scans[1].append(augmented['image'])
        augmented_segmentations[1].append(augmented['mask'])
        augmented = aug3(image=image, mask=mask)
        augmented_scans[2].append(augmented['image'])
        augmented_segmentations[2].append(augmented['mask'])
    return augmented_scans, augmented_segmentations
    