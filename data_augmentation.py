import numpy as np
from torchvision import transforms


def expand_to_3_channels(image):
    if image.size(0) != 3:
        return image.repeat(3, 1, 1)
    return image


def get_data_transform_2D_vit(patch_size):
    img_trans = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
        ]),
        'val': transforms.Compose([
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
        ]),
        'test': transforms.Compose([
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
        ]),
    }
    return img_trans


