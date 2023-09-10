import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch


class ImageLoader:
    def __init__(self):
        self.IMAGE_SIZE = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    def open_images(self, paths):
        images = []

        for path in paths:
            image = Image.open(path)
            image = self.transform(image)
            image = image.mean(dim=0, keepdim=True) / 255.0  # Normalize to [0, 1]
            images.append(image)

        return torch.stack(images)