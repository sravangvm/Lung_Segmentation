import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
        Creates a Custom dataset to train in batches
        Returns:
            Tensors: image,mask
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize((256, 256)),  # Resize the image
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = self.load_paths(image_paths)
        self.mask_paths = self.load_paths(mask_paths)
        self.transform = transform

    def load_paths(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        paths = []

        for line in lines:
            line = line.strip()
            paths.append(line)

        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)

        # Convert mask to binary
        mask = mask.convert('L')  # Convert to grayscale
        mask = np.array(mask)  # Convert to a NumPy array
        mask = (mask > 0.5).astype(np.float32)  # Threshold and convert to float32
        mask = torch.from_numpy(mask)  # Convert to a PyTorch tensor

        # Convert image and mask to PyTorch tensors
        image = torch.Tensor(np.array(image))
        mask = torch.Tensor(np.array(mask))

        # Normalize the image to the range [0, 1]
        image = image / 255.0

        return image, mask

def datagen(image_paths, mask_paths, batch_size=16):
    """
        Creates a data generator
        Returns:
            List: image,mask
    """

    for x in range(0, len(image_paths), batch_size):
        images = open_images(image_paths[x:x+batch_size])
        masks = open_images(mask_paths[x:x+batch_size])
        yield images, masks

def open_images(paths):
    """
        Converts Images to tensors to feed to Model
        Returns:
            List: images
    """
    images = []
    for path in paths:
        image = Image.open(path)
        image = image.resize((256, 256))
        image = np.mean(np.array(image), axis=-1) / 255.0
        images.append(image)

    images = torch.tensor(images).unsqueeze(1).float()
    return images
