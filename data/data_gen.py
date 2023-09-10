from PIL import Image
import torch
import torchvision.transforms as transforms


class ImageLoader:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def open_images(self, paths):
        images = []

        for path in paths:
            image = Image.open(path)
            image = self.preprocess_image(image)
            images.append(image)

        return torch.stack(images)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.mean(dim=0, keepdim=True) / 255.0
        return image
