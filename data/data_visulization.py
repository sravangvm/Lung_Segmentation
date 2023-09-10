import matplotlib.pyplot as plt
from data.data_gen import ImageLoader  # Import the ImageLoader class from your data_gen module


image_loader = ImageLoader()  # Create an instance of the ImageLoader class


def visualize_images(image_paths):
    fig = plt.figure(figsize=(32, 32))
    for i in range(1, 5):
        fig.add_subplot(3, 9, i)
        plt.axis('off')
        plt.title('Images')
        plt.imshow(image_loader.open_images([image_paths[i - 1]][0]), cmap='gray', interpolation='none')
    plt.show()


def visualize_masks(mask_paths):
    fig = plt.figure(figsize=(32, 32))
    for i in range(1, 5):
        fig.add_subplot(3, 12, i)
        plt.axis('off')
        plt.title('Masks')
        plt.imshow(image_loader.open_images([mask_paths[i - 1]][0]), cmap='Spectral_r', alpha=0.3)
    plt.show()


def visualize_image_mask_pairs(image_paths, mask_paths):
    fig = plt.figure(figsize=(12, 12))
    c = 3
    r = 3
    for i in range(1, c * r + 1):
        fig.add_subplot(r, c, i)
        plt.axis('off')
        plt.imshow(image_loader.open_images([image_paths[i - 1]][0]), cmap='gray', interpolation='none')
        plt.imshow(image_loader.open_images([mask_paths[i - 1]][0]), cmap='Spectral_r', alpha=0.3)
    plt.show()