import matplotlib.pyplot as plt
from data_handle import open_images


def visualize_images(image_paths):
    """
        This function is used to visualize images
    """

    fig = plt.figure(figsize=(100, 100))
    for i in range(1, 5):
        fig.add_subplot(18, 54, i)
        plt.axis('off')
        plt.title('Images')
        plt.imshow(open_images([image_paths[i - 1]]), cmap='gray', interpolation='none')
    plt.show()


def visualize_masks(mask_paths):
    """
        This function is used to visualize masks
    """
    fig = plt.figure(figsize=(100, 100))
    for i in range(1, 5):
        fig.add_subplot(18, 54, i)
        plt.axis('off')
        plt.title('Masks')
        plt.imshow(open_images([mask_paths[i - 1]]), cmap='Spectral_r', alpha=0.3)
    plt.show()


def visualize_image_mask_pairs(image_paths, mask_paths):
    """
        This function is used to visualize images paired with masks
    """
    fig = plt.figure(figsize=(12, 12))
    c = 3
    r = 3
    for i in range(1, c * r + 1):
        fig.add_subplot(r, c, i)
        plt.axis('off')
        plt.imshow(open_images([image_paths[i - 1]][0]), cmap='gray', interpolation='none')
        plt.imshow(open_images([mask_paths[i - 1]][0]), cmap='Spectral_r', alpha=0.3)
    plt.show()