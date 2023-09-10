import os
from sklearn.utils import shuffle


def data_preparation():
    covid_image_path = "Datasets/archive/COVID-19_Radiography_Dataset/COVID/images/"
    normal_image_path = "Datasets/archive/COVID-19_Radiography_Dataset/Normal/images/"

    # Paths to masks
    covid_mask_path = 'Datasets/archive/COVID-19_Radiography_Dataset/COVID/masks/'
    normal_mask_path = 'Datasets/archive/COVID-19_Radiography_Dataset/Normal/masks/'

    # All paths to images and masks
    all_image_paths = [[covid_image_path + file for file in os.listdir(covid_image_path)] +
                      [normal_image_path + file for file in os.listdir(normal_image_path)]]

    all_mask_paths = [[covid_mask_path + file for file in os.listdir(covid_mask_path)] +
                     [normal_mask_path + file for file in os.listdir(normal_mask_path)]]

    # Shuffle the arrays
    return shuffle(all_image_paths, all_mask_paths)