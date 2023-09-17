import os
from sklearn.utils import shuffle


def data_preparation():
    """
        It handles the data division into train, and val files

        Returns:
          list:train_images_paths, train_masks_paths, val_images_paths, val_masks_paths
    """
    covid_image_path = "Datasets/archive/COVID-19_Radiography_Dataset/COVID/images/"
    normal_image_path = "Datasets/archive/COVID-19_Radiography_Dataset/Normal/images/"

    # Paths to masks
    covid_mask_path = 'Datasets/archive/COVID-19_Radiography_Dataset/COVID/masks/'
    normal_mask_path = 'Datasets/archive/COVID-19_Radiography_Dataset/Normal/masks/'

    # All paths to images and masks
    all_image_paths = [
        os.path.join(covid_image_path, file) for file in os.listdir(covid_image_path)
    ] + [
        os.path.join(normal_image_path, file) for file in os.listdir(normal_image_path)
    ]

    all_mask_paths = [
        os.path.join(covid_mask_path, file) for file in os.listdir(covid_mask_path)
    ] + [
        os.path.join(normal_mask_path, file) for file in os.listdir(normal_mask_path)
    ]

    all_image_paths, all_mask_paths = shuffle(all_image_paths, all_mask_paths)

    # Step 3: Split the data into training and validation sets
    x=int(len(all_image_paths)*0.7)
    train_image_paths = all_image_paths[:x]
    train_mask_paths = all_mask_paths[:x]
    val_image_paths = all_image_paths[x:]
    val_mask_paths = all_mask_paths[x:]

    return train_image_paths,train_mask_paths,val_image_paths,val_mask_paths

    # train_image_paths = [str(path) for path in train_image_paths]
    # train_mask_paths = [str(path) for path in train_mask_paths]
    # val_image_paths = [str(path) for path in val_image_paths]
    # val_mask_paths = [str(path) for path in val_mask_paths]

    # dataset_folder = "Datasets"
    # os.makedirs(dataset_folder, exist_ok=True)

    # # Save the image and mask paths to text files
    # with open(os.path.join(dataset_folder, "train_image_paths.txt"), "w") as image_file:
    #     for path in train_image_paths:
    #         path = path.strip()  # Remove trailing newline
    #         image_file.write(path)
    #         image_file.write("\n")

    # with open(os.path.join(dataset_folder, "train_mask_paths.txt"), "w") as image_file:
    #     for path in train_mask_paths:
    #         path = path.strip()  # Remove trailing newline
    #         image_file.write(path)
    #         image_file.write("\n")

    # with open(os.path.join(dataset_folder, "val_image_paths.txt"), "w") as image_file:
    #     for path in val_image_paths:
    #         path = path.strip()  # Remove trailing newline
    #         image_file.write(path)
    #         image_file.write("\n")

    # with open(os.path.join(dataset_folder, "val_mask_paths.txt"), "w") as image_file:
    #     for path in val_mask_paths:
    #         path = path.strip()  # Remove trailing newline
    #         image_file.write(path)
    #         image_file.write("\n")
