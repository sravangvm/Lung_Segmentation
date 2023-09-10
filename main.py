from data import dataset, data_visulization
from my_models import unet
from train_evaluate import train

if __name__ == "__main__":
    # Define the paths and configurations

    # Step 1: Data Preprocessing
    all_image_paths, all_mask_paths = dataset.data_preparation()
    print(all_image_paths[0])

    # Step 2: Data Visualization (Optional)
    data_visulization.visualize_images(all_image_paths)
    data_visulization.visualize_masks(all_mask_paths)
    data_visulization.visualize_image_mask_pairs(all_image_paths,all_mask_paths)

    # Step 3: Split the data into training and validation sets
    train_image_paths = all_image_paths[:13000]
    train_mask_paths = all_mask_paths[:13000]
    val_image_paths = all_image_paths[13000:]
    val_mask_paths = all_mask_paths[13000:]

    # Step 4: Define and Train the Model
    model = unet.Unet()

    train.train(model,train_image_paths,train_mask_paths)


