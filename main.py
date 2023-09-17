from train_evaluate.train import train
from train_evaluate.evaluate import evaluate_model
from data_processing.data_handle import CustomDataset, datagen
from configs import TrainingConfig
from models.unet import UNet
from torch.utils.data import DataLoader

def run_main():
    config = TrainingConfig()

    # Create custom dataset instances for training and validation
    train_dataset = CustomDataset(config.train_image_paths, config.train_mask_paths)
    val_dataset = CustomDataset(config.val_image_paths, config.val_mask_paths)

    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize and train the model
    model = UNet()
    train(model, train_loader, val_loader, config.num_epochs, config.device)
    evaluate_model(model, train_loader, val_loader, config.num_epochs, config.device)

if __name__ == '__main__':
    run_main()