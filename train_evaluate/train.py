import torch
from configs import config
import torch.nn as nn
from data_processing.data_handle import datagen  # Import the CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader  # Import DataLoader


def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[1, 2, 3])
    union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou.item()


def train(model, train_image_paths, train_mask_paths):

    """
        This function is used to train the model
    """
    # Initialize training configurations
    training_config = config.TrainingConfig()

    IMAGE_SIZE = 256
    # Define your custom transformations (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Create instances of the dataset with the specified transformations
    # train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=transform)

    # Create a DataLoader for training data
    train_dataloader = datagen(train_image_paths,train_mask_paths)

    # Define your loss function
    loss_function = nn.BCELoss()

    # Define optimizer with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(training_config.num_epochs):
        running_loss = 0.0
        running_iou = 0.0
        for inputs, labels in train_dataloader:
            print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate IoU for each batch and accumulate
            batch_iou = iou_coef(labels, outputs)
            running_iou += batch_iou

        # Calculate and print the average loss and IoU for this epoch
        avg_loss = running_loss / len(train_dataloader)
        avg_iou = running_iou / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{training_config.num_epochs}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "your_trained_model.pth")
