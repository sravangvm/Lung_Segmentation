# train_evaluate/train_evaluate.py
import torch
from utils import config
from data import data_gen


def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[1,2,3])
    union = torch.sum(y_true, dim=[1,2,3]) + torch.sum(y_pred, dim=[1,2,3]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou.item()


def train(model, train_image_paths, train_mask_paths):
    # Initialize training configurations
    training_config = config.TrainingConfig()

    # Create a DataLoader from the custom dataset
    train_dataloader = data_gen.datagen(train_image_paths,train_mask_paths);

    # Define your loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Define optimizer with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(training_config.num_epochs):
        running_loss = 0.0
        running_iou = 0.0
        for inputs, labels in train_dataloader:
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