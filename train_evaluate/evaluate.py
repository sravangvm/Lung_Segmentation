import torch
import numpy as np


def evaluate_model(model, dataloader, device):

    """
        This function is used to evaluate the model
    """
    model.eval()  # Set the model to evaluation mode
    iou_scores = []
    accuracy_scores = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass to get predictions
            outputs = model(images)

            # Convert predictions to binary masks
            predictions = (outputs > 0.5).float()

            # Calculate Intersection over Union (IoU) for each batch
            intersection = torch.sum(predictions * masks)
            union = torch.sum(predictions) + torch.sum(masks) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.append(iou.item())

            # Calculate pixel-wise accuracy
            correct_pixels = torch.sum(predictions == masks)
            total_pixels = masks.numel()
            accuracy = (correct_pixels / total_pixels).item()
            accuracy_scores.append(accuracy)

    # Calculate the mean IoU and accuracy across all batches
    mean_iou = np.mean(iou_scores)
    mean_accuracy = np.mean(accuracy_scores)

    return mean_iou, mean_accuracy