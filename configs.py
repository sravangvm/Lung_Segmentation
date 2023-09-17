import torch
class TrainingConfig:
    def __init__(self):

        """
        This function contains different model configs
        """
        # Data paths
        self.train_image_paths = 'Datasets/train_image_paths.txt'
        self.train_mask_paths = 'Datasets/train_mask_paths.txt'
        self.val_image_paths = 'Datasets/val_image_paths.txt'
        self.val_mask_paths = 'Datasets/val_mask_paths.txt'

        # Model parameters
        self.batch_size = 16
        self.num_epochs = 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
