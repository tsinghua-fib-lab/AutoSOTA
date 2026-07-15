import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    def __init__(self, num_images=100, height=224, width=224, channels=3):
        """
        Initialize the dataset with the given parameters.
        :param num_images: Number of random images.
        :param height: Height of each image.
        :param width: Width of each image.
        :param channels: Number of channels (e.g., 3 for RGB).
        """
        self.num_images = num_images
        self.height = height
        self.width = width
        self.channels = channels
        # Pre-generate all images (optional, could generate on the fly in __getitem__)
        self.images = torch.randn(num_images, channels, height, width)
        self.labels = torch.randint(0, 2, (num_images,))

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return self.num_images

    def __getitem__(self, idx):
        """
        Retrieve an image by index.
        :param idx: Index of the image to retrieve.
        :return: Image tensor.
        """
        return self.images[idx], self.labels[idx]


def get_vit_fake_dataset():
    """
    Return a random training dataset and a random test dataset.
    """
    return RandomImageDataset(num_images=100), RandomImageDataset(num_images=10)
