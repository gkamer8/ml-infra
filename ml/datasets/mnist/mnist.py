from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms


def get_data_loaders(
    data_cache: str,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    random_flip: bool = True,

) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for MNIST dataset.
    
    Args:
        data_cache: Directory to download/load MNIST data
        batch_size: Batch size for training and validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    # Define transforms

    transformations = [
        transforms.ToTensor()  # Does /255 scaling
    ]

    if random_flip:
        transformations.append(transforms.RandomVerticalFlip())
        transformations.append(transforms.RandomHorizontalFlip())

    transform = transforms.Compose(transformations)
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=data_cache,
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        root=data_cache,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
