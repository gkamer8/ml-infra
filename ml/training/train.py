import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import time
from datetime import datetime

# Add parent directory to path to import the classifier
sys.path.append(str(Path(__file__).parent.parent))
from models.classifier.classifier import Classifier


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for MNIST dataset.
    
    Args:
        batch_size: Batch size for training and validation
        data_dir: Directory to download/load MNIST data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        root=data_dir,
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
        epoch: Current epoch number
        log_interval: How often to log training progress
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Log progress
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (cuda/cpu)
    
    Returns:
        average_loss, accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            val_loss += criterion(output, target).item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_dir: str = "./checkpoints"
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'mnist_classifier_epoch_{epoch}_acc_{accuracy:.2f}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as 'best.pth' for easy access to best model
    best_path = os.path.join(checkpoint_dir, 'best.pth')
    torch.save(checkpoint, best_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, Optional[optim.Optimizer], int]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: The neural network model
        optimizer: Optimizer (optional)
        device: Device to load to
    
    Returns:
        model, optimizer, epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {checkpoint['loss']:.4f}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model, optimizer, epoch


def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    resume_from: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    log_interval: int = 100
):
    """
    Main training function for MNIST classifier.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for L2 regularization
        data_dir: Directory for MNIST data
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        device: Device to use (cuda/cpu/mps)
        seed: Random seed for reproducibility
        log_interval: How often to log training progress
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=2 if device.type != 'cpu' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    # Create model
    model = Classifier(
        in_channels=1,
        n_classes=10,
        n_layers=2,
        initial_filters=16,
        kernel_size=5,
        input_size=(28, 28),
        dropout_rate=0.2
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(
            resume_from, model, optimizer, device
        )
        start_epoch += 1
    
    # Training loop
    best_val_accuracy = 0.0
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Training
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, log_interval
        )
        
        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, checkpoint_dir
            )
            print(f"  ✓ New best model saved: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("="*60)
    
    return model, best_val_accuracy


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Train MNIST Classifier')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay for L2 regularization (default: 0.0001)')
    
    # Data and checkpoint directories
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory for MNIST data (default: ./data)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    
    # Device and logging
    parser.add_argument('--device', type=str, default=None,
                        help='device to use (cuda/cpu/mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    
    # Run training
    model, best_accuracy = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval
    )
    
    print(f"\nFinal best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()