from omegaconf import DictConfig
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


DATA_CACHE_DIR = "/tmp/mnist"


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """

    running_loss = 0.0
    total = 0
    correct = 0

    # Do the training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data: torch.Tensor
        target: torch.Tensor
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

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


def training_loop(launch_config: DictConfig):

    # Load the PyTorch model
    model = launch_config.model
    model_module_str, model_class_str = model.model_class.rsplit('.', 1)
    model_module = importlib.import_module(model_module_str)
    model_class = getattr(model_module, model_class_str)
    model: nn.Module = model_class(**model.model_kwargs)

    model = torch.compile(model)
    
    # Get data loaders
    get_data_loaders_str = launch_config.data.get_data_loaders_function
    get_data_loaders_module_str, get_data_loaders_function_str = get_data_loaders_str.rsplit('.', 1)
    get_data_loaders_module = importlib.import_module(get_data_loaders_module_str)
    get_data_loaders = getattr(get_data_loaders_module, get_data_loaders_function_str)
    train_loader, val_loader = get_data_loaders(
        data_cache=DATA_CACHE_DIR,
        **launch_config.data.data_kwargs
    )

    # Get validation function
    validate_str = launch_config.validation.validate_function
    validate_module_str, validate_function_str = validate_str.rsplit('.', 1)
    validate_module = importlib.import_module(validate_module_str)
    validate = getattr(validate_module, validate_function_str)

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        print("No GPU available, using CPU")
        device = torch.device('cpu')

    print(f"Using device: {device}")
    model.to(device)

    learning_rate: float = launch_config.training.training_config.learning_rate
    weight_decay: float = launch_config.training.training_config.weight_decay
    epochs: int = launch_config.training.training_config.epochs
    log_interval: int = launch_config.training.training_config.log_interval

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

    start_epoch = 1
    best_val_accuracy = 0.0
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
        val_loss, val_acc = validate(model, val_loader)
        
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
            # TODO: Save checkpoint
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("="*60)


if __name__ == "__main__":
    exit(training_loop())
