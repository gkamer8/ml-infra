import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def validate(
    model: nn.Module,
    val_loader: DataLoader
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
    
    Returns:
        average_loss, accuracy
    """
    device = next(model.parameters()).device
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    
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
