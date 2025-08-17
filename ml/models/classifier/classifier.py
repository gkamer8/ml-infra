from torch import nn

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        n_layers: int = 2,
        initial_filters: int = 16,
        kernel_size: int = 5,
        input_size: tuple = (28, 28),
        dropout_rate: float = 0.0
    ):
        """
        Configurable CNN Classifier
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            n_classes (int): Number of output classes
            n_layers (int): Number of convolutional layers
            initial_filters (int): Number of filters in first conv layer (doubles each layer)
            kernel_size (int): Size of convolutional kernels
            input_size (tuple): Input image dimensions (height, width)
            dropout_rate (float): Dropout probability (0 to disable)
        """
        super(Classifier, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        current_filters = initial_filters
        height, width = input_size
        
        # Build convolutional layers dynamically
        for _ in range(n_layers):
            conv_block = nn.Sequential(
                nn.Conv2d(
                    current_channels, 
                    current_filters, 
                    kernel_size=kernel_size, 
                    padding=kernel_size//2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.conv_layers.append(conv_block)
            
            # Update dimensions for next layer
            current_channels = current_filters
            current_filters *= 2
            height //= 2
            width //= 2
        
        # Calculate flattened features size for final layer
        self.flat_features = current_channels * height * width
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.flat_features, n_classes)
        )
    
    def forward(self, x):
        # Pass through all conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
