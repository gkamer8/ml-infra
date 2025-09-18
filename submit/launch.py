import importlib
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig


def launch(config_path: str) -> None:
    """Launch a job with the specified configuration file.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Parse the config path to get directory and config name
    config_path = Path(config_path)
    config_dir = config_path.parent.absolute()
    config_name = config_path.name  # stem also works
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration
        launch_config: DictConfig = compose(config_name=config_name)
    #   
    # Run the training loop
    #

    # Get the training function as a string
    training_function_str: str = launch_config.training.training_function
    
    # Split the module path and function name
    module_path, function_name = training_function_str.rsplit('.', 1)
    
    # Dynamically import the module
    training_module = importlib.import_module(module_path)
    
    # Get the function from the module
    training_function = getattr(training_module, function_name)
    
    # Call the training function with the launch config
    training_function(launch_config)
