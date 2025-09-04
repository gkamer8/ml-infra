"""Submit CLI - A command-line tool for submitting ML jobs."""

import argparse
import sys
import importlib
import inspect
from pathlib import Path
from importlib.metadata import version
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf


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
    training_function_str: str = launch_config.workload.training_fn
    
    # Split the module path and function name
    module_path, function_name = training_function_str.rsplit('.', 1)
    
    # Dynamically import the module
    training_module = importlib.import_module(module_path)
    
    # Get the function from the module
    training_function = getattr(training_module, function_name)
    
    # Call the training function with the launch config
    training_function()


def show_version() -> None:
    """Show the version of the submit CLI."""
    pkg_version = version("ml-infra")
    print(pkg_version)


def resources() -> None:
    """Show available resources."""
    print("[Placeholder] Showing available resources")


def info(launch_id: str) -> None:
    """Show information about a specific launch.
    
    Args:
        launch_id: The ID of the launch to get information about
    """
    print(f"[Placeholder] Showing info for launch ID: {launch_id}")


def main():
    """Main entry point for the submit CLI."""
    parser = argparse.ArgumentParser(
        prog='submit',
        description='Submit ML jobs locally or to remote GPUs',
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 'launch' command
    launch_parser = subparsers.add_parser('launch', help='Launch a job with a configuration file')
    launch_parser.add_argument('config_path', help='Path to the configuration YAML file')
    
    # 'version' command
    subparsers.add_parser('version', help='Show the version of the submit CLI')
    
    # 'resources' command
    subparsers.add_parser('resources', help='Show available resources')
    
    # 'info' command
    info_parser = subparsers.add_parser('info', help='Get information about a launch')
    info_parser.add_argument('launch_id', help='The ID of the launch')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'launch':
        launch(args.config_path)
    elif args.command == 'version':
        show_version()
    elif args.command == 'resources':
        resources()
    elif args.command == 'info':
        info(args.launch_id)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
