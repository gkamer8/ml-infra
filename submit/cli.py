"""Submit CLI - A command-line tool for submitting ML jobs."""

import argparse
import sys
from importlib.metadata import version
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme
from submit.launch import launch as imported_launch


from submit.cloud_providers.aggregate import (
    get_available_cloud_providers,
    CLOUD_PROVIDERS,
)


custom_theme = Theme({
    "header": "bold cyan",
    "success": "green",
    "error": "red",
    "muted": "dim",
})

console = Console(theme=custom_theme)


def print_header(text: str) -> None:
    console.print(Panel.fit(f"[header]{text}[/header]", border_style="cyan"))


def launch(config_path: str) -> None:
    """Launch a job with the specified configuration file.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    imported_launch(config_path)


def show_version() -> None:
    """Show the version of the submit CLI."""
    pkg_version = version("ml-infra")
    console.print(f"[success]{pkg_version}[/success]")


def resources() -> None:
    """Show available resources."""
    available_cloud_providers = get_available_cloud_providers()
    
    print_header("Cloud Providers")
    
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Instances")
    for cloud_provider in CLOUD_PROVIDERS:
        is_avail = cloud_provider in available_cloud_providers
        if is_avail:
            instances = cloud_provider.get_instances()
            status = f"[success]AVAILABLE[/success]"
        else:
            status = "[error]UNAVAILABLE[/error]"
        table.add_row(cloud_provider.name, status, str(len(instances)))
    console.print(table)


def info(launch_id: str) -> None:
    """Show information about a specific launch.
    
    Args:
        launch_id: The ID of the launch to get information about
    """
    console.print(f"[muted][Placeholder][/muted] Showing info for launch ID: [bold]{launch_id}[/bold]")


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
