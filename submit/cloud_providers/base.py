
from abc import ABC, abstractmethod


class CloudProvider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the cloud provider is available.
        """
        ...

    @abstractmethod
    def get_instances(self) -> list[str]:
        """
        Get all instances associated with the cloud provider.
        """
        ...

    @abstractmethod
    def start_instances(self) -> list[str]:
        """
        Start instances and return the instance IDs.
        """
        ...

    @abstractmethod
    def stop_instances(self, instance_ids: list[str]):
        """
        Stop instances.
        """
        ...

    @abstractmethod
    def run_command_on_instances(self, instance_ids: list[str], command: str):
        """
        Run a command on instances.
        """
        ...

    @abstractmethod
    def copy_to_instances(self, instance_ids: list[str], local_path: str, instance_path: str):
        """
        Copy a file to instances.
        """
        ...

