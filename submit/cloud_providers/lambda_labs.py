from submit.cloud_providers.base import CloudProvider
import os
import subprocess
from typing import Any, Dict, List, Optional
import requests


class LambdaLabs(CloudProvider):

    def __init__(self):
        super().__init__("Lambda Labs")
        # Support both typical env var names
        self.api_key = os.getenv("LAMBDA_API_KEY") or os.getenv("LAMBDALABS_API_KEY")
        self.base_url = "https://cloud.lambdalabs.com/api/v1"

        # Optional configuration for launching/ssh
        self.default_region = os.getenv("LAMBDA_REGION")
        self.default_instance_type = os.getenv("LAMBDA_INSTANCE_TYPE")
        self.default_ssh_key_name = os.getenv("LAMBDA_SSH_KEY_NAME")
        self.default_quantity = int(os.getenv("LAMBDA_QUANTITY", "1"))
        self.default_instance_name = os.getenv("LAMBDA_INSTANCE_NAME", "ml-infra-instance")
        self.default_disk_size_gb: Optional[int] = (
            int(os.getenv("LAMBDA_DISK_SIZE_GB")) if os.getenv("LAMBDA_DISK_SIZE_GB") else None
        )

        # SSH config
        self.ssh_user = os.getenv("LAMBDA_SSH_USER", "ubuntu")
        # Prefer explicit path, fallback to common default
        self.ssh_private_key = os.path.expanduser(
            os.getenv("LAMBDA_SSH_PRIVATE_KEY", os.getenv("SSH_PRIVATE_KEY", "~/.ssh/id_rsa"))
        )

    # ----------------------------- API helpers ----------------------------- #
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        if not self.api_key:
            raise RuntimeError("LAMBDA_API_KEY/LAMBDALABS_API_KEY is not set")
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            response = e.response
            details = response.text if response is not None else ""
            code = response.status_code if response is not None else "?"
            reason = response.reason if response is not None else "HTTPError"
            raise RuntimeError(f"Lambda API error {code} {reason}: {details}") from None
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to reach Lambda API: {e}") from None

        if not resp.text:
            return None
        try:
            parsed = resp.json()
        except ValueError:
            return None
        return parsed.get("data", parsed)

    def _get_instances_full(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/instances")
        # Accept either a list or an object containing a list
        if isinstance(data, dict):
            instances = data.get("instances") or data.get("data") or []
        else:
            instances = data
        return list(instances or [])

    def _extract_instance_ids(self, data: Any) -> List[str]:
        if isinstance(data, dict):
            if "instance_ids" in data:
                return list(data["instance_ids"])
            if "instances" in data and isinstance(data["instances"], list):
                return [i.get("id") for i in data["instances"] if i.get("id")]
        if isinstance(data, list):
            return [i.get("id") for i in data if isinstance(i, dict) and i.get("id")]
        return []

    def _public_ip_from_record(self, inst: Dict[str, Any]) -> Optional[str]:
        # Try common keys observed in Lambda responses
        for key in ("ip", "public_ip", "public_ipv4", "public_ip_address"):
            if key in inst and inst[key]:
                return inst[key]
        # Sometimes nested under network
        network = inst.get("network") if isinstance(inst, dict) else None
        if isinstance(network, dict):
            for key in ("public_ip", "public_ipv4", "ip"):
                if key in network and network[key]:
                    return network[key]
        return None

    def _get_public_ip(self, instance_id: str) -> Optional[str]:
        for inst in self._get_instances_full():
            if inst.get("id") == instance_id:
                return self._public_ip_from_record(inst)
        return None

    # ----------------------------- Interface ------------------------------ #
    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_instances(self) -> list[str]:
        instances = self._get_instances_full()
        ids: List[str] = []
        for inst in instances:
            status = (inst.get("status") or "").lower()
            if status not in {"terminated", "terminating", "stopped"}:
                if inst.get("id"):
                    ids.append(inst["id"])
        return ids
    
    def start_instances(self) -> list[str]:
        if not (self.default_region and self.default_instance_type and self.default_ssh_key_name):
            raise RuntimeError(
                "Starting instances requires LAMBDA_REGION, LAMBDA_INSTANCE_TYPE, and LAMBDA_SSH_KEY_NAME env vars"
            )
        payload: Dict[str, Any] = {
            "region_name": self.default_region,
            "instance_type_name": self.default_instance_type,
            "ssh_key_names": [self.default_ssh_key_name],
            "quantity": self.default_quantity,
            "name": self.default_instance_name,
        }
        if self.default_disk_size_gb:
            payload["disk_size_gb"] = self.default_disk_size_gb

        data = self._request("POST", "/instances", payload)
        instance_ids = self._extract_instance_ids(data)
        return instance_ids

    def stop_instances(self, instance_ids: list[str]):
        if not instance_ids:
            return
        self._request("POST", "/instances/terminate", {"instance_ids": instance_ids})

    def run_command_on_instances(self, instance_ids: list[str], command: str):
        if not instance_ids:
            return
        for instance_id in instance_ids:
            ip = self._get_public_ip(instance_id)
            if not ip:
                raise RuntimeError(f"No public IP found for instance {instance_id}")
            ssh_cmd = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                self.ssh_private_key,
                f"{self.ssh_user}@{ip}",
                command,
            ]
            subprocess.run(ssh_cmd, check=True)

    def copy_to_instances(self, instance_ids: list[str], local_path: str, instance_path: str):
        if not instance_ids:
            return
        recursive_flag = ["-r"] if os.path.isdir(local_path) else []
        for instance_id in instance_ids:
            ip = self._get_public_ip(instance_id)
            if not ip:
                raise RuntimeError(f"No public IP found for instance {instance_id}")
            scp_cmd = [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                self.ssh_private_key,
                *recursive_flag,
                local_path,
                f"{self.ssh_user}@{ip}:{instance_path}",
            ]
            subprocess.run(scp_cmd, check=True)
