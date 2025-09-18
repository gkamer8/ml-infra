from submit.cloud_providers.base import CloudProvider
from submit.cloud_providers.lambda_labs import LambdaLabs


# If you want to add a new cloud provider, add it here.
CLOUD_PROVIDERS = [
    LambdaLabs(),
]

def get_available_cloud_providers() -> list[CloudProvider]:
    available = []
    for cloud_provider in CLOUD_PROVIDERS:
        if cloud_provider.is_available():
            available.append(cloud_provider)
    return available
