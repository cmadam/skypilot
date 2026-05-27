"""LSF provisioner config (bootstrap step)."""
from sky.provision import common


def bootstrap_instances(
    region: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionConfig:
    """No bootstrap needed for LSF — config is used as-is."""
    del region, cluster_name_on_cloud
    return config
