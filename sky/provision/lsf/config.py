"""LSF provisioner config (bootstrap step)."""
from typing import Any, Dict


def bootstrap_instances(
    region: str,
    cluster_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """No bootstrap needed for LSF — config is used as-is."""
    del region, cluster_name
    return config
