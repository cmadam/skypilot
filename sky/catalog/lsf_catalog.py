"""LSF catalog for SkyPilot.

LSF does not have a pre-generated hardware catalog. Instance types are
virtual and constructed dynamically from resource requests. Pricing is
read from ~/.sky/config.yaml (on-prem clusters are typically free).
"""
from typing import Dict, List, Optional, Tuple

from sky import sky_logging
from sky import skypilot_config
from sky.provision.lsf import utils as lsf_utils
from sky.utils import resources_utils

logger = sky_logging.init_logger(__name__)

_DEFAULT_NUM_VCPUS = 2
_DEFAULT_MEMORY_PER_CPU = 4  # GB per CPU


def instance_type_exists(instance_type: str) -> bool:
    """Check if the given instance type is valid."""
    return lsf_utils.LsfInstanceType.is_valid_instance_type(instance_type)


def get_default_instance_type(
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    disk_tier: Optional[resources_utils.DiskTier] = None,
    local_disk: Optional[int] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
) -> Optional[str]:
    """Get the default instance type for given resource constraints."""
    del disk_tier, local_disk  # Not applicable for LSF

    # Determine CPU count
    if cpus is not None:
        cpu_count = float(cpus.rstrip('+'))
    else:
        cpu_count = _DEFAULT_NUM_VCPUS

    # Determine memory
    if memory is not None:
        mem_gb = float(memory.rstrip('+'))
    else:
        mem_gb = cpu_count * _DEFAULT_MEMORY_PER_CPU

    inst = lsf_utils.LsfInstanceType.from_resources(cpu_count, mem_gb)

    # If a region (cluster) is specified, validate the instance fits
    if region is not None:
        fits, _ = lsf_utils.check_instance_fits(region, inst.name, zone)
        if not fits:
            return None

    return inst.name


def get_hourly_cost(
    instance_type: str,
    use_spot: bool,
    region: Optional[str] = None,
    zone: Optional[str] = None,
) -> float:
    """Get the hourly cost for an instance type.

    On-prem LSF clusters typically have no direct hourly cost.
    Pricing can be configured in ~/.sky/config.yaml for cost estimation.
    """
    del use_spot  # No spot on LSF

    pricing = _get_pricing(region, zone)
    if pricing:
        # Support per-GPU or per-instance pricing from config
        inst = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
        gpu_price = pricing.get('gpu_hourly', 0.0)
        cpu_price = pricing.get('cpu_hourly', 0.0)
        cost = (inst.cpus * cpu_price +
                (inst.accelerator_count or 0) * gpu_price)
        return cost

    return 0.0


def _get_pricing(region: Optional[str],
                 zone: Optional[str]) -> Dict[str, float]:
    """Get pricing configuration from sky config.

    Merges: global < cluster-level < queue-level.
    """
    global_pricing = skypilot_config.get_nested(
        ('lsf', 'pricing'), {})

    if region is None:
        return global_pricing

    cluster_pricing = skypilot_config.get_nested(
        ('lsf', 'cluster_configs', region, 'pricing'), {})

    merged = {}
    merged.update(global_pricing)
    merged.update(cluster_pricing)

    if zone is not None:
        queue_pricing = skypilot_config.get_nested(
            ('lsf', 'cluster_configs', region,
             'queue_configs', zone, 'pricing'), {})
        merged.update(queue_pricing)

    return merged


def validate_region_zone(
    region_name: Optional[str],
    zone_name: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    return (region_name, zone_name)


def list_accelerators(
    gpus_only: bool = True,
    name_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    quantity_filter: Optional[int] = None,
    case_sensitive: bool = True,
) -> Dict[str, List[Dict]]:
    acc_info, _, _ = list_accelerators_realtime(
        gpus_only=gpus_only,
        name_filter=name_filter,
        region_filter=region_filter,
        quantity_filter=quantity_filter,
        case_sensitive=case_sensitive,
    )
    return acc_info


def list_accelerators_realtime(
    gpus_only: bool = True,
    name_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    quantity_filter: Optional[int] = None,
    case_sensitive: bool = True,
) -> Tuple[Dict[str, List[Dict]], Dict[str, int], Dict[str, int]]:
    """Query LSF clusters for real-time accelerator availability.

    Returns:
        Tuple of (accelerator_info, total_count, available_count).
    """
    clusters = lsf_utils.get_all_lsf_cluster_names()
    if region_filter:
        clusters = [c for c in clusters if c == region_filter]

    acc_info: Dict[str, List[Dict]] = {}
    total_counts: Dict[str, int] = {}
    available_counts: Dict[str, int] = {}

    for cluster in clusters:
        try:
            nodes = lsf_utils.get_lsf_nodes_info(cluster)
        except Exception:  # pylint: disable=broad-except
            continue

        for node in nodes:
            if node.gpus <= 0:
                continue

            gpu_name = node.gpu_model or 'GPU'
            if name_filter:
                if case_sensitive:
                    if name_filter not in gpu_name:
                        continue
                else:
                    if name_filter.lower() not in gpu_name.lower():
                        continue

            if quantity_filter and node.gpus < quantity_filter:
                continue

            key = gpu_name
            if key not in acc_info:
                acc_info[key] = []
                total_counts[key] = 0
                available_counts[key] = 0

            acc_info[key].append({
                'instance_type': lsf_utils.LsfInstanceType.from_resources(
                    node.cpus, node.memory_gb, node.gpus, gpu_name).name,
                'region': cluster,
                'gpu_count': node.gpus,
                'cpu_count': node.cpus,
                'memory': node.memory_gb,
            })
            total_counts[key] += node.gpus
            if node.status == 'ok':
                available_counts[key] += node.gpus

    return acc_info, total_counts, available_counts
