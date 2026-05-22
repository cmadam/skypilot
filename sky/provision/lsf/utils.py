"""LSF utilities for SkyPilot."""
import json
import math
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from paramiko.config import SSHConfig

from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import lsf
from sky.utils import annotations
from sky.utils import common_utils
from sky.utils.db import kv_cache

logger = sky_logging.init_logger(__name__)

DEFAULT_LSF_PATH = '~/.lsf/config'

LSF_MARKER_FILE = '.sky_lsf_cluster'
LSF_CONTAINER_MARKER_FILE = '.sky_lsf_container'

_LSF_NODES_INFO_CACHE_TTL = 30 * 60
_LSF_ENROOT_CHECK_CACHE_TTL = 24 * 60 * 60
_LSF_FUSE_CHECK_CACHE_TTL = 24 * 60 * 60


class LsfInstanceType:
    """Class to represent the "Instance Type" in an LSF cluster.

    Since LSF does not have a notion of instances, we generate
    virtual instance types that represent the resources requested by a
    worker node.

    The name format is "{n}CPU--{k}GB" where n is the number of vCPUs and
    k is the amount of memory in GB. Accelerators can be specified by
    appending "--{type}:{a}" where type is the accelerator type and a
    is the number of accelerators.

    Examples:
        - 4CPU--16GB
        - 4CPU--16GB--H100:8
        - 96CPU--512GB--H100:8
    """

    def __init__(self,
                 cpus: float,
                 memory: float,
                 accelerator_count: Optional[int] = None,
                 accelerator_type: Optional[str] = None):
        self.cpus = cpus
        self.memory = memory
        self.accelerator_count = accelerator_count
        self.accelerator_type = accelerator_type

    @property
    def name(self) -> str:
        assert self.cpus is not None
        assert self.memory is not None
        name = (f'{common_utils.format_float(self.cpus)}CPU--'
                f'{common_utils.format_float(self.memory)}GB')
        if self.accelerator_count is not None:
            assert self.accelerator_type is not None, self.accelerator_count
            acc_name = self.accelerator_type.replace(' ', '_')
            name += f'--{acc_name}:{self.accelerator_count}'
        return name

    @staticmethod
    def is_valid_instance_type(name: str) -> bool:
        pattern = re.compile(
            r'^(\d+(\.\d+)?CPU--\d+(\.\d+)?GB)(--[\w\d-]+:\d+)?$')
        return bool(pattern.match(name))

    @classmethod
    def _parse_instance_type(
            cls,
            name: str) -> Tuple[float, float, Optional[int], Optional[str]]:
        pattern = re.compile(
            r'^(?P<cpus>\d+(\.\d+)?)CPU--(?P<memory>\d+(\.\d+)?)GB'
            r'(?:--(?P<accelerator_type>[\w\d-]+):(?P<accelerator_count>\d+))?$'
        )
        match = pattern.match(name)
        if match is not None:
            cpus = float(match.group('cpus'))
            memory = float(match.group('memory'))
            accelerator_count = match.group('accelerator_count')
            accelerator_type = match.group('accelerator_type')
            if accelerator_count is not None:
                accelerator_count = int(accelerator_count)
                accelerator_type = str(accelerator_type).replace(' ', '_')
            else:
                accelerator_count = None
                accelerator_type = None
            return cpus, memory, accelerator_count, accelerator_type
        else:
            raise ValueError(f'Invalid instance name: {name}')

    @classmethod
    def from_instance_type(cls, name: str) -> 'LsfInstanceType':
        if not cls.is_valid_instance_type(name):
            raise ValueError(f'Invalid instance name: {name}')
        cpus, memory, accelerator_count, accelerator_type = \
            cls._parse_instance_type(name)
        return cls(cpus=cpus,
                   memory=memory,
                   accelerator_count=accelerator_count,
                   accelerator_type=accelerator_type)

    @classmethod
    def from_resources(cls,
                       cpus: float,
                       memory: float,
                       accelerator_count: Union[float, int] = 0,
                       accelerator_type: str = '') -> 'LsfInstanceType':
        accelerator_count = math.ceil(accelerator_count)
        if accelerator_count > 0:
            return cls(cpus=cpus,
                       memory=memory,
                       accelerator_count=accelerator_count,
                       accelerator_type=accelerator_type)
        return cls(cpus=cpus, memory=memory)

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'LsfInstanceType(cpus={self.cpus!r}, '
                f'memory={self.memory!r}, '
                f'accelerator_count={self.accelerator_count!r}, '
                f'accelerator_type={self.accelerator_type!r})')


def instance_id(job_id: str, node: str) -> str:
    """Generates the SkyPilot-defined instance ID for LSF."""
    return f'job{job_id}-{node}'


def get_lsf_ssh_config() -> SSHConfig:
    """Get the LSF SSH config."""
    lsf_config_path = os.path.expanduser(DEFAULT_LSF_PATH)
    return SSHConfig.from_path(lsf_config_path)


def get_identity_file(ssh_config_dict: Dict[str, Any]) -> Optional[str]:
    """Get the first identity file from SSH config."""
    identity_files = ssh_config_dict.get('identityfile')
    if identity_files:
        return identity_files[0]
    return None


def get_identities_only(ssh_config_dict: Dict[str, Any]) -> bool:
    """Check if IdentitiesOnly is set to yes in SSH config."""
    identities_only = ssh_config_dict.get('identitiesonly', '')
    return identities_only.lower() == 'yes'


def get_lsf_cluster_from_config(provider_config: Dict[str, Any]) -> str:
    """Return the LSF cluster from the provider config."""
    cluster = provider_config.get('cluster')
    if cluster is None:
        raise ValueError('LSF cluster not specified in provider config.')
    return cluster


def get_queue_from_config(provider_config: Dict[str, Any]) -> Optional[str]:
    """Return the queue from the provider config.

    The concept of queue maps to a cloud zone.
    """
    return provider_config.get('queue')


def _create_lsf_client(cluster: str) -> lsf.LsfClient:
    """Create an LsfClient for the given cluster name."""
    ssh_config = get_lsf_ssh_config()
    ssh_config_dict = ssh_config.lookup(cluster)
    return lsf.LsfClient(
        ssh_config_dict['hostname'],
        int(ssh_config_dict.get('port', 22)),
        ssh_config_dict['user'],
        get_identity_file(ssh_config_dict),
        ssh_proxy_command=ssh_config_dict.get('proxycommand', None),
        ssh_proxy_jump=ssh_config_dict.get('proxyjump', None),
        identities_only=get_identities_only(ssh_config_dict),
    )


@annotations.lru_cache(scope='request')
def get_lsf_nodes_info(cluster: str) -> List[lsf.NodeInfo]:
    """Get node info for an LSF cluster, with caching."""
    cache_key = f'lsf:nodes_info:{cluster}'
    cached = kv_cache.get_cache_entry(cache_key)
    if cached is not None:
        logger.debug(f'LSF nodes info found in cache ({cache_key})')
        return [lsf.NodeInfo(**item) for item in json.loads(cached)]

    client = _create_lsf_client(cluster)
    nodes_info = client.info_nodes()

    try:
        kv_cache.add_or_update_cache_entry(
            cache_key, json.dumps([n._asdict() for n in nodes_info]),
            time.time() + _LSF_NODES_INFO_CACHE_TTL)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug(f'Failed to cache LSF nodes info for {cluster}: '
                     f'{common_utils.format_exception(e)}')

    return nodes_info


def _check_cluster_feature(
    cluster: str,
    feature_name: str,
    check_fn: Callable[[lsf.LsfClient], bool],
    cache_ttl: int,
) -> bool:
    """Check if a feature is available on an LSF cluster, with caching."""
    cache_key = f'lsf:{feature_name}_enabled:{cluster}'
    cached = kv_cache.get_cache_entry(cache_key)
    if cached is not None:
        logger.debug(f'LSF {feature_name} check found in cache ({cache_key})')
        return cached == 'true'

    client = _create_lsf_client(cluster)
    enabled = check_fn(client)

    try:
        kv_cache.add_or_update_cache_entry(cache_key,
                                           'true' if enabled else 'false',
                                           time.time() + cache_ttl)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug(f'Failed to cache LSF {feature_name} check for '
                     f'{cluster}: {common_utils.format_exception(e)}')

    return enabled


def check_enroot_enabled(cluster: str) -> bool:
    """Check if enroot is available on an LSF cluster."""
    return _check_cluster_feature(cluster, 'enroot',
                                  lambda c: c.check_enroot_available(),
                                  _LSF_ENROOT_CHECK_CACHE_TTL)


def check_fuse_enabled(cluster: str) -> bool:
    """Check if FUSE is available on an LSF cluster."""
    return _check_cluster_feature(cluster, 'fuse',
                                  lambda c: c.check_fuse_enabled(),
                                  _LSF_FUSE_CHECK_CACHE_TTL)


def get_all_lsf_cluster_names() -> List[str]:
    """Get all LSF cluster names from ~/.lsf/config."""
    try:
        ssh_config = get_lsf_ssh_config()
    except FileNotFoundError:
        return []
    except Exception as e:
        raise ValueError(
            f'Failed to load SSH configuration from {DEFAULT_LSF_PATH}: '
            f'{common_utils.format_exception(e)}') from e

    cluster_names = []
    for cluster in ssh_config.get_hostnames():
        if cluster == '*':
            continue
        cluster_names.append(cluster)
    return cluster_names


@annotations.lru_cache(scope='request')
def get_cluster_default_queue(cluster_name: str) -> Optional[str]:
    """Get the default queue for an LSF cluster."""
    try:
        client = _create_lsf_client(cluster_name)
    except Exception as e:
        raise ValueError(
            f'Failed to connect to LSF cluster {cluster_name}: '
            f'{common_utils.format_exception(e)}') from e
    return client.get_default_queue()


def get_queues(cluster_name: str) -> List[str]:
    """Get all queue names for an LSF cluster."""
    try:
        client = _create_lsf_client(cluster_name)
    except Exception as e:
        raise ValueError(
            f'Failed to connect to LSF cluster {cluster_name}: '
            f'{common_utils.format_exception(e)}') from e
    return [q.name for q in client.get_queues()]


def check_instance_fits(
        cluster: str,
        instance_type: str,
        queue: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Check if the given instance type fits in the given cluster/queue.

    Returns:
        Tuple of (fits, reason).
    """
    try:
        nodes = get_lsf_nodes_info(cluster)
    except FileNotFoundError:
        return (False, f'Could not query LSF cluster {cluster} because '
                f'the config file {DEFAULT_LSF_PATH} does not exist.')
    except Exception as e:  # pylint: disable=broad-except
        return (False, f'Could not query LSF cluster {cluster}: '
                f'{common_utils.format_exception(e)}.')

    lsf_instance = LsfInstanceType.from_instance_type(instance_type)
    acc_count = (lsf_instance.accelerator_count
                 if lsf_instance.accelerator_count is not None else 0)
    acc_type = lsf_instance.accelerator_type

    candidate_nodes = nodes
    if acc_type is not None and acc_count > 0:
        gpu_nodes = [n for n in nodes if n.gpus >= acc_count]
        if not gpu_nodes:
            return (False, f'No nodes with >= {acc_count} GPUs in cluster '
                    f'{cluster}.')
        candidate_nodes = gpu_nodes

    # Check CPU and memory fit
    for node in candidate_nodes:
        cpu_fits = node.cpus >= lsf_instance.cpus
        mem_fits = (lsf_instance.memory == 0 or
                    node.memory_gb >= lsf_instance.memory)
        if cpu_fits and mem_fits:
            return True, None

    max_cpu = max((n.cpus for n in candidate_nodes), default=0)
    max_mem = max((n.memory_gb for n in candidate_nodes), default=0.0)
    return (False, f'No nodes with enough resources. Max found: '
            f'{max_cpu} CPUs, {common_utils.format_float(max_mem)}G memory')


def get_enroot_config(cluster: str) -> Dict[str, Any]:
    """Get the enroot configuration for an LSF cluster from sky config.

    Returns config dict with keys: enabled, share_path, use_local_nvme,
    squash_options, nccl_tuning_file.
    """
    config = skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'enroot'), {})
    return {
        'enabled': config.get('enabled', False),
        'share_path': config.get('share_path', ''),
        'use_local_nvme': config.get('use_local_nvme', False),
        'squash_options': config.get('squash_options',
                                     '-comp lz4 -Xhc -no-xattrs'),
        'nccl_tuning_file': config.get('nccl_tuning_file', ''),
    }


def get_bsub_options(cluster: str,
                     queue: Optional[str] = None) -> Dict[str, str]:
    """Get bsub options from sky config with three-level merge.

    Merges: global lsf.bsub_options < cluster-level < queue-level.
    """
    global_opts = skypilot_config.get_nested(
        ('lsf', 'bsub_options'), {})
    cluster_opts = skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'bsub_options'), {})
    queue_opts = {}
    if queue is not None:
        queue_opts = skypilot_config.get_nested(
            ('lsf', 'cluster_configs', cluster,
             'queue_configs', queue, 'bsub_options'), {})

    merged = {}
    merged.update(global_opts)
    merged.update(cluster_opts)
    merged.update(queue_opts)
    return merged


def get_workdir(cluster: str) -> Optional[str]:
    """Get the configured workdir for an LSF cluster."""
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'workdir'), None)


def get_tmpdir(cluster: str) -> Optional[str]:
    """Get the configured tmpdir for an LSF cluster."""
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'tmpdir'), None)
