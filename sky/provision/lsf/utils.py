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

# How long to wait for LSF to allocate nodes to a submitted job. Generous by
# design: the LSF backend always pins a queue (SkyPilot's `zone`), so there is
# no other zone to fail over to, and abandoning the wait only forfeits the
# job's place in the queue — the next attempt is resubmitted at the back of it.
# On a busy shared queue a wait of hours is normal, not a fault. This mirrors
# the Slurm backend, which uses the same 24h value whenever a partition is
# pinned (see `sky/clouds/slurm.py`). Override per cluster or per queue with
# `lsf.cluster_configs.<cluster>.provision_timeout` (negative = indefinitely).
DEFAULT_PROVISION_TIMEOUT = 24 * 60 * 60

# How long to wait, once nodes are allocated, for the bsub script to signal
# container readiness. This window covers `enroot import` of the step image,
# which on a cold cache means squashing a multi-GB image onto shared storage.
# Override with `lsf.cluster_configs.<cluster>.ready_timeout`.
DEFAULT_READY_TIMEOUT = 60 * 60


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


def _pick_probe_node(cluster: str) -> Optional[str]:
    """Pick a compute node to run feature-detection commands on.

    Feature probes must not run on the submit (login) host, which often does
    not have the same software installed as the nodes that run jobs. GPU nodes
    are preferred since they are where containerized workloads land; master
    and other non-compute hosts report zero GPUs.

    Returns:
        A node name, or None if no suitable node could be determined.
    """
    try:
        nodes = get_lsf_nodes_info(cluster)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug(f'Failed to list nodes for LSF cluster {cluster}: '
                     f'{common_utils.format_exception(e)}')
        return None
    ok_nodes = [n for n in nodes if n.status == 'ok']
    for node in ok_nodes:
        if node.gpus > 0:
            return node.node
    if ok_nodes:
        return ok_nodes[0].node
    return None


def _check_cluster_feature(
    cluster: str,
    feature_name: str,
    check_fn: Callable[[lsf.LsfClient], Optional[bool]],
    cache_ttl: int,
    default_if_unknown: bool = False,
) -> bool:
    """Check if a feature is available on an LSF cluster, with caching.

    Args:
        default_if_unknown: What to report when the check is inconclusive.
            Inconclusive results are not cached, so a later launch retries.
    """
    cache_key = f'lsf:{feature_name}_enabled:{cluster}'
    cached = kv_cache.get_cache_entry(cache_key)
    if cached is not None:
        logger.debug(f'LSF {feature_name} check found in cache ({cache_key})')
        return cached == 'true'

    client = _create_lsf_client(cluster)
    enabled = check_fn(client)
    if enabled is None:
        logger.debug(f'LSF {feature_name} check on {cluster} was '
                     f'inconclusive; assuming {default_if_unknown}')
        return default_if_unknown

    try:
        kv_cache.add_or_update_cache_entry(cache_key,
                                           'true' if enabled else 'false',
                                           time.time() + cache_ttl)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug(f'Failed to cache LSF {feature_name} check for '
                     f'{cluster}: {common_utils.format_exception(e)}')

    return enabled


def check_enroot_enabled(cluster: str) -> bool:
    """Check if enroot can be used on an LSF cluster.

    Config is authoritative: provisioning only invokes enroot when
    `lsf.cluster_configs.<cluster>.enroot.enabled` is set (see
    LSF.make_deploy_resources_variables), so a cluster that does not declare
    it cannot use containers regardless of what is installed. When it is
    declared, verify on a compute node -- never on the submit host, which may
    not have enroot even though the compute nodes do.
    """
    if not get_enroot_config(cluster)['enabled']:
        return False
    return _check_cluster_feature(
        cluster,
        'enroot_compute',
        lambda c: c.check_enroot_available(node=_pick_probe_node(cluster)),
        _LSF_ENROOT_CHECK_CACHE_TTL,
        default_if_unknown=True)


def check_fuse_enabled(cluster: str) -> bool:
    """Check if FUSE is available on an LSF cluster's compute nodes."""
    return _check_cluster_feature(
        cluster,
        'fuse_compute',
        lambda c: c.check_fuse_enabled(node=_pick_probe_node(cluster)),
        _LSF_FUSE_CHECK_CACHE_TTL,
        default_if_unknown=True)


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


_PREFERRED_QUEUE_ORDER = ['normal', 'short', 'interactive', 'priority']


def get_queues(cluster_name: str) -> List[str]:
    """Get open queue names for an LSF cluster, ordered by preference.

    The 'normal' queue is preferred over others since it's the standard
    general-purpose batch queue on most LSF clusters.
    """
    try:
        client = _create_lsf_client(cluster_name)
    except Exception as e:
        raise ValueError(
            f'Failed to connect to LSF cluster {cluster_name}: '
            f'{common_utils.format_exception(e)}') from e
    open_queues = [q.name for q in client.get_queues() if q.is_open]

    def _queue_sort_key(name: str) -> int:
        try:
            return _PREFERRED_QUEUE_ORDER.index(name)
        except ValueError:
            return len(_PREFERRED_QUEUE_ORDER)

    open_queues.sort(key=_queue_sort_key)
    return open_queues


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
    squash_options.

    Note that nccl_tuning_file and enroot_mounts are NOT here: the schema places
    both at cluster level, as siblings of `enroot` rather than inside it (and
    `enroot` sets additionalProperties: False, so they could not be nested even
    if a user tried). See get_nccl_tuning_file() and get_enroot_mounts().
    """
    config = skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'enroot'), {})
    return {
        'enabled': config.get('enabled', False),
        'share_path': config.get('share_path', ''),
        'use_local_nvme': config.get('use_local_nvme', False),
        'squash_options': config.get('squash_options',
                                     '-comp lz4 -Xhc -no-xattrs'),
    }


def get_nccl_tuning_file(cluster: str) -> str:
    """Get the NCCL tuning script to source on each node, from sky config.

    Read from cluster level, where the schema defines it. It was previously read
    from inside the `enroot` block, which the schema forbids, so it always
    resolved to '' and the tuning script was never sourced.
    """
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'nccl_tuning_file'), '')


def get_enroot_mounts(cluster: str) -> List[str]:
    """Get the extra enroot bind-mount specs for a cluster, from sky config.

    Each entry is an enroot mounts(5) line, e.g. ``"/gpfs /gpfs"``. These are
    added to the built-in identity mounts, and any that qualify as shared
    identity mounts also become file_mount wrap-exemption roots (see
    _derive_shared_fs_roots in sky/provision/lsf/instance.py).
    """
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'enroot_mounts'), [])


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


def _get_timeout_config(cluster: str, queue: Optional[str], key: str,
                        default: int) -> int:
    """Read a launch-phase timeout from sky config with three-level merge.

    Merges: global `lsf.<key>` < cluster-level < queue-level, so a queue whose
    scheduling behaviour differs (e.g. `preemptable` vs `normal`) can override
    the cluster default. A negative value means "wait indefinitely".
    """
    config_keys = [
        ('lsf', key),
        ('lsf', 'cluster_configs', cluster, key),
    ]
    if queue is not None:
        config_keys.append(
            ('lsf', 'cluster_configs', cluster, 'queue_configs', queue, key))

    value = None
    for keys in config_keys:
        level_value = skypilot_config.get_nested(keys, None)
        if level_value is not None:
            value = level_value

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f'Ignoring non-integer lsf {key} {value!r} for cluster '
                       f'{cluster}; using {default}s.')
        return default


def get_provision_timeout(cluster: str, queue: Optional[str] = None) -> int:
    """Seconds to wait for LSF to allocate nodes to a submitted job.

    Negative means wait indefinitely. See DEFAULT_PROVISION_TIMEOUT for why
    the default is generous.
    """
    return _get_timeout_config(cluster, queue, 'provision_timeout',
                               DEFAULT_PROVISION_TIMEOUT)


def get_ready_timeout(cluster: str, queue: Optional[str] = None) -> int:
    """Seconds to wait for container readiness once nodes are allocated.

    Negative means wait indefinitely.
    """
    return _get_timeout_config(cluster, queue, 'ready_timeout',
                               DEFAULT_READY_TIMEOUT)


def get_workdir(cluster: str) -> Optional[str]:
    """Get the configured workdir for an LSF cluster."""
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'workdir'), None)


def get_tmpdir(cluster: str) -> Optional[str]:
    """Get the configured tmpdir for an LSF cluster."""
    return skypilot_config.get_nested(
        ('lsf', 'cluster_configs', cluster, 'tmpdir'), None)
