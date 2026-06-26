"""IBM Spectrum LSF cloud for SkyPilot."""
import functools
import logging
import typing
from typing import Dict, Iterator, List, Optional, Tuple, Union

from sky import clouds
from sky import exceptions
from sky import sky_logging
from sky import skypilot_config
from sky.provision.lsf import utils as lsf_utils
from sky.utils import annotations
from sky.utils import registry
from sky.utils import resources_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky import status_lib

logger = sky_logging.init_logger(__name__)

_DEFAULT_NUM_VCPUS_WITH_GPU = 4
_DEFAULT_MEMORY_CPU_RATIO_WITH_GPU = 4

_SHARED_FS_TYPES = frozenset({
    'nfs', 'nfs4', 'lustre', 'gpfs', 'beegfs',
    'ceph', 'fuse.ceph', 'glusterfs', 'fuse.glusterfs',
})


@registry.CLOUD_REGISTRY.register
class LSF(clouds.Cloud):
    """IBM Spectrum LSF."""

    _REPR = 'LSF'
    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP:
            ('LSF does not support stopping jobs.'),
        clouds.CloudImplementationFeatures.AUTOSTOP:
            ('LSF does not support autostop.'),
        clouds.CloudImplementationFeatures.SPOT_INSTANCE:
            ('LSF does not support spot instances.'),
        clouds.CloudImplementationFeatures.OPEN_PORTS:
            ('LSF does not support opening ports.'),
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            ('LSF does not support host controllers.'),
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            ('LSF does not support local disk requests.'),
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            ('LSF does not support disk cloning.'),
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            ('LSF does not support custom disk tiers.'),
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            ('LSF does not support custom multi-network.'),
        clouds.CloudImplementationFeatures.AUTO_TERMINATE:
            ('LSF does not support auto-terminate.'),
        clouds.CloudImplementationFeatures.AUTODOWN:
            ('LSF does not support autodown.'),
    }
    _DYNAMICALLY_CHECKED_FEATURES = {
        clouds.CloudImplementationFeatures.DOCKER_IMAGE,
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING,
    }

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT

    _MAX_CLUSTER_NAME_LEN_LIMIT = 120
    _regions: List[clouds.Region] = []

    @classmethod
    def _unsupported_features_for_resources(
        cls, resources: 'resources_lib.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        unsupported = cls._CLOUD_UNSUPPORTED_FEATURES.copy()

        # Docker image support depends on enroot availability
        cluster = region
        if cluster is None and resources.infra is not None:
            cluster = resources.infra.region
        if cluster is not None:
            if not lsf_utils.check_enroot_enabled(cluster):
                unsupported[clouds.CloudImplementationFeatures.DOCKER_IMAGE] = (
                    'Docker image support requires enroot on the LSF cluster. '
                    f'Cluster {cluster!r} does not have enroot installed.')

            if not lsf_utils.check_fuse_enabled(cluster):
                unsupported[
                    clouds.CloudImplementationFeatures.STORAGE_MOUNTING] = (
                        'Storage mounting requires FUSE on the LSF cluster. '
                        f'Cluster {cluster!r} does not have FUSE available.')

        return unsupported

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    @classmethod
    def uses_ray(cls) -> bool:
        return False

    @classmethod
    def existing_allowed_clusters(
        cls,
    ) -> List[str]:
        """Returns list of LSF clusters available and allowed by config."""
        all_clusters = lsf_utils.get_all_lsf_cluster_names()
        if not all_clusters:
            return []

        allowed = skypilot_config.get_nested(
            ('lsf', 'allowed_clusters'), None)

        if allowed is None or allowed == 'all':
            return all_clusters

        if isinstance(allowed, list):
            return [c for c in allowed if c in all_clusters]

        return all_clusters

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, float]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None,
    ) -> List[clouds.Region]:
        """Returns regions (clusters) with the requested resources.

        For LSF: region = cluster name, zone = queue name.
        """
        del use_spot  # LSF has no spot instances

        allowed_clusters = cls.existing_allowed_clusters()
        if not allowed_clusters:
            return []

        regions = []
        clusters_to_check = (
            [region] if region is not None else allowed_clusters)

        for cluster in clusters_to_check:
            if cluster not in allowed_clusters:
                continue

            # Get available queues for this cluster
            try:
                queues = lsf_utils.get_queues(cluster)
            except Exception:  # pylint: disable=broad-except
                continue

            if zone is not None:
                queues = [q for q in queues if q == zone]

            # Check if instance type fits
            if instance_type is not None:
                fitting_queues = []
                for queue in queues:
                    fits, _ = lsf_utils.check_instance_fits(
                        cluster, instance_type, queue)
                    if fits:
                        fitting_queues.append(queue)
                queues = fitting_queues

            if queues:
                zones = [clouds.Zone(name=q) for q in queues]
                regions.append(clouds.Region(name=cluster).set_zones(zones))

        return regions

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, float]] = None,
        use_spot: bool = False,
    ) -> Iterator[Optional[List[clouds.Zone]]]:
        """Iterates over queues for provisioning attempts."""
        regions = cls.regions_with_offering(
            instance_type, accelerators, use_spot, region, zone=None)

        for r in regions:
            if r.name == region:
                if r.zones is not None:
                    for z in r.zones:
                        yield [z]
                else:
                    yield None
                return

        yield None

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls, instance_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        inst = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
        return inst.cpus, inst.memory

    @classmethod
    def get_accelerators_from_instance_type(
        cls, instance_type: str
    ) -> Optional[Dict[str, Union[int, float]]]:
        inst = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
        if inst.accelerator_type is not None and inst.accelerator_count:
            return {inst.accelerator_type: inst.accelerator_count}
        return None

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[int] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
    ) -> Optional[str]:
        from sky.catalog import lsf_catalog
        return lsf_catalog.get_default_instance_type(
            cpus=cpus, memory=memory, region=region, zone=zone)

    @classmethod
    def _get_feasible_launchable_resources(
        cls, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        """Returns feasible resources for the given constraints."""
        # If instance type is already specified, validate it
        if resources.instance_type is not None:
            regions = cls.regions_with_offering(
                resources.instance_type,
                resources.accelerators,
                resources.use_spot,
                resources.region,
                resources.zone)
            if not regions:
                return resources_utils.FeasibleResources([], [], None)
            r = resources.copy(cloud=LSF(),
                               instance_type=resources.instance_type)
            return resources_utils.FeasibleResources([r], [], None)

        # Construct instance type from resource requirements
        cpus = float(resources.cpus) if resources.cpus else (
            _DEFAULT_NUM_VCPUS_WITH_GPU if resources.accelerators
            else 2.0)
        memory = float(resources.memory) if resources.memory else (
            cpus * _DEFAULT_MEMORY_CPU_RATIO_WITH_GPU
            if resources.accelerators else cpus * 4)

        if resources.accelerators:
            acc_type, acc_count = list(resources.accelerators.items())[0]
            inst = lsf_utils.LsfInstanceType.from_resources(
                cpus, memory, acc_count, acc_type)
        else:
            inst = lsf_utils.LsfInstanceType.from_resources(cpus, memory)

        instance_type = inst.name
        regions = cls.regions_with_offering(
            instance_type,
            resources.accelerators,
            resources.use_spot,
            resources.region,
            resources.zone)

        if not regions:
            return resources_utils.FeasibleResources([], [], None)

        r = resources.copy(cloud=LSF(), instance_type=instance_type)
        return resources_utils.FeasibleResources([r], [], None)

    @classmethod
    def _check_compute_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        """Check if LSF credentials are configured."""
        allowed_clusters = cls.existing_allowed_clusters()
        if not allowed_clusters:
            return (False, f'No LSF clusters found. Please create '
                    f'{lsf_utils.DEFAULT_LSF_PATH} with cluster SSH config.')

        # Try connecting to at least one cluster
        for cluster in allowed_clusters:
            try:
                client = lsf_utils._create_lsf_client(cluster)
                client.info()
                return (True, None)
            except Exception:  # pylint: disable=broad-except
                continue

        return (False, f'Could not connect to any LSF cluster. '
                f'Checked: {allowed_clusters}. Verify SSH config in '
                f'{lsf_utils.DEFAULT_LSF_PATH}.')

    @classmethod
    def get_credential_file_mounts(cls) -> Dict[str, str]:
        return {}

    @classmethod
    def instance_type_to_hourly_cost(
        cls, instance_type: str, use_spot: bool, region: Optional[str],
        zone: Optional[str]
    ) -> float:
        from sky.catalog import lsf_catalog
        return lsf_catalog.get_hourly_cost(
            instance_type, use_spot, region, zone)

    @classmethod
    def accelerators_to_hourly_cost(
        cls, accelerators: Dict[str, float], use_spot: bool,
        region: Optional[str], zone: Optional[str]
    ) -> float:
        # On-prem clusters typically have no hourly cost
        return 0.0

    @classmethod
    def get_egress_cost(cls, num_gigabytes: float) -> float:
        return 0.0

    def get_image_size(self, image_id: str, region: Optional[str]) -> float:
        return 0.0

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: 'resources_lib.ClusterName',
        region: clouds.Region,
        zones: Optional[List[clouds.Zone]],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List] = None,
    ) -> Dict[str, Optional[str]]:
        """Convert Resources to LSF-specific deployment variables."""
        cluster = region.name
        queue = zones[0].name if zones else None

        # Parse instance type
        inst = lsf_utils.LsfInstanceType.from_instance_type(
            resources.instance_type)

        # Get SSH config for the cluster
        ssh_config = lsf_utils.get_lsf_ssh_config()
        ssh_config_dict = ssh_config.lookup(cluster)

        # Get bsub options from config
        bsub_options = lsf_utils.get_bsub_options(cluster, queue)

        # Get enroot config
        enroot_config = lsf_utils.get_enroot_config(cluster)

        deploy_vars = {
            'instance_type': resources.instance_type,
            'cpus': str(inst.cpus),
            'memory': str(inst.memory),
            'lsf_cluster': cluster,
            'lsf_queue': queue,
            'num_nodes': str(num_nodes),
            'accelerator_count': (str(inst.accelerator_count)
                                  if inst.accelerator_count else '0'),
            'accelerator_type': inst.accelerator_type or '',
            'ssh_hostname': ssh_config_dict.get('hostname', ''),
            'ssh_port': ssh_config_dict.get('port', '22'),
            'ssh_user': ssh_config_dict.get('user', ''),
            'lsf_private_key': lsf_utils.get_identity_file(
                ssh_config_dict) or '',
            'lsf_proxy_command': ssh_config_dict.get('proxycommand', ''),
            'lsf_proxy_jump': ssh_config_dict.get('proxyjump', ''),
            'lsf_identities_only': str(
                lsf_utils.get_identities_only(ssh_config_dict)),
            'image_id': (list(resources.image_id.values())[0]
                        if resources.image_id else ''),
            'bsub_options': bsub_options,
            'enroot_enabled': str(enroot_config['enabled']),
            'enroot_share_path': enroot_config['share_path'],
            'enroot_use_local_nvme': str(enroot_config['use_local_nvme']),
            'enroot_squash_options': enroot_config['squash_options'],
            'nccl_tuning_file': enroot_config.get('nccl_tuning_file', ''),
            'workdir': lsf_utils.get_workdir(cluster) or '',
            'tmpdir': lsf_utils.get_tmpdir(cluster) or '',
        }

        return deploy_vars

    def __repr__(self):
        return self._REPR

    def is_same_cloud(self, other: clouds.Cloud) -> bool:
        return isinstance(other, LSF)
