"""Tests that cluster-level LSF config actually reaches the provisioner.

Config travels sky config -> lsf_utils getter -> make_deploy_resources_variables
-> lsf-ray.yml.j2 -> provider_config -> the bsub script. A break anywhere in that
chain is silent: the provisioner reads the key with a default, gets the default,
and produces a script that runs fine but without whatever was configured.

Two such breaks are covered here, both found by reading the schema against the
getters:

- ``enroot_mounts`` was never emitted by the template, so extra bind mounts and
  the wrap-exemption roots derived from them were dropped.
- ``nccl_tuning_file`` was read from inside the ``enroot`` block, where the
  schema forbids it (``additionalProperties: False``), so it always resolved to
  '' and the tuning script was never sourced on any node.
"""

import os
from unittest import mock

import pytest

from sky.provision.lsf import instance as lsf_instance
from sky.provision.lsf import utils as lsf_utils

# The real BlueVela shape, per the schema at sky/utils/schemas.py and
# configurations/assets/environments/skypilot/lsf/ibm-bluevela/environment.yaml
# in the granite.build repo. Note where each key sits: `enroot` is a nested
# block, while nccl_tuning_file and enroot_mounts are its siblings.
BLUEVELA_CLUSTER_CONFIG = {
    'workdir': '/proj/granite-build/g4os/skypilot',
    'tmpdir': '/opt/nvme/$USER/skypilot-tmp',
    'enroot': {
        'enabled': True,
        'share_path': '/proj/granite-build/g4os',
        'use_local_nvme': True,
        'squash_options': '-comp lz4 -Xhc -no-xattrs',
    },
    'nccl_tuning_file': '/proj/granite-build/g4os/bv-nccl-tuning.sh',
    'enroot_mounts': [
        '/dev/shm /dev/shm',
        '/dev/infiniband /dev/infiniband',
    ],
}


@pytest.fixture
def bluevela_config():
    """Patch skypilot_config so the getters see the BlueVela cluster config."""

    def fake_get_nested(keys, default=None, **kwargs):
        del kwargs
        assert keys[:3] == ('lsf', 'cluster_configs', 'bluevela'), keys
        node = BLUEVELA_CLUSTER_CONFIG
        for key in keys[3:]:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    with mock.patch.object(lsf_utils.skypilot_config,
                           'get_nested',
                           side_effect=fake_get_nested):
        yield


class TestClusterLevelGetters:
    """The getters must read each key at the level the schema defines."""

    def test_nccl_tuning_file_is_read_from_cluster_level(self, bluevela_config):
        """Previously read from inside `enroot`, so it was always ''.

        A silent miss: the bsub script's nccl_block is conditional on a
        non-empty value, so the tuning script was simply never sourced, and a
        job that should ride InfiniBand would fall back to whatever NCCL
        autodetects — slower, but not an error.
        """
        del bluevela_config
        assert lsf_utils.get_nccl_tuning_file('bluevela') == (
            '/proj/granite-build/g4os/bv-nccl-tuning.sh')

    def test_enroot_mounts_are_read_from_cluster_level(self, bluevela_config):
        del bluevela_config
        assert lsf_utils.get_enroot_mounts('bluevela') == [
            '/dev/shm /dev/shm',
            '/dev/infiniband /dev/infiniband',
        ]

    def test_enroot_config_no_longer_claims_nccl_tuning_file(
            self, bluevela_config):
        """It is not enroot config, and leaving it there invited the bug back."""
        del bluevela_config
        assert 'nccl_tuning_file' not in lsf_utils.get_enroot_config('bluevela')

    def test_enroot_config_still_reads_its_own_nested_keys(
            self, bluevela_config):
        del bluevela_config
        config = lsf_utils.get_enroot_config('bluevela')
        assert config['enabled'] is True
        assert config['share_path'] == '/proj/granite-build/g4os'
        assert config['use_local_nvme'] is True

    def test_missing_keys_fall_back_to_empty(self):
        """An unconfigured cluster must not raise."""

        def empty(keys, default=None, **kwargs):
            del keys, kwargs
            return default

        with mock.patch.object(lsf_utils.skypilot_config,
                               'get_nested',
                               side_effect=empty):
            assert lsf_utils.get_nccl_tuning_file('other') == ''
            assert lsf_utils.get_enroot_mounts('other') == []


class TestEnrootMountsReachTheScript:
    """provider_config -> bsub script, the last hop of the chain."""

    def _config(self, mounts):
        return {
            'cluster': 'bluevela',
            'cpus': '4',
            'memory': '64',
            'memory_explicit': 'False',
            'cpus_explicit': 'False',
            'accelerator_count': '8',
            'image_id': 'docker:example.io/img:1.0',
            'enroot_enabled': 'True',
            'enroot_share_path': '/proj/granite-build/g4os',
            'workdir': '/proj/granite-build/g4os/skypilot',
            'enroot_mounts': mounts,
        }

    def test_configured_mounts_appear_in_the_enroot_mount_block(self):
        script = lsf_instance._build_bsub_script(
            'sky-test',
            self._config(['/dev/shm /dev/shm', '/gpfs /gpfs']),
            num_nodes=2)
        assert 'echo "/dev/shm /dev/shm"' in script
        assert 'echo "/gpfs /gpfs"' in script

    def test_builtin_mounts_are_still_emitted(self):
        """They are unconditional, independent of configuration."""
        script = lsf_instance._build_bsub_script('sky-test',
                                                 self._config([]),
                                                 num_nodes=1)
        for path in ('/proj', '/tmp', '/opt/nvme', '/opt/share'):
            assert f'echo "{path} {path}"' in script

    def test_nccl_tuning_file_is_sourced_when_configured(self):
        config = self._config([])
        config['nccl_tuning_file'] = '/proj/granite-build/g4os/bv-nccl.sh'
        script = lsf_instance._build_bsub_script('sky-test',
                                                 config,
                                                 num_nodes=2)
        assert 'source "/proj/granite-build/g4os/bv-nccl.sh"' in script

    def test_no_source_line_when_unconfigured(self):
        script = lsf_instance._build_bsub_script('sky-test',
                                                 self._config([]),
                                                 num_nodes=2)
        assert 'bv-nccl' not in script


class TestWrapExemptionFollowsConfiguredMounts:
    """A configured shared FS must also become a wrap-exemption root.

    Without this, a file_mount targeting it is symlink-wrapped instead of written
    through, so the payload does not appear at the identity path the container
    sees.
    """

    def test_shared_configured_mount_becomes_a_root(self):
        roots = lsf_instance._derive_shared_fs_roots(['/gpfs /gpfs'])
        assert '/gpfs' in roots
        assert '/proj' in roots, 'built-ins must survive'

    def test_node_local_device_mounts_do_not_become_roots(self):
        """/dev/shm and /dev/infiniband are identity-mounted but node-local, so
        a login-node write to them is not visible to the job."""
        roots = lsf_instance._derive_shared_fs_roots(
            ['/dev/shm /dev/shm', '/dev/infiniband /dev/infiniband'])
        assert '/dev/shm' not in roots
        assert '/dev/infiniband' not in roots
        assert roots == ['/proj', '/opt/share']

    def test_real_bluevela_mounts_add_no_roots(self):
        """The configured set is entirely node-local devices, so fixing the
        propagation must not change the exemption list on this cluster."""
        roots = lsf_instance._derive_shared_fs_roots(
            BLUEVELA_CLUSTER_CONFIG['enroot_mounts'])
        assert roots == ['/proj', '/opt/share']


class TestTemplateRendersProviderKeys:
    """Render lsf-ray.yml.j2 and read the provider block back as YAML.

    This is the hop that silently dropped enroot_mounts: the getter had it, the
    provisioner read it, and the template in between simply never wrote it out.
    Asserting on the rendered document covers that gap in a way that testing
    either side alone does not.
    """

    TEMPLATE = 'lsf-ray.yml.j2'

    def _render(self, **overrides):
        import jinja2
        import yaml

        import sky

        template_path = os.path.join(os.path.dirname(sky.__file__), 'templates',
                                     self.TEMPLATE)

        variables = {
            'cluster_name_on_cloud': 'sky-test',
            'num_nodes': 2,
            'credentials': {},
            'lsf_cluster': 'bluevela',
            'ssh_user': 'granitebuild',
            'ssh_hostname': 'login4.example.com',
            'ssh_port': 22,
            'lsf_private_key': '~/.ssh/key',
            'lsf_identities_only': True,
            'cpus': '4',
            'memory': '64',
            'disk_size': 256,
            'accelerator_count': '8',
            'accelerator_type': 'H100',
            'instance_type': 'lsf-8H100',
            'image_id': 'docker:example.io/img:1.0',
            'provision_timeout': 1800,
            'ready_timeout': 1800,
            'enroot_enabled': 'True',
            'enroot_share_path': '/proj/granite-build/g4os',
            'enroot_use_local_nvme': 'True',
            'enroot_squash_options': '-comp lz4',
            'nccl_tuning_file': '',
            'enroot_mounts': [],
            'workdir': '/proj/granite-build/g4os/skypilot',
            'tmpdir': '/opt/nvme/$USER/skypilot-tmp',
            'bsub_options': {},
            'sky_wheel_hash': 'deadbeef',
            'sky_local_path': '/tmp/wheel',
            'sky_remote_path': '/tmp/wheel',
            'sky_ray_yaml_local_path': '/tmp/ray.yml',
            'sky_ray_yaml_remote_path': '/tmp/ray.yml',
        }
        variables.update(overrides)

        with open(template_path, 'r', encoding='utf-8') as f:
            # Undefined-tolerant: the template carries keys unrelated to this
            # test (file mounts, wheel paths) that the real caller supplies.
            env = jinja2.Environment(undefined=jinja2.ChainableUndefined)
            rendered = env.from_string(f.read()).render(**variables)
        return yaml.safe_load(rendered)

    def test_enroot_mounts_reach_provider_config(self):
        doc = self._render(enroot_mounts=['/dev/shm /dev/shm', '/gpfs /gpfs'])
        assert doc['provider']['enroot_mounts'] == [
            '/dev/shm /dev/shm', '/gpfs /gpfs'
        ]

    def test_enroot_mounts_omitted_when_empty(self):
        doc = self._render(enroot_mounts=[])
        assert 'enroot_mounts' not in doc['provider']

    def test_resource_explicitness_flags_reach_provider_config(self):
        """The provisioner cannot tell a requested value from a catalog default
        on its own, so these two booleans are the whole basis of the
        explicit-request-beats-bsub_options precedence. This is also the hop that
        silently dropped enroot_mounts, so assert on the rendered document."""
        doc = self._render(memory_explicit='True', cpus_explicit='False')
        assert doc['provider']['memory_explicit'] == 'True'
        assert doc['provider']['cpus_explicit'] == 'False'

    def test_nccl_tuning_file_reaches_provider_config(self):
        doc = self._render(nccl_tuning_file='/proj/bv-nccl.sh')
        assert doc['provider']['nccl_tuning_file'] == '/proj/bv-nccl.sh'
