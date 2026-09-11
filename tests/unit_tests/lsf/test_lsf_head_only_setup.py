"""Tests that cluster setup runs once on LSF, not once per allocated node.

Per-instance setup stages (internal file mounts, runtime installation, logging
agent) are driven by _parallel_ssh_with_cache, which submits one job per
instance. That is right for clouds where an instance is a machine.

On LSF it is not: every allocated node is reached through the same login node, so
N instances share one filesystem, one SkyPilot runtime directory and one sshd.
Running the stages per instance means N concurrent identical installs writing the
same paths, plus N simultaneous SSH authentications against a server that limits
them. Neither failure is legible — a corrupt half-installed runtime, or auth
errors that read as a flaky network.
"""

import dataclasses
from unittest import mock

import pytest

from sky.provision import common
from sky.provision import provisioner


def _instance(instance_id: str, ip: str, node: str, rank: int):
    return common.InstanceInfo(
        instance_id=instance_id,
        internal_ip=ip,
        external_ip='login4.example.com',
        tags={
            'job_id': '12345',
            'node': node,
            'rank': str(rank),
        },
        ssh_port=22,
    )


def _cluster_info(num_nodes: int, provider: str = 'lsf'):
    """A ClusterInfo shaped like the LSF provisioner's get_cluster_info output."""
    hosts = [f'p1-r08-n{i + 4}' for i in range(num_nodes)]
    instances = {}
    for i, host in enumerate(hosts):
        inst_id = f'12345-{host}'
        instances[inst_id] = [_instance(inst_id, f'10.0.0.{i + 1}', host, i)]
    return common.ClusterInfo(
        instances=instances,
        head_instance_id=f'12345-{hosts[0]}',
        provider_name=provider,
        provider_config={'cluster': 'bluevela'},
    )


class TestHeadOnlyClusterInfo:
    """The narrowed view keeps everything but the worker instances."""

    def test_keeps_only_the_head_instance(self):
        info = _cluster_info(4)
        narrowed = provisioner._head_only_cluster_info(info)
        assert narrowed.num_instances == 1
        assert list(narrowed.instances) == [info.head_instance_id]

    def test_head_instance_is_unchanged(self):
        info = _cluster_info(4)
        narrowed = provisioner._head_only_cluster_info(info)
        assert (narrowed.get_head_instance() == info.get_head_instance())

    def test_provider_metadata_survives(self):
        """The runners built from this view still need the provider config."""
        info = _cluster_info(2)
        narrowed = provisioner._head_only_cluster_info(info)
        assert narrowed.provider_name == 'lsf'
        assert narrowed.provider_config == {'cluster': 'bluevela'}
        assert narrowed.head_instance_id == info.head_instance_id

    def test_original_is_not_mutated(self):
        """The task-execution path needs every node, so narrowing must copy."""
        info = _cluster_info(4)
        provisioner._head_only_cluster_info(info)
        assert info.num_instances == 4

    def test_single_node_is_a_no_op(self):
        info = _cluster_info(1)
        assert provisioner._head_only_cluster_info(info).num_instances == 1

    def test_requires_a_head_instance(self):
        info = dataclasses.replace(_cluster_info(2), head_instance_id=None)
        with pytest.raises(AssertionError, match='head_instance_id'):
            provisioner._head_only_cluster_info(info)


class TestParallelSshFanOut:
    """Show what the narrowing prevents, at the layer that would repeat work."""

    def _count_submissions(self, cluster_info):
        """Count how many per-instance jobs _parallel_ssh_with_cache submits."""
        from sky.provision import instance_setup

        runners = [
            mock.Mock(name=f'runner{i}')
            for i in range(cluster_info.num_instances)
        ]
        calls = []

        with mock.patch.object(instance_setup.provision,
                               'get_command_runners',
                               return_value=runners), \
             mock.patch.object(instance_setup.metadata_utils,
                               'cache_func',
                               side_effect=lambda *a, **k: (lambda f: f)), \
             mock.patch.object(instance_setup.provision_logging,
                               'get_log_path',
                               return_value='/tmp/provision.log'), \
             mock.patch.object(instance_setup.metadata_utils,
                               'get_instance_log_dir',
                               return_value=mock.MagicMock()):

            def record(runner, log_path):
                del log_path
                calls.append(runner)
                return None

            instance_setup._parallel_ssh_with_cache(record,
                                                    'cluster',
                                                    'stage',
                                                    None,
                                                    cluster_info,
                                                    ssh_credentials={},
                                                    max_workers=1)
        return len(calls)

    def test_all_instances_would_each_get_setup(self):
        """Baseline: the fan-out is per instance, as it is for real clouds."""
        assert self._count_submissions(_cluster_info(4)) == 4

    def test_narrowed_view_gets_setup_once(self):
        """With four LSF instances mapping to one login node, this is the
        difference between one install and four concurrent identical ones."""
        narrowed = provisioner._head_only_cluster_info(_cluster_info(4))
        assert self._count_submissions(narrowed) == 1


class TestClusterInfoForSetup:
    """The policy: which clouds get the narrowed view, and when."""

    def test_lsf_multi_node_is_narrowed(self):
        info = _cluster_info(4)
        assert provisioner._cluster_info_for_setup('lsf',
                                                   info).num_instances == 1

    def test_lsf_single_node_is_unchanged(self):
        """Nothing to deduplicate, so return the same object."""
        info = _cluster_info(1)
        assert provisioner._cluster_info_for_setup('lsf', info) is info

    @pytest.mark.parametrize('cloud', ['LSF', 'Lsf', 'lsf'])
    def test_cloud_name_is_matched_case_insensitively(self, cloud):
        """cloud_name comes from repr(cloud), whose casing is the cloud's own."""
        info = _cluster_info(4)
        assert provisioner._cluster_info_for_setup(cloud,
                                                   info).num_instances == 1

    @pytest.mark.parametrize('cloud', ['slurm', 'aws', 'kubernetes', 'runpod'])
    def test_other_clouds_keep_every_instance(self, cloud):
        """An instance really is a separate machine elsewhere; narrowing there
        would skip setup on nodes that need it."""
        info = _cluster_info(4, provider=cloud)
        result = provisioner._cluster_info_for_setup(cloud, info)
        assert result is info
        assert result.num_instances == 4
