"""Backend file_mount wrap-exemption, driven by a real SlurmCommandRunner.

_execute_file_mounts symlink-wraps absolute, non-``~/``/non-``/tmp/``
destinations. On a bare-host Slurm cluster that wrap runs ``sudo mkdir -p`` on
the compute node as the login user, which fails without passwordless sudo
("sudo: a password is required"). SlurmCommandRunner therefore declares the
configured Slurm ``workdir`` via get_unwrapped_mount_prefixes(), and
get_command_runners derives that root from the ``workdir`` config.

Mirrors tests/unit_tests/lsf/test_file_mount_wrap_exemption.py: drive the real
_execute_file_mounts with a real runner and assert the rsync target, with the
transfer fan-out and symlink execution stubbed so no SSH connection is opened.
"""

from unittest import mock

from sky.backends import cloud_vm_ray_backend
from sky.provision import common as provision_common
from sky.provision import constants as provision_constants
from sky.provision.slurm import instance as slurm_instance
from sky.utils import command_runner

_WORKDIR = '/proj/data-eng/llmb-read-write'


def _slurm_runner(shared_fs_roots, container_args=None):
    """Build a side-effect-free SlurmCommandRunner.

    ``ssh_private_key=None`` skips key-file creation in the SSH base class, so
    construction opens no connection.

    Args:
        shared_fs_roots: value forwarded to the ``shared_fs_roots`` kwarg.
        container_args: pyxis args, or None for a bare-host runner.

    Returns:
        A constructed SlurmCommandRunner.
    """
    return command_runner.SlurmCommandRunner(('login-host', 22),
                                             'me',
                                             None,
                                             sky_dir=f'{_WORKDIR}/.sky-c',
                                             skypilot_runtime_dir='/tmp/rt',
                                             job_id='1',
                                             slurm_node='n1',
                                             container_args=container_args,
                                             shared_fs_roots=shared_fs_roots)


def _run_file_mounts(tmp_path, monkeypatch, runner, file_mounts):
    """Drive the real _execute_file_mounts and capture per-mount rsync targets.

    Args:
        tmp_path: pytest tmp dir fixture (holds the real source file + logs).
        monkeypatch: pytest monkeypatch fixture.
        runner: the command runner the handle returns.
        file_mounts: destinations; each is mapped to one real source file.

    Returns:
        The ``target`` values in file_mounts insertion order.
    """
    src = tmp_path / 'payload.txt'
    src.write_text('data')
    file_mounts = {dst: str(src) for dst in file_mounts}

    handle = mock.MagicMock()
    handle.get_command_runners.return_value = [runner]
    handle.launched_resources.cloud = 'slurm'

    targets = []
    monkeypatch.setattr(
        cloud_vm_ray_backend.backend_utils, 'parallel_data_transfer_to_nodes',
        lambda runners, *, source, target, **kw: targets.append(target))
    monkeypatch.setattr(cloud_vm_ray_backend.subprocess_utils,
                        'get_max_workers_for_file_mounts', lambda *a, **k: 1)
    monkeypatch.setattr(cloud_vm_ray_backend.subprocess_utils,
                        'run_in_parallel', lambda fn, items, *a, **k: None)
    monkeypatch.setattr(cloud_vm_ray_backend.rich_utils, 'force_update_status',
                        lambda *a, **k: None)

    backend = cloud_vm_ray_backend.CloudVmRayBackend.__new__(
        cloud_vm_ray_backend.CloudVmRayBackend)
    backend.log_dir = str(tmp_path / 'logs')

    backend._execute_file_mounts(handle, file_mounts)
    return targets


def test_workdir_destination_bypasses_wrap(tmp_path, monkeypatch):
    """The gbserver-remapped dst under workdir rsyncs straight through; a
    sibling sharing a string prefix and an unrelated abs path are wrapped."""
    dst = f'{_WORKDIR}/builds/builds/b1/runs/r1/src'
    targets = _run_file_mounts(tmp_path,
                               monkeypatch,
                               _slurm_runner([_WORKDIR]),
                               file_mounts=[
                                   dst,
                                   f'{_WORKDIR}-other/src',
                                   '/opt/other',
                               ])
    assert targets[0] == dst  # under workdir -> un-wrapped
    assert targets[1] != f'{_WORKDIR}-other/src'  # prefix sibling -> wrapped
    assert targets[2] != '/opt/other'  # unrelated abs path -> wrapped


def test_containerized_runner_also_exempts(tmp_path, monkeypatch):
    """The exemption holds in container mode: workdir is bind-mounted identity
    into the pyxis container, so the rsync writes through to the host path."""
    dst = f'{_WORKDIR}/builds/src'
    targets = _run_file_mounts(tmp_path,
                               monkeypatch,
                               _slurm_runner([_WORKDIR],
                                             container_args='--container-x'),
                               file_mounts=[dst])
    assert targets[0] == dst


def test_no_roots_wraps_everything(tmp_path, monkeypatch):
    """Without shared roots (no workdir configured) the wrap is unchanged."""
    dst = f'{_WORKDIR}/builds/src'
    targets = _run_file_mounts(tmp_path,
                               monkeypatch,
                               _slurm_runner(None),
                               file_mounts=[dst])
    assert targets[0] != dst


def test_default_runner_has_no_roots():
    """Omitting shared_fs_roots keeps the base-class behavior ([])."""
    runner = command_runner.SlurmCommandRunner(('login-host', 22),
                                               'me',
                                               None,
                                               sky_dir='/home/me/.sky-c',
                                               skypilot_runtime_dir='/tmp/rt',
                                               job_id='1',
                                               slurm_node='n1',
                                               container_args=None)
    assert runner.get_unwrapped_mount_prefixes() == []


def _runners_for_workdir(monkeypatch, workdir):
    """Call get_command_runners with the workdir config set to ``workdir``.

    Stubs the SlurmClient (no SSH) and the skypilot_config lookup.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        workdir: value returned for the ('workdir',) config key, or None.

    Returns:
        The runners get_command_runners built.
    """
    client = mock.MagicMock()
    client.get_remote_home_dir.return_value = '/home/me'
    client.get_env.return_value = {}
    client.check_file_exists.return_value = False
    monkeypatch.setattr(slurm_instance.slurm, 'SlurmClient',
                        lambda *a, **k: client)

    def _config(cloud, region, keys, default_value=None):
        del cloud, region, default_value
        return workdir if keys == ('workdir',) else None

    monkeypatch.setattr(slurm_instance.skypilot_config,
                        'get_effective_region_config', _config)

    instance = provision_common.InstanceInfo(
        instance_id='i1',
        internal_ip='10.0.0.1',
        external_ip='login-host',
        tags={
            'job_id': '1',
            'node': 'n1',
            provision_constants.TAG_SKYPILOT_CLUSTER_NAME: 'c1',
        },
        ssh_port=22)
    cluster_info = provision_common.ClusterInfo(
        instances={'i1': [instance]},
        head_instance_id='i1',
        provider_name='slurm',
        provider_config={
            'cluster': 'bluevela',
            'ssh': {
                'hostname': 'login-host',
                'port': 22,
                'user': 'me',
                'private_key': None,
            },
        })
    return slurm_instance.get_command_runners(cluster_info)


def test_get_command_runners_exempts_configured_workdir(monkeypatch):
    """A configured workdir becomes the runner's only unwrapped root."""
    runners = _runners_for_workdir(monkeypatch, _WORKDIR)
    assert [r.get_unwrapped_mount_prefixes() for r in runners] == [[_WORKDIR]]


def test_get_command_runners_without_workdir_keeps_wrap(monkeypatch):
    """No workdir (home-based cluster) -> no exemption."""
    runners = _runners_for_workdir(monkeypatch, None)
    assert [r.get_unwrapped_mount_prefixes() for r in runners] == [[]]
