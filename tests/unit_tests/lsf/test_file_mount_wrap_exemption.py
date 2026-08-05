"""Backend file_mount wrap-exemption, driven by a real LsfCommandRunner.

_execute_file_mounts symlink-wraps absolute, non-``~/``/non-``/tmp/``
destinations so they can be written without cluster write access. A runner that
declares shared roots via get_unwrapped_mount_prefixes() (today only
LsfCommandRunner) makes destinations under those roots bypass the wrap and
rsync straight to the shared path.

This is the one backend line the LSF file_mounts feature changes, and it lives
in a large, shared module. Rather than edit that module, these tests drive the
real _execute_file_mounts with a real LSF runner and assert the rsync target:
unchanged for a destination under a shared root, redirected (wrapped) otherwise.
The heavy I/O (rsync fan-out, symlink command execution) is stubbed so no SSH
connection is opened.
"""

from unittest import mock

import pytest

from sky.backends import cloud_vm_ray_backend
from sky.utils import command_runner


def _lsf_runner(shared_fs_roots):
    """Build a side-effect-free LsfCommandRunner.

    Passing ``ssh_private_key=None`` skips key-file creation in the SSH base
    class, so construction opens no connection and is safe in a unit test.

    Args:
        shared_fs_roots: value forwarded to the ``shared_fs_roots`` kwarg.

    Returns:
        A constructed LsfCommandRunner.
    """
    return command_runner.LsfCommandRunner(('login-host', 22),
                                           'me',
                                           None,
                                           sky_dir='/proj/sky',
                                           skypilot_runtime_dir='/proj/rt',
                                           shared_fs_roots=shared_fs_roots)


def _run_file_mounts(tmp_path, monkeypatch, shared_fs_roots, file_mounts):
    """Drive the real _execute_file_mounts and capture per-mount rsync targets.

    Stubs the transfer fan-out (to record the resolved target instead of
    rsyncing) and the symlink-command execution (so wrapped destinations do not
    trigger a real ``runner.run`` over SSH).

    Args:
        tmp_path: pytest tmp dir fixture (holds the real source file + logs).
        monkeypatch: pytest monkeypatch fixture.
        shared_fs_roots: roots for the LsfCommandRunner under test.
        file_mounts: {dst: src} mapping passed to the backend.

    Returns:
        The list of ``target`` values in file_mounts insertion order; each is
        the destination unchanged (un-wrapped) or a wrapped safe path.
    """
    src = tmp_path / 'payload.txt'
    src.write_text('data')
    file_mounts = {dst: str(src) for dst in file_mounts}

    runner = _lsf_runner(shared_fs_roots)
    handle = mock.MagicMock()
    handle.get_command_runners.return_value = [runner]
    handle.launched_resources.cloud = 'lsf'

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


def test_shared_destination_bypasses_wrap(tmp_path, monkeypatch):
    """A dst equal to or under a shared root rsyncs straight through; a sibling
    with a shared substring and an unrelated abs path are both wrapped."""
    targets = _run_file_mounts(
        tmp_path,
        monkeypatch,
        shared_fs_roots=['/proj'],
        file_mounts=['/proj/data', '/projfoo/data', '/opt/other'])
    assert targets[0] == '/proj/data'  # under /proj -> un-wrapped
    assert targets[1] != '/projfoo/data'  # sibling substring -> wrapped
    assert targets[2] != '/opt/other'  # unrelated abs path -> wrapped


def test_exact_root_bypasses_wrap(tmp_path, monkeypatch):
    """A dst exactly equal to a shared root is exempt (not just nested paths)."""
    targets = _run_file_mounts(tmp_path,
                               monkeypatch,
                               shared_fs_roots=['/opt/share'],
                               file_mounts=['/opt/share'])
    assert targets[0] == '/opt/share'


def test_base_runner_wraps_everything(tmp_path, monkeypatch):
    """With no shared roots (base-runner default -> [] prefixes), even a
    shared-looking dst is wrapped -- behavior unchanged from before exemption."""
    targets = _run_file_mounts(tmp_path,
                               monkeypatch,
                               shared_fs_roots=None,
                               file_mounts=['/proj/data'])
    assert targets[0] != '/proj/data'
