"""Tests for retrying idempotent backend operations on ssh exit code 255."""
from unittest import mock

import pytest

from sky import exceptions
from sky.backends import cloud_vm_ray_backend as backend


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(backend.time, 'sleep', lambda _: None)


def _error(returncode):
    return exceptions.CommandError(returncode, 'cmd', 'failed', None)


def test_retries_raised_255_then_succeeds():
    run = mock.MagicMock(side_effect=[_error(255), _error(255), 'ok'])
    assert backend._retry_on_ssh_255(run, 'op') == 'ok'
    assert run.call_count == 3


def test_retries_returned_255_then_succeeds():
    run = mock.MagicMock(side_effect=[(255, '', ''), (0, 'out', '')])
    result = backend._retry_on_ssh_255(run, 'op', returncode_of=lambda r: r[0])
    assert result == (0, 'out', '')
    assert run.call_count == 2


def test_other_failures_are_not_retried():
    run = mock.MagicMock(side_effect=_error(1))
    with pytest.raises(exceptions.CommandError):
        backend._retry_on_ssh_255(run, 'op')
    assert run.call_count == 1
    run = mock.MagicMock(return_value=(1, '', ''))
    assert backend._retry_on_ssh_255(run, 'op',
                                     returncode_of=lambda r: r[0])[0] == 1
    assert run.call_count == 1


def test_gives_up_after_the_retry_budget():
    run = mock.MagicMock(side_effect=_error(255))
    with pytest.raises(exceptions.CommandError):
        backend._retry_on_ssh_255(run, 'op')
    assert run.call_count == backend._SSH_255_RETRIES + 1
    run = mock.MagicMock(return_value=(255, '', ''))
    assert backend._retry_on_ssh_255(run, 'op',
                                     returncode_of=lambda r: r[0])[0] == 255
    assert run.call_count == backend._SSH_255_RETRIES + 1


def test_can_retry_false_stops_retrying():
    run = mock.MagicMock(return_value=(255, '', ''))
    can_retry = mock.MagicMock(return_value=False)
    result = backend._retry_on_ssh_255(run,
                                       'op',
                                       returncode_of=lambda r: r[0],
                                       can_retry=can_retry)
    assert result[0] == 255
    assert run.call_count == 1
    can_retry.assert_called_once()
    run = mock.MagicMock(side_effect=_error(255))
    with pytest.raises(exceptions.CommandError):
        backend._retry_on_ssh_255(run, 'op', can_retry=lambda: False)
    assert run.call_count == 1


def _submit(monkeypatch, run_results, status, cloud=None):
    """Drive the SSH submit path of _exec_code_on_head; return run calls."""
    from sky import clouds  # pylint: disable=import-outside-toplevel
    b = backend.CloudVmRayBackend()
    b.log_dir = '/tmp/sky_logs'
    handle = mock.MagicMock()
    handle.is_grpc_enabled_with_flag = False
    handle.cluster_name = 'c'
    handle.launched_resources.cloud = cloud or clouds.AWS()
    runner = mock.MagicMock()
    runner.is_command_length_over_limit.return_value = False
    handle.get_command_runners.return_value = [runner]
    run_on_head = mock.MagicMock(side_effect=run_results)
    monkeypatch.setattr(b, 'run_on_head', run_on_head)
    get_status = mock.MagicMock(return_value={7: status})
    monkeypatch.setattr(b, 'get_job_status', get_status)
    monkeypatch.setattr(backend.controller_utils.Controllers, 'from_name',
                        lambda _: None)
    b._exec_code_on_head(handle, 'print(1)', job_id=7)
    return run_on_head, get_status


def test_queue_job_retried_while_job_still_init(monkeypatch):
    from sky.skylet import job_lib  # pylint: disable=import-outside-toplevel
    run_on_head, _ = _submit(monkeypatch, [(255, '', ''), (0, '', '')],
                             job_lib.JobStatus.INIT)
    assert run_on_head.call_count == 2


def test_queue_job_not_retried_once_job_was_queued(monkeypatch):
    from sky.skylet import job_lib  # pylint: disable=import-outside-toplevel
    with pytest.raises(exceptions.CommandError):
        _submit(monkeypatch, [(255, '', ''), (0, '', '')],
                job_lib.JobStatus.PENDING)


def test_queue_job_not_retried_on_slurm(monkeypatch):
    from sky import clouds  # pylint: disable=import-outside-toplevel
    from sky.skylet import job_lib  # pylint: disable=import-outside-toplevel
    with pytest.raises(exceptions.CommandError):
        _submit(monkeypatch, [(255, '', ''), (0, '', '')],
                job_lib.JobStatus.INIT,
                cloud=clouds.Slurm())
