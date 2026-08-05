"""Tests for LsfCommandRunner's shared-FS wrap-exemption plumbing.

Locks in the get_unwrapped_mount_prefixes() contract (see that method for the
rationale): roots passed at construction are surfaced for the backend's
file_mount wrap-exemption, and a runner with no roots — like the CommandRunner
base — exempts nothing.
"""

import pytest

from sky.utils import command_runner


def _make_runner(shared_fs_roots=None):
    """Build an LsfCommandRunner without any SSH/key/network I/O.

    Passing ``ssh_private_key=None`` skips key-file creation in the SSH base
    class, so construction is side-effect free and safe in a unit test.

    Args:
        shared_fs_roots: value forwarded to the ``shared_fs_roots`` kwarg
            (``None`` exercises the default).

    Returns:
        A constructed LsfCommandRunner.
    """
    return command_runner.LsfCommandRunner(('login-host', 22),
                                           'me',
                                           None,
                                           sky_dir='/proj/sky',
                                           skypilot_runtime_dir='/proj/rt',
                                           shared_fs_roots=shared_fs_roots)


class TestGetUnwrappedMountPrefixes:
    """Tests for LsfCommandRunner.get_unwrapped_mount_prefixes."""

    def test_returns_configured_roots(self):
        """Roots passed at construction are returned verbatim, in order.

        Also the behavioral guard that the LSF override is present: if it were
        dropped, the runner would inherit the base ``[]`` and this would fail.
        """
        roots = ['/proj', '/opt/share']
        runner = _make_runner(roots)
        assert runner.get_unwrapped_mount_prefixes() == roots

    @pytest.mark.parametrize('kwargs', [
        {},  # kwarg omitted -> default None
        {'shared_fs_roots': None},  # explicit None
    ])
    def test_no_roots_is_empty(self, kwargs):
        """Both an omitted kwarg and an explicit ``None`` normalize to no
        exemptions, so the LSF runner matches the base CommandRunner and the
        backend (which reads ``runners[0]``) leaves every mount wrapped.
        """
        runner = command_runner.LsfCommandRunner(
            ('login-host', 22),
            'me',
            None,
            sky_dir='/proj/sky',
            skypilot_runtime_dir='/proj/rt',
            **kwargs)
        assert runner.get_unwrapped_mount_prefixes() == []

    def test_returns_defensive_copy(self):
        """Callers cannot mutate the runner's internal roots via the getter."""
        runner = _make_runner(['/proj'])
        returned = runner.get_unwrapped_mount_prefixes()
        returned.append('/tampered')
        # A subsequent call is unaffected by the caller's mutation.
        assert runner.get_unwrapped_mount_prefixes() == ['/proj']

    def test_constructor_copies_input(self):
        """Mutating the caller's list after construction does not leak in."""
        roots = ['/proj']
        runner = _make_runner(roots)
        roots.append('/opt/share')
        assert runner.get_unwrapped_mount_prefixes() == ['/proj']


class TestBaseRunnerBehaviorUnchanged:
    """The base CommandRunner exempts nothing; the LSF empty-roots case above
    inherits the same ``[]`` contract, so non-LSF clouds are unaffected.
    """

    def test_base_class_returns_empty(self):
        """The base CommandRunner exempts nothing (the structural default the
        LSF override builds on)."""
        base = command_runner.CommandRunner(('login-host', 22))
        assert base.get_unwrapped_mount_prefixes() == []
