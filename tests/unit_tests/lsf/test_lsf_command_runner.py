"""Tests for LsfCommandRunner's shared-FS wrap-exemption plumbing.

Locks in the get_unwrapped_mount_prefixes() contract (see that method for the
rationale): roots passed at construction are surfaced for the backend's
file_mount wrap-exemption, and a runner with no roots — like the CommandRunner
base — exempts nothing.
"""

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
        """Roots passed at construction are returned verbatim, in order."""
        roots = ['/proj', '/opt/share']
        runner = _make_runner(roots)
        assert runner.get_unwrapped_mount_prefixes() == roots

    def test_default_is_empty(self):
        """No shared_fs_roots kwarg -> no exemptions (empty list)."""
        runner = command_runner.LsfCommandRunner(
            ('login-host', 22),
            'me',
            None,
            sky_dir='/proj/sky',
            skypilot_runtime_dir='/proj/rt')
        assert runner.get_unwrapped_mount_prefixes() == []

    def test_none_is_empty(self):
        """Explicit shared_fs_roots=None normalizes to an empty list."""
        runner = _make_runner(None)
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
    """A runner given no shared roots must exempt nothing (empty prefixes)."""

    def test_lsf_runner_without_roots_returns_empty(self):
        """An LSF runner constructed with no roots inherits the [] contract.

        The backend calls ``runners[0].get_unwrapped_mount_prefixes()`` directly;
        with no shared_fs_roots the LSF runner returns ``[]`` so the wrap
        behavior is unchanged.
        """
        assert _make_runner().get_unwrapped_mount_prefixes() == []

    def test_base_class_returns_empty(self):
        """The base CommandRunner exempts nothing (the structural default)."""
        base = command_runner.CommandRunner(('login-host', 22))
        assert base.get_unwrapped_mount_prefixes() == []

    def test_lsf_overrides_base_method(self):
        """LsfCommandRunner overrides the base to surface its shared roots."""
        assert (command_runner.LsfCommandRunner.get_unwrapped_mount_prefixes is
                not command_runner.CommandRunner.get_unwrapped_mount_prefixes)
