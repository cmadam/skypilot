"""Tests for LsfContainerCommandRunner's shared-FS wrap-exemption plumbing.

The container runner does exactly one thing on top of the base LsfCommandRunner:
it records the shared network-filesystem roots (bind-mounted identity into the
enroot container) and exposes them via ``get_unwrapped_mount_prefixes()``. The
backend's ``_execute_file_mounts`` reads that list to decide which file_mount
destinations skip the sudo-symlink-wrap. These tests lock in that contract and,
crucially, the "behavior elsewhere is unchanged" guarantee: the CommandRunner
base class defines get_unwrapped_mount_prefixes() returning ``[]``, so every
non-LSF-container runner exempts nothing. That guarantee is now structural (a
base method) rather than tested-by-absence.
"""

from sky.provision.lsf.command_runner import LsfContainerCommandRunner
from sky.utils import command_runner


def _make_container_runner(shared_fs_roots=None):
    """Build an LsfContainerCommandRunner without any SSH/key/network I/O.

    Passing ``ssh_private_key=None`` skips key-file creation in the SSH base
    class, so construction is side-effect free and safe in a unit test.

    :param shared_fs_roots: value forwarded to the ``shared_fs_roots`` kwarg
        (``None`` exercises the default).
    :returns: a constructed LsfContainerCommandRunner.
    """
    return LsfContainerCommandRunner(('login-host', 22),
                                     'me',
                                     None,
                                     sky_dir='/proj/sky',
                                     skypilot_runtime_dir='/proj/rt',
                                     shared_fs_roots=shared_fs_roots)


def _make_base_runner():
    """Build a plain LsfCommandRunner (no container wrap-exemption).

    :returns: a constructed base LsfCommandRunner.
    """
    return command_runner.LsfCommandRunner(('login-host', 22),
                                           'me',
                                           None,
                                           sky_dir='/proj/sky',
                                           skypilot_runtime_dir='/proj/rt')


class TestGetUnwrappedMountPrefixes:
    """Tests for LsfContainerCommandRunner.get_unwrapped_mount_prefixes."""

    def test_returns_configured_roots(self):
        """Roots passed at construction are returned verbatim, in order."""
        roots = ['/proj', '/opt/share']
        runner = _make_container_runner(roots)
        assert runner.get_unwrapped_mount_prefixes() == roots

    def test_default_is_empty(self):
        """No shared_fs_roots kwarg -> no exemptions (empty list)."""
        runner = LsfContainerCommandRunner(('login-host', 22),
                                           'me',
                                           None,
                                           sky_dir='/proj/sky',
                                           skypilot_runtime_dir='/proj/rt')
        assert runner.get_unwrapped_mount_prefixes() == []

    def test_none_is_empty(self):
        """Explicit shared_fs_roots=None normalizes to an empty list."""
        runner = _make_container_runner(None)
        assert runner.get_unwrapped_mount_prefixes() == []

    def test_returns_defensive_copy(self):
        """Callers cannot mutate the runner's internal roots via the getter."""
        runner = _make_container_runner(['/proj'])
        returned = runner.get_unwrapped_mount_prefixes()
        returned.append('/tampered')
        # A subsequent call is unaffected by the caller's mutation.
        assert runner.get_unwrapped_mount_prefixes() == ['/proj']

    def test_constructor_copies_input(self):
        """Mutating the caller's list after construction does not leak in."""
        roots = ['/proj']
        runner = _make_container_runner(roots)
        roots.append('/opt/share')
        assert runner.get_unwrapped_mount_prefixes() == ['/proj']


class TestBaseRunnerBehaviorUnchanged:
    """Non-container runners must exempt nothing (empty prefixes)."""

    def test_base_runner_returns_empty(self):
        """A plain LsfCommandRunner inherits the base [] (no exemptions).

        The backend calls ``runners[0].get_unwrapped_mount_prefixes()`` directly;
        for any runner that does not override it, the CommandRunner base returns
        ``[]`` so the wrap behavior is unchanged.
        """
        assert _make_base_runner().get_unwrapped_mount_prefixes() == []

    def test_base_class_defines_method(self):
        """The guarantee is structural: CommandRunner defines the method."""
        assert (command_runner.CommandRunner.get_unwrapped_mount_prefixes is
                not LsfContainerCommandRunner.get_unwrapped_mount_prefixes)
        # The base method exists and returns [] without an override.
        assert command_runner.LsfCommandRunner.get_unwrapped_mount_prefixes is (
            command_runner.CommandRunner.get_unwrapped_mount_prefixes)
