"""Tests for LsfContainerCommandRunner's shared-FS wrap-exemption plumbing.

The container runner does exactly one thing on top of the base LsfCommandRunner:
it records the shared network-filesystem roots (bind-mounted identity into the
enroot container) and exposes them via ``get_unwrapped_mount_prefixes()``. The
backend's ``_execute_file_mounts`` reads that list to decide which file_mount
destinations skip the sudo-symlink-wrap. These tests lock in that contract and,
crucially, the "behavior elsewhere is unchanged" guarantee: the base runner does
not expose the method, so the backend's ``getattr(..., lambda: [])`` fallback
yields ``[]`` for every non-LSF-container runner.
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
    """The exemption hook must be absent on non-container runners."""

    def test_base_runner_lacks_method(self):
        """A plain LsfCommandRunner does not expose the exemption hook."""
        assert not hasattr(_make_base_runner(),
                           'get_unwrapped_mount_prefixes')

    def test_backend_getattr_fallback_yields_empty(self):
        """The backend's getattr fallback returns [] for a base runner.

        Mirrors the exact call the backend makes in _execute_file_mounts, so a
        non-LSF-container runner produces no exemptions and the wrap behavior is
        unchanged.
        """
        base = _make_base_runner()
        prefixes = getattr(base, 'get_unwrapped_mount_prefixes',
                           lambda: [])()
        assert prefixes == []
