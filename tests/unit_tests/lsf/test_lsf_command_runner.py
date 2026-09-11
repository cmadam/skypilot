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

    @pytest.mark.parametrize(
        'kwargs',
        [
            {},  # kwarg omitted -> default None
            {
                'shared_fs_roots': None
            },  # explicit None
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


class TestNodeIdentity:
    """Runners carry which allocated node they correspond to.

    get_cluster_info has always set job_id/node/rank tags, but the runner ignored
    them, so every runner of a multi-node cluster was byte-identical — pointed at
    the login node with nothing to distinguish it. "Do this on node 3" was not
    expressible. This mirrors SlurmCommandRunner, which takes job_id and
    slurm_node for exactly this reason.
    """

    def _runner(self, **kwargs):
        from sky.utils import command_runner
        params = dict(
            node=('login4.example.com', 22),
            ssh_user='granitebuild',
            ssh_private_key='/dev/null',
            sky_dir='/proj/builds/cluster',
            skypilot_runtime_dir='/tmp/rt',
        )
        params.update(kwargs)
        return command_runner.LsfCommandRunner(**params)

    def test_identity_is_recorded(self):
        r = self._runner(job_id='12345', lsf_node='p1-r08-n4', node_rank=1)
        assert (r.job_id, r.lsf_node, r.node_rank) == ('12345', 'p1-r08-n4', 1)

    def test_identity_is_optional(self):
        """A runner built without tags must still work."""
        r = self._runner()
        assert r.job_id is None and r.lsf_node is None and r.node_rank is None

    def test_runners_for_different_nodes_are_distinguishable(self):
        a = self._runner(lsf_node='p1-r08-n4', node_rank=0)
        b = self._runner(lsf_node='p4-r22-n1', node_rank=1)
        assert a.lsf_node != b.lsf_node


class TestCommandsRunOnTheLoginNode:
    """The dispatch path is gone from the runner, deliberately.

    It never executed — the enabling check read a nested `enroot.enabled` key while
    the template writes a flat `enroot_enabled`, so dispatch_dir was always None —
    and its accidental absence was the correct behaviour. What the backend sends
    through a runner is cluster setup: the SkyPilot runtime install, internal file
    mounts, the logging agent, all writing into the shared-filesystem cluster home
    that every node reads. Dispatching those to a compute node would install the
    runtime inside a container that is then discarded, on a filesystem the next
    node cannot see.

    Removing it replaces an accidental disablement with a deliberate absence, so a
    future reader "fixing" the key typo cannot silently reroute runtime setup into
    the training container. The dispatch protocol itself is implemented, properly
    and per node, in sky.skylet.executor.lsf.
    """

    def test_runner_has_no_dispatch_machinery(self):
        from sky.utils import command_runner
        assert not hasattr(command_runner.LsfCommandRunner,
                           '_wrap_with_dispatch')

    def test_dispatch_dir_is_not_an_accepted_argument(self):
        """Passing it should fail loudly rather than be silently ignored."""
        import inspect

        from sky.utils import command_runner
        params = inspect.signature(
            command_runner.LsfCommandRunner.__init__).parameters
        assert 'dispatch_dir' not in params

    def test_rsync_still_targets_the_login_node(self):
        """The premise of get_unwrapped_mount_prefixes: file_mounts are written on
        the login node to a shared root, where the job sees them at the identical
        path. Routing transfers anywhere else breaks that identity mapping."""
        from sky.utils import command_runner
        source = inspect_source(command_runner.LsfCommandRunner.rsync)
        assert 'rsync_path' in source or '_rsync_with_path' in source


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)
