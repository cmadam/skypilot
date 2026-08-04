"""Tests for LSF shared-FS wrap-exemption root derivation.

file_mounts destined for a shared, identity-bind-mounted root (e.g. /proj) must
be left un-wrapped by the backend so the payload persists at the identity path
visible inside the enroot container. These tests lock in which enroot_mounts
qualify as shared roots (_is_shared_identity_mount) and how the built-in list is
unioned with user-configured mounts (_derive_shared_fs_roots).
"""

import pytest

from sky.provision.lsf import instance as lsf_instance


class TestIsSharedIdentityMount:
    """Tests for the _is_shared_identity_mount predicate."""

    @pytest.mark.parametrize('spec', [
        '/gpfs /gpfs',
        '/proj /proj',
        '/gpfs /gpfs x-create=dir',  # trailing mount flags ignored
        '/opt/share /opt/share',
    ])
    def test_shared_identity_mounts(self, spec):
        assert lsf_instance._is_shared_identity_mount(spec) is True

    @pytest.mark.parametrize('spec', [
        '/dev/shm /dev/shm',  # node-local device
        '/dev/infiniband /dev/infiniband',  # node-local device
        '/tmp /tmp',  # node-local scratch
        '/opt/nvme/$USER /opt/nvme/$USER',  # under node-local prefix
        '/run/x /run/x',
        '/var/tmp/x /var/tmp/x',
    ])
    def test_node_local_mounts_excluded(self, spec):
        assert lsf_instance._is_shared_identity_mount(spec) is False

    @pytest.mark.parametrize('spec', [
        '/host /container',  # non-identity: host != container
        '/onlyone',  # malformed: single field
        '',  # empty
        'relative relative',  # not absolute
    ])
    def test_malformed_or_non_identity_excluded(self, spec):
        assert lsf_instance._is_shared_identity_mount(spec) is False

    def test_prefix_boundary_not_substring(self):
        """/tmpfoo is not under /tmp — must not be excluded as node-local."""
        assert lsf_instance._is_shared_identity_mount('/tmpfoo /tmpfoo') is True


class TestDeriveSharedFsRoots:
    """Tests for _derive_shared_fs_roots.

    The input is the enroot mount-spec list frozen into the container at
    provision time (provider_config['enroot_mounts']) — the same source
    _build_enroot_block uses — so the exemption can never claim a root the
    container does not actually mount.
    """

    def test_no_mounts_returns_builtins_only(self):
        assert lsf_instance._derive_shared_fs_roots([]) == [
            '/proj', '/opt/share'
        ]

    def test_shared_mount_appended(self):
        assert lsf_instance._derive_shared_fs_roots(['/gpfs /gpfs']) == [
            '/proj', '/opt/share', '/gpfs'
        ]

    def test_node_local_mounts_filtered_out(self):
        assert lsf_instance._derive_shared_fs_roots(
            ['/dev/shm /dev/shm', '/dev/infiniband /dev/infiniband']) == [
                '/proj', '/opt/share'
            ]

    def test_mixed_mounts(self):
        # /gpfs added; /dev/shm and /tmp filtered; /proj de-duped.
        assert lsf_instance._derive_shared_fs_roots(
            ['/dev/shm /dev/shm', '/gpfs /gpfs', '/proj /proj', '/tmp /tmp']) == [
                '/proj', '/opt/share', '/gpfs'
            ]

    def test_duplicate_builtin_deduped(self):
        assert lsf_instance._derive_shared_fs_roots(
            ['/proj /proj', '/opt/share /opt/share']) == ['/proj', '/opt/share']

    def test_returns_new_list_not_builtin(self):
        """Result must not alias the module-level _SHARED_FS_ROOTS."""
        result = lsf_instance._derive_shared_fs_roots([])
        assert result is not lsf_instance._SHARED_FS_ROOTS
        result.append('/mutated')
        assert '/mutated' not in lsf_instance._SHARED_FS_ROOTS
