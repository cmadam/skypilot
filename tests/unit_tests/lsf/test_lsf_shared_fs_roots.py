"""Tests for LSF shared-FS wrap-exemption root derivation.

file_mounts destined for a shared, identity-bind-mounted root (e.g. /proj) must
be left un-wrapped by the backend so the payload persists at the identity path
visible inside the enroot container. These tests lock in which enroot_mounts
qualify as shared roots (_is_shared_identity_mount) and how the built-in list is
unioned with user-configured mounts (_derive_shared_fs_roots).
"""

from unittest import mock

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
    """Tests for _derive_shared_fs_roots."""

    def _patch_mounts(self, mounts):
        """Patch lsf_utils.get_enroot_mounts to return the given specs."""
        return mock.patch.object(lsf_instance.lsf_utils,
                                 'get_enroot_mounts',
                                 return_value=mounts)

    def test_none_cluster_returns_builtins_only(self):
        assert lsf_instance._derive_shared_fs_roots(None) == ['/proj',
                                                              '/opt/share']

    def test_no_configured_mounts_returns_builtins(self):
        with self._patch_mounts([]):
            assert lsf_instance._derive_shared_fs_roots('bluevela') == [
                '/proj', '/opt/share'
            ]

    def test_shared_mount_appended(self):
        with self._patch_mounts(['/gpfs /gpfs']):
            assert lsf_instance._derive_shared_fs_roots('bluevela') == [
                '/proj', '/opt/share', '/gpfs'
            ]

    def test_node_local_mounts_filtered_out(self):
        with self._patch_mounts(
            ['/dev/shm /dev/shm', '/dev/infiniband /dev/infiniband']):
            assert lsf_instance._derive_shared_fs_roots('bluevela') == [
                '/proj', '/opt/share'
            ]

    def test_mixed_mounts(self):
        with self._patch_mounts([
                '/dev/shm /dev/shm', '/gpfs /gpfs', '/proj /proj', '/tmp /tmp'
        ]):
            # /gpfs added; /dev/shm and /tmp filtered; /proj de-duped.
            assert lsf_instance._derive_shared_fs_roots('bluevela') == [
                '/proj', '/opt/share', '/gpfs'
            ]

    def test_duplicate_builtin_deduped(self):
        with self._patch_mounts(['/proj /proj', '/opt/share /opt/share']):
            assert lsf_instance._derive_shared_fs_roots('bluevela') == [
                '/proj', '/opt/share'
            ]

    def test_returns_new_list_not_builtin(self):
        """Result must not alias the module-level _SHARED_FS_ROOTS."""
        with self._patch_mounts([]):
            result = lsf_instance._derive_shared_fs_roots('bluevela')
        assert result is not lsf_instance._SHARED_FS_ROOTS
        result.append('/mutated')
        assert '/mutated' not in lsf_instance._SHARED_FS_ROOTS
