"""Tests for the atomic per-hash wheel refresh in sky.backends.wheel_utils."""
import os
from unittest import mock

import pytest

from sky.backends import wheel_utils


def test_replace_keeps_old_inode_for_readers(tmp_path):
    """The new wheel is renamed into place, never rewritten in place."""
    src = tmp_path / 'src' / 'skypilot-1.0-py3-none-any.whl'
    src.parent.mkdir()
    src.write_bytes(b'new wheel')
    dst_dir = tmp_path / 'dst'
    dst_dir.mkdir()
    dst = dst_dir / src.name
    dst.write_bytes(b'old wheel')
    # A reader that opened the old wheel keeps reading it whole.
    with open(dst, 'rb') as reader:
        wheel_utils._replace_file_atomically(src, dst_dir)
        assert reader.read() == b'old wheel'
    assert dst.read_bytes() == b'new wheel'
    assert sorted(p.name for p in dst_dir.iterdir()) == [src.name]


def test_failed_copy_leaves_existing_wheel(tmp_path):
    src = tmp_path / 'skypilot-1.0-py3-none-any.whl'
    src.write_bytes(b'new wheel')
    dst_dir = tmp_path / 'dst'
    dst_dir.mkdir()
    dst = dst_dir / src.name
    dst.write_bytes(b'old wheel')

    def partial_copy(_src, tmp_path_arg):
        with open(tmp_path_arg, 'wb') as f:
            f.write(b'new')
        raise OSError('disk full')

    with mock.patch.object(wheel_utils.shutil, 'copy',
                           side_effect=partial_copy):
        with pytest.raises(OSError):
            wheel_utils._replace_file_atomically(src, dst_dir)
    assert dst.read_bytes() == b'old wheel'
    assert sorted(os.listdir(dst_dir)) == [src.name]
