"""Tests for the goofys install command in sky.data.mounting_utils."""
import os
import subprocess

from sky.data import mounting_utils


def test_goofys_install_replaces_existing_empty_binary(tmp_path):
    """A 0-byte goofys from an earlier attempt is downloaded again."""
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    goofys = tmp_path / 'goofys'
    goofys.write_bytes(b'')
    # Behaves like wget: with -nc, an existing output file is never
    # replaced. Otherwise -O gets the full download.
    (fake_bin / 'wget').write_text('#!/bin/bash\n'
                                   'out=""; nc=false\n'
                                   'while [ $# -gt 0 ]; do\n'
                                   '  case "$1" in\n'
                                   '    -nc) nc=true ;;\n'
                                   '    -O) shift; out="$1" ;;\n'
                                   '  esac\n'
                                   '  shift\n'
                                   'done\n'
                                   'if $nc && [ -e "$out" ]; then exit 1; fi\n'
                                   'echo goofys-binary > "$out"\n')
    (fake_bin / 'sudo').write_text('#!/bin/bash\n"$@"\n')
    (fake_bin / 'uname').write_text('#!/bin/bash\necho x86_64\n')
    for tool in ('wget', 'sudo', 'uname'):
        os.chmod(fake_bin / tool, 0o755)
    cmd = mounting_utils.get_s3_mount_install_cmd().replace(
        '/usr/local/bin/goofys', str(goofys))

    subprocess.run(['bash', '-c', cmd],
                   env={
                       **os.environ, 'PATH': f'{fake_bin}:{os.environ["PATH"]}'
                   },
                   check=True)

    assert goofys.read_text() == 'goofys-binary\n'
