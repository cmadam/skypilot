"""Snapshot tests for the LSF provisioning bsub script.

``_build_bsub_script`` is a pure function of ``provider_config`` and
``num_nodes``, so its full text can be frozen as a golden file. That makes it the
review artifact for changes to the provisioning script: a commit that intends to
change only multi-node behavior must leave the single-node goldens untouched.

Snapshots live in ``testdata/lsf_bsub/*.sh``. To update them after an
intentional change:

    UPDATE_SNAPSHOT=1 pytest tests/unit_tests/lsf/test_lsf_bsub_script.py

Then read ``git diff testdata/`` as part of the change.
"""

import difflib
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from sky.provision.lsf import instance as lsf_instance

BSUB_TESTDATA_DIR = Path(__file__).parent / 'testdata' / 'lsf_bsub'

CLUSTER_NAME = 'sky-gold-kd-abc123'


def assert_bsub_matches_snapshot(test_name: str, script: str) -> None:
    """Compare a generated bsub script against its snapshot file."""
    snapshot_path = BSUB_TESTDATA_DIR / f'{test_name}.sh'

    if os.environ.get('UPDATE_SNAPSHOT') == '1':
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(script)
        print(f'Updated snapshot: {snapshot_path}')
        return

    if not snapshot_path.exists():
        pytest.fail(f'Snapshot file not found: {snapshot_path}\n'
                    f'Run with UPDATE_SNAPSHOT=1 to create it.')

    expected = snapshot_path.read_text()
    if script != expected:
        diff = ''.join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                script.splitlines(keepends=True),
                fromfile=f'{test_name}.sh (expected)',
                tofile=f'{test_name}.sh (generated)',
            ))
        pytest.fail(f'Generated bsub script does not match snapshot:\n{diff}\n'
                    f'If intentional, re-run with UPDATE_SNAPSHOT=1.')


def _base_provider_config() -> Dict[str, Any]:
    """A bare (no-container) BlueVela-shaped provider config."""
    return {
        'cluster': 'bluevela',
        'cpus': '4',
        'memory': '64',
        'accelerator_count': '0',
        'workdir': '/proj/granite-build/g4os/skypilot',
        'tmpdir': '/opt/nvme/$USER/skypilot-tmp',
    }


def _enroot_provider_config() -> Dict[str, Any]:
    """A containerized BlueVela-shaped provider config.

    Mirrors configurations/assets/environments/skypilot/lsf/ibm-bluevela/
    environment.yaml as the fields actually reach provider_config today.
    """
    config = _base_provider_config()
    config.update({
        'accelerator_count': '8',
        'accelerator_type': 'H100',
        'image_id': ('docker:us.icr.io/cil15-shared-registry/'
                     'kd-sandbox-distill:0.1.0-uv'),
        'enroot_enabled': 'True',
        'enroot_share_path': '/proj/granite-build/g4os',
        'enroot_use_local_nvme': 'True',
        'enroot_squash_options': '-comp lz4 -Xhc -no-xattrs',
        'nccl_tuning_file': '/proj/granite-build/g4os/bv-nccl-tuning.sh',
        'bsub_options': {
            'G': 'grp_granite_dot_build',
            'M': '64G',
        },
    })
    return config


class TestBsubScriptSnapshots:
    """Golden-file coverage of the provisioning script across node counts."""

    def test_single_node_bare(self):
        """No container: the script tail is a bare `sleep infinity`."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _base_provider_config(),
                                                num_nodes=1)
        assert_bsub_matches_snapshot('single_node_bare', script)

    def test_single_node_enroot(self):
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _enroot_provider_config(),
                                                num_nodes=1)
        assert_bsub_matches_snapshot('single_node_enroot', script)

    def test_two_node_enroot(self):
        """The GOLD-distillation shape: 2 nodes x 8 GPUs, containerized."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _enroot_provider_config(),
                                                num_nodes=2)
        assert_bsub_matches_snapshot('two_node_enroot', script)

    def test_four_node_enroot_cpus(self):
        """4 nodes with cpus > 1.

        Records that `cpus` is currently dropped: `-n` carries the node count
        alone, so a request for 8 CPUs per node is not represented in any
        directive. Fixing that must change this snapshot and no single-node one.
        """
        config = _enroot_provider_config()
        config['cpus'] = '8'
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                config,
                                                num_nodes=4)
        assert_bsub_matches_snapshot('four_node_enroot_cpus', script)


class TestBsubScriptInvariants:
    """Assertions that must hold regardless of how the goldens evolve.

    These are the properties whose regression would be silent in a diff: a
    reviewer scanning a large golden change will not notice a lost `span[ptile]`
    or a `bash -lc` creeping in, and neither failure surfaces until a live
    multi-node run misbehaves.
    """

    def test_multinode_pins_one_task_per_host(self):
        """`-n N` means slots, not hosts.

        Without `span[ptile=1]` LSF may satisfy `-n 4` with four slots on one
        host, silently collapsing a 4-node job onto a single machine.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _enroot_provider_config(),
                                                num_nodes=4)
        assert '#BSUB -n 4' in script
        assert '#BSUB -R "span[ptile=1]"' in script
        assert '#BSUB -hl' in script

    def test_single_node_requests_no_span(self):
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _enroot_provider_config(),
                                                num_nodes=1)
        assert '#BSUB -n 1' in script
        assert 'span[ptile' not in script

    def test_dispatcher_uses_non_login_shell(self):
        """The container venv is on PATH only in a non-login shell.

        A login shell re-runs /etc/profile and drops /stage/.venv/bin, so a
        `bash -lc` here would break every step that relies on the image's venv.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                _enroot_provider_config(),
                                                num_nodes=2)
        assert 'bash -lc' not in script

    def test_queue_options_override_cluster_options(self):
        config = _enroot_provider_config()
        config['queue'] = 'preemptable'
        config['queue_configs'] = {
            'preemptable': {
                'bsub_options': {
                    'G': 'grp_preemptable'
                }
            }
        }
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                config,
                                                num_nodes=2)
        assert '#BSUB -q preemptable' in script
        assert '#BSUB -G grp_preemptable' in script
        assert '#BSUB -G grp_granite_dot_build' not in script
