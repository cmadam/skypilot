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
import subprocess
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

    def test_two_node_enroot_one_cpu(self):
        """One slot per host: the directives must match the pre-fix form.

        This is the shape every existing cluster ran, so it is the regression
        guard for the cpus fix — `-n` equals the node count, ptile is 1, and the
        GPU request keeps its bare per-task form.
        """
        config = _enroot_provider_config()
        config['cpus'] = '1'
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config,
                                                 num_nodes=2)
        assert_bsub_matches_snapshot('two_node_enroot_one_cpu', script)

    def test_four_node_enroot_cpus(self):
        """4 nodes with cpus > 1: 32 slots, 8 per host."""
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

    def test_multinode_span_makes_the_node_count_real(self):
        """`-n` means slots, not hosts.

        Without a `span[ptile]` term LSF may satisfy the slot count from any mix
        of hosts, silently collapsing a 4-node job onto fewer machines. The
        fixture requests 4 CPUs per node, so 4 nodes is 16 slots at 4 per host.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=4)
        assert '#BSUB -n 16' in script
        assert '#BSUB -R "span[ptile=4]"' in script
        assert '#BSUB -hl' in script

    def test_single_node_multi_cpu_is_still_pinned_to_one_host(self):
        """The span term matters at one node too: `-n 4` without it could be
        satisfied by four separate hosts."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=1)
        assert '#BSUB -n 4' in script
        assert '#BSUB -R "span[ptile=4]"' in script
        assert '#BSUB -hl' not in script

    def test_single_slot_job_needs_no_span(self):
        config = _enroot_provider_config()
        config['cpus'] = '1'
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config,
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


class TestPerHostDispatch:
    """Each node must own its dispatch directory.

    A single shared directory cannot work for multi-node: every blaunch task runs
    the same block, so each would `rm -rf` a directory its peers are using, and
    whichever dispatcher noticed a cmd_*.sh first would run it.
    """

    def test_dispatch_dir_is_per_host(self):
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'DISPATCH_DIR="' in script
        assert '/.sky/dispatch/$(hostname -s)"' in script

    def test_shared_dispatch_root_is_never_removed(self):
        """The `rm -rf` must target the per-host dir, never its parent."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=4)
        assert 'rm -rf "$DISPATCH_DIR"' in script
        assert 'rm -rf "/proj/granite-build/g4os/skypilot/'\
               f'{CLUSTER_NAME}/.sky/dispatch"' not in script

    def test_blaunch_targets_deduplicated_hosts(self):
        """`blaunch -z <hosts>` launches once per host, not once per slot."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'blaunch -z "${UNIQUE_HOSTS[*]}" bash "$SHARED_SCRIPT"' in script

    def test_unique_hosts_is_defined_before_blaunch_uses_it(self):
        """Ordering: topology computes UNIQUE_HOSTS, blaunch consumes it.

        Both live in separately-built blocks, so nothing but their assembly order
        in _build_bsub_script keeps this sound — and getting it wrong would
        expand to an empty host list, which blaunch accepts.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert script.index('UNIQUE_HOSTS=($(echo "$LSB_HOSTS"') < script.index(
            'blaunch -z "${UNIQUE_HOSTS[*]}"')


class TestTopologyBlock:
    """Topology is computed for every job, containerized or not."""

    def test_bare_metal_gets_topology(self):
        """A bare-metal job needs rank/master too, and must publish a manifest.

        Before this, topology was emitted only when enroot was enabled, so a
        non-container multi-node job had no rank at all.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _base_provider_config(),
                                                 num_nodes=2)
        assert '=== Compute topology ===' in script
        assert 'export MASTER_ADDR="$MASTER_HOST"' in script

    def test_rank_manifest_is_published_by_the_job(self):
        """The job records host -> rank; the driver must not re-derive it.

        Driver-side rank (from `bjobs -o EXEC_HOST` order) and in-job rank (from
        $LSB_HOSTS order) are independent derivations that LSF does not promise
        to agree. A mismatched rank/master pairing does not fail fast: it hangs
        NCCL init until the collective timeout.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'TOPOLOGY_DIR="' in script
        assert 'echo "$LOCAL_HOST" > "$TOPOLOGY_DIR/rank-$RANK"' in script
        assert '$TOPOLOGY_DIR/master' in script

    def test_single_node_skips_rank_detection(self):
        """With one host there is nothing to search; rank is 0 by definition."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=1)
        assert 'UNIQUE_HOSTS' not in script
        assert 'RANK=0' in script

    def test_master_port_is_derived_from_job_id(self):
        """Two concurrent jobs must not pick the same rendezvous port."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'MASTER_PORT=$((29500 + (${LSB_JOBID:-0} % 1000)))' in script


class TestReadySignals:
    """Readiness is per node, so the driver can wait for the whole allocation."""

    def test_each_node_signals_separately(self):
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'touch "/proj/granite-build/g4os/skypilot/'\
               f'{CLUSTER_NAME}/.sky_ready.$(hostname -s)"' in script

    def test_rank_zero_also_writes_the_legacy_shared_signal(self):
        """Kept so a driver predating per-host signals still sees readiness."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'if [[ "${RANK:-0}" == "0" ]]; then' in script

    def test_stale_signal_removal_is_scoped_to_this_host(self):
        """A worker must not delete a peer's fresh signal on startup."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert 'rm -f "/proj/granite-build/g4os/skypilot/'\
               f'{CLUSTER_NAME}/.sky_ready.$(hostname -s)"' in script


class _StubRunner:
    """Records `test -f` probes and answers from a set of existing paths."""

    def __init__(self, existing):
        self.existing = set(existing)
        self.commands = []

    def run(self, cmd, **kwargs):
        self.commands.append(cmd)
        # The waiter ANDs one `test -f <path>` per expected node.
        paths = [
            part.split('test -f ')[1].strip().strip("'")
            for part in cmd.split('&&')
            if 'test -f' in part
        ]
        rc = 0 if paths and all(p in self.existing for p in paths) else 1
        return rc, '', ''


class TestWaitForAllReadySignals:
    """The driver must wait for every node, not for the fastest one."""

    READY = '/proj/builds/cluster/.sky_ready'

    def test_returns_when_all_nodes_signalled(self):
        runner = _StubRunner([f'{self.READY}.host1', f'{self.READY}.host2'])
        lsf_instance._wait_for_all_ready_signals(runner,
                                                 self.READY,
                                                 'cluster', ['host1', 'host2'],
                                                 timeout=5)

    def test_times_out_when_one_node_is_missing(self):
        """And names the node that never signalled.

        With a multi-node job the useful question is not "did it time out" but
        "which host is stuck", so the message must identify it.
        """
        runner = _StubRunner([f'{self.READY}.host1'])
        with pytest.raises(TimeoutError, match='host2'):
            lsf_instance._wait_for_all_ready_signals(runner,
                                                     self.READY,
                                                     'cluster',
                                                     ['host1', 'host2'],
                                                     timeout=1)

    def test_one_ready_node_does_not_satisfy_a_two_node_job(self):
        """The regression this guards: a shared signal returned on first touch."""
        runner = _StubRunner([f'{self.READY}.host1'])
        with pytest.raises(TimeoutError):
            lsf_instance._wait_for_all_ready_signals(runner,
                                                     self.READY,
                                                     'cluster',
                                                     ['host1', 'host2'],
                                                     timeout=1)

    def test_fqdn_from_lsf_matches_short_name_written_by_the_job(self):
        """LSF may report p1-r08-n4.bluevela.example.com; the job writes the
        short name (`hostname -s`), so the comparison uses the first label."""
        runner = _StubRunner([f'{self.READY}.host1', f'{self.READY}.host2'])
        lsf_instance._wait_for_all_ready_signals(
            runner,
            self.READY,
            'cluster', ['host1.bluevela.example.com', 'host2.example.com'],
            timeout=5)


class TestScriptIsColumnZero:
    """Three things in this script are only meaningful at column 0.

    The template is interpolated from separately-built blocks that each start at
    column 0, which makes textwrap.dedent a silent no-op over the whole
    template — so this was previously emitting an indented `#!` line and an
    indented first `#BSUB` directive. Nothing failed: `submit_job` passes
    `bsub -J <name> -q <queue> < script`, so the ignored in-script `-J`/`-q` were
    already supplied on the command line, and every other directive happened to
    be on an interpolated (column-0) line. The `#!` line was simply unused, with
    LSF falling back to the submitting user's login shell for a script full of
    bashisms.
    """

    ALL_CASES = [
        ('single_node_bare', _base_provider_config, 1),
        ('single_node_enroot', _enroot_provider_config, 1),
        ('two_node_enroot', _enroot_provider_config, 2),
        ('four_node_enroot', _enroot_provider_config, 4),
    ]

    @pytest.mark.parametrize('name,config_fn,num_nodes', ALL_CASES)
    def test_shebang_is_the_first_line(self, name, config_fn, num_nodes):
        del name  # only for test ids
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config_fn(),
                                                 num_nodes=num_nodes)
        assert script.startswith('#!/bin/bash\n'), (
            'an indented #! is a comment, not a shebang, leaving the '
            "interpreter to the submitting user's login shell")

    @pytest.mark.parametrize('name,config_fn,num_nodes', ALL_CASES)
    def test_every_bsub_directive_is_unindented(self, name, config_fn,
                                                num_nodes):
        del name
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config_fn(),
                                                 num_nodes=num_nodes)
        indented = [
            line for line in script.split('\n') if
            line.lstrip().startswith('#BSUB') and not line.startswith('#BSUB')
        ]
        assert not indented, f'LSF ignores indented directives: {indented}'

    @pytest.mark.parametrize('name,config_fn,num_nodes', ALL_CASES)
    def test_script_is_valid_bash(self, name, config_fn, num_nodes):
        """`bash -n` the whole script.

        Cheap insurance for a file assembled from four independently-built
        heredoc-bearing blocks, where a terminator drifting off column 0 would
        swallow the rest of the script as heredoc body.
        """
        del name
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config_fn(),
                                                 num_nodes=num_nodes)
        result = subprocess.run(['bash', '-n'],
                                input=script,
                                capture_output=True,
                                text=True,
                                check=False)
        assert result.returncode == 0, f'bash -n failed: {result.stderr}'

    def test_heredoc_terminators_are_unindented(self):
        """The block builders emit `<< 'EOF'`, not `<<-`, so no leading tabs."""
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        for terminator in ('DISPATCH_EOF', 'ENROOT_CFG_STATIC',
                           'ENROOT_CFG_DYNAMIC', 'WRAPPER'):
            assert f'\n{terminator}\n' in script, (
                f'{terminator} is not at column 0')

    def test_in_script_job_name_matches_the_command_line_one(self):
        """`#BSUB -J` becomes effective now that it is unindented.

        submit_job also passes `-J <cluster_name_on_cloud>`; the command line
        wins, and both carry the same value, so nothing changes behaviorally.
        This pins that agreement rather than leaving it to coincidence.
        """
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 _enroot_provider_config(),
                                                 num_nodes=2)
        assert f'#BSUB -J {CLUSTER_NAME}\n' in script


class TestSlotsAndDistribution:
    """-n counts slots; the node count is only real because of span[ptile]."""

    def _script(self, cpus, num_nodes, acc_count='8'):
        config = _enroot_provider_config()
        config['cpus'] = cpus
        config['accelerator_count'] = acc_count
        return lsf_instance._build_bsub_script(CLUSTER_NAME,
                                               config,
                                               num_nodes=num_nodes)

    def test_slots_are_nodes_times_cpus(self):
        """cpus was read and then never used in any directive, so every node got
        one slot regardless of what was requested."""
        assert '#BSUB -n 16' in self._script('8', 2)
        assert '#BSUB -R "span[ptile=8]"' in self._script('8', 2)

    def test_one_cpu_per_node_keeps_the_previous_directives(self):
        script = self._script('1', 4)
        assert '#BSUB -n 4' in script
        assert '#BSUB -R "span[ptile=1]"' in script

    def test_single_node_multi_cpu_still_pins_to_one_host(self):
        """Without ptile, `-n 8` on a 1-node request could spread over 8 hosts,
        which would quietly break the single-node assumption."""
        script = self._script('8', 1)
        assert '#BSUB -n 8' in script
        assert '#BSUB -R "span[ptile=8]"' in script

    def test_single_node_single_cpu_needs_no_span(self):
        script = self._script('1', 1)
        assert '#BSUB -n 1' in script
        assert 'span[ptile' not in script

    def test_host_level_limits_only_for_multi_node(self):
        assert '#BSUB -hl' in self._script('8', 2)
        assert '#BSUB -hl' not in self._script('8', 1)

    @pytest.mark.parametrize('cpus', ['4.0', '4', 4])
    def test_cpus_accepts_the_forms_skypilot_passes(self, cpus):
        assert '#BSUB -n 8' in self._script(cpus, 2)

    @pytest.mark.parametrize('cpus', ['', None, 'many', '0', '-2'])
    def test_unparseable_or_nonpositive_cpus_falls_back_to_one_slot(self, cpus):
        """Fail soft: a job that schedules with one slot per node beats a
        provisioning crash, and the fallback is logged."""
        script = self._script(cpus, 2)
        assert '#BSUB -n 2' in script
        assert '#BSUB -R "span[ptile=1]"' in script


class TestGpuRequestIsPerHost:
    """The GPU count must not scale with slots-per-host.

    `-gpu num=` is per *task* by default and a task is a slot, so once ptile > 1
    a per-task count multiplies: num=8 with 8 slots/host asks for 64 GPUs on
    every host, and the job simply never schedules — it sits pending, which reads
    as a busy cluster rather than a bad request.
    """

    def _gpu_line(self, cpus, num_nodes):
        config = _enroot_provider_config()
        config['cpus'] = cpus
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config,
                                                 num_nodes=num_nodes)
        return next(l for l in script.split('\n') if l.startswith('#BSUB -gpu'))

    def test_per_host_when_several_slots_per_host(self):
        assert self._gpu_line(
            '8', 2) == '#BSUB -gpu "num=8/host:mode=exclusive_process"'

    def test_bare_form_kept_at_one_slot_per_host(self):
        """Equivalent to /host there, and kept byte-identical so existing
        clusters see no change to a directive that is already working."""
        assert self._gpu_line('1',
                              2) == '#BSUB -gpu "num=8:mode=exclusive_process"'

    def test_no_gpu_directive_when_none_requested(self):
        config = _base_provider_config()
        config['cpus'] = '8'
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config,
                                                 num_nodes=2)
        assert '#BSUB -gpu' not in script


class TestBsubOptionsOverrideDerivedDirectives:
    """A user-supplied bsub_options entry must REPLACE the derived directive.

    LSF honours the FIRST occurrence of a single-valued option. Appending the
    user's value after ours therefore ignored it silently, and the default
    `memory` of 16 is small enough to matter: a real 2-node training job was
    killed with TERM_MEMLIMIT at 46.5 GB of usage because a derived `-M 16G`
    shadowed the environment's configured `M: 64G`. Nothing in the output said
    so — the script contained both values and looked correct.
    """

    def _directives(self, config, num_nodes=2):
        script = lsf_instance._build_bsub_script(CLUSTER_NAME,
                                                 config,
                                                 num_nodes=num_nodes)
        return [l for l in script.split('\n') if l.startswith('#BSUB')]

    def test_memory_is_emitted_once_with_the_user_value(self):
        config = _enroot_provider_config()
        config['memory'] = '16'  # the provisioner default that caused the kill
        config['bsub_options'] = {'M': '64G'}
        directives = self._directives(config)
        mem = [d for d in directives if d.startswith('#BSUB -M')]
        assert mem == ['#BSUB -M 64G'], mem

    def test_derived_value_survives_when_not_overridden(self):
        config = _enroot_provider_config()
        config['memory'] = '32'
        config['bsub_options'] = {'G': 'grp_x'}
        assert '#BSUB -M 32G' in self._directives(config)

    @pytest.mark.parametrize('flag,value', [
        ('n', '99'),
        ('gpu', '"num=1:mode=shared"'),
        ('q', 'special'),
        ('J', 'custom-name'),
    ])
    def test_any_single_valued_flag_is_overridable(self, flag, value):
        """Including the ones the provisioner computes. If a user overrides them
        they take responsibility for the result, but it must actually take
        effect rather than being silently dropped."""
        config = _enroot_provider_config()
        config['bsub_options'] = {flag: value}
        directives = self._directives(config)
        matching = [d for d in directives if d.startswith(f'#BSUB -{flag} ')]
        assert matching == [f'#BSUB -{flag} {value}'], matching

    def test_resource_requirements_accumulate_rather_than_replace(self):
        """-R is the exception: LSF ANDs multiple -R expressions.

        Suppressing the derived `span[ptile=N]` because a user added a
        `rusage[...]` would let LSF satisfy the slot count from fewer hosts,
        silently collapsing a multi-node job onto one machine.
        """
        config = _enroot_provider_config()
        config['bsub_options'] = {'R': '"rusage[mem=1000]"'}
        directives = self._directives(config)
        assert '#BSUB -R "span[ptile=4]"' in directives
        assert '#BSUB -R "rusage[mem=1000]"' in directives

    def test_queue_options_still_override_cluster_options(self):
        config = _enroot_provider_config()
        config['queue'] = 'preemptable'
        config['queue_configs'] = {
            'preemptable': {
                'bsub_options': {
                    'G': 'grp_preemptable'
                }
            }
        }
        directives = self._directives(config)
        groups = [d for d in directives if d.startswith('#BSUB -G')]
        assert groups == ['#BSUB -G grp_preemptable'], groups

    def test_no_single_valued_flag_is_ever_emitted_twice(self):
        """The property the bug violated, checked over the whole directive set."""
        config = _enroot_provider_config()
        config['bsub_options'] = {'M': '64G', 'G': 'grp_x', 'n': '4'}
        directives = self._directives(config)
        flags = [d.split()[1] for d in directives]
        duplicated = {
            f for f in flags
            if flags.count(f) > 1 and f.lstrip('-') not in ('R',)
        }
        assert not duplicated, f'emitted twice: {duplicated}'

    def test_valueless_flags_render_without_a_value(self):
        config = _enroot_provider_config()
        assert '#BSUB -hl' in self._directives(config, num_nodes=2)
