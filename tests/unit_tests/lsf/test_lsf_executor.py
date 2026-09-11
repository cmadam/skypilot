"""Tests for the driver-side LSF task executor.

The executor is what makes a multi-node LSF task actually multi-node: it reads
the rank each node published, then dispatches the script to every node with that
node's own rank, log filename and streaming prefix.

Because it runs driver-side rather than on the compute node, all of it is
testable with no cluster — which is the main practical reason for that design.
The dispatch protocol is exercised against a fake dispatcher that implements the
same contract as the bash one the bsub script starts (glob cmd_*.sh, run it,
write out_<seq>.log and rc_<seq>), so the tests cover the real handshake rather
than a mock of it.
"""

import os
import re
import subprocess
import threading
import time

import pytest

from sky.skylet.executor import lsf as lsf_executor


def _publish_manifest(topology_dir, host_by_rank):
    """Write the rank manifest the way each node's bsub script does."""
    os.makedirs(topology_dir, exist_ok=True)
    for rank, host in host_by_rank.items():
        with open(os.path.join(topology_dir, f'rank-{rank}'),
                  'w',
                  encoding='utf-8') as f:
            f.write(host + '\n')


class _FakeDispatcher(threading.Thread):
    """A Python stand-in for the per-host bash dispatcher.

    Implements the same shared-filesystem contract: pick up cmd_<seq>.sh, run it,
    write its output to out_<seq>.log and its exit code to rc_<seq>.
    """

    def __init__(self, dispatch_dir):
        super().__init__(daemon=True)
        self.dispatch_dir = dispatch_dir
        self.stop = threading.Event()
        self.ran = []

    def run(self):
        os.makedirs(self.dispatch_dir, exist_ok=True)
        while not self.stop.is_set():
            for name in sorted(os.listdir(self.dispatch_dir)):
                if not (name.startswith('cmd_') and name.endswith('.sh')):
                    continue
                seq = name[len('cmd_'):-len('.sh')]
                path = os.path.join(self.dispatch_dir, name)
                with open(path, 'r', encoding='utf-8') as f:
                    self.ran.append(f.read())
                result = subprocess.run(['bash', path],
                                        capture_output=True,
                                        text=True,
                                        check=False)
                out = os.path.join(self.dispatch_dir, f'out_{seq}.log')
                with open(out, 'w', encoding='utf-8') as f:
                    f.write(result.stdout + result.stderr)
                with open(os.path.join(self.dispatch_dir, f'rc_{seq}'),
                          'w',
                          encoding='utf-8') as f:
                    f.write(str(result.returncode))
                os.replace(path,
                           os.path.join(self.dispatch_dir, f'done_{seq}.sh'))
            time.sleep(0.05)


@pytest.fixture
def cluster(tmp_path):
    """A two-node allocation with dispatchers running on both hosts."""
    dispatch_root = str(tmp_path / 'dispatch')
    topology_dir = str(tmp_path / 'topology')
    hosts = ['p1-r08-n4', 'p1-r08-n5']
    _publish_manifest(topology_dir, {0: hosts[0], 1: hosts[1]})
    dispatchers = [
        _FakeDispatcher(os.path.join(dispatch_root, h)) for h in hosts
    ]
    for d in dispatchers:
        d.start()
    yield {
        'dispatch_root': dispatch_root,
        'topology_dir': topology_dir,
        'hosts': hosts,
        'ips': ['10.0.0.1', '10.0.0.2'],
        'log_dir': str(tmp_path / 'logs'),
        'dispatchers': dispatchers,
    }
    for d in dispatchers:
        d.stop.set()


class TestReadRankManifest:
    """Rank comes from the job, never from the driver's host ordering."""

    def test_reads_published_ranks(self, tmp_path):
        topology = str(tmp_path / 't')
        _publish_manifest(topology, {0: 'hostA', 1: 'hostB'})
        assert lsf_executor.read_rank_manifest(topology,
                                               ['hostA', 'hostB']) == {
                                                   'hostA': 0,
                                                   'hostB': 1
                                               }

    def test_driver_host_order_does_not_determine_rank(self, tmp_path):
        """The whole point of the manifest.

        The driver learns hosts from `bjobs -o EXEC_HOST`; the job derives rank
        from $LSB_HOSTS. LSF does not promise those agree, so a manifest that
        contradicts the driver's order must win.
        """
        topology = str(tmp_path / 't')
        _publish_manifest(topology, {0: 'hostB', 1: 'hostA'})
        ranks = lsf_executor.read_rank_manifest(topology, ['hostA', 'hostB'])
        assert ranks == {'hostB': 0, 'hostA': 1}

    def test_fqdn_from_lsf_matches_short_name_in_manifest(self, tmp_path):
        topology = str(tmp_path / 't')
        _publish_manifest(topology, {0: 'hostA', 1: 'hostB'})
        ranks = lsf_executor.read_rank_manifest(
            topology, ['hostA.bluevela.example.com', 'hostB.example.com'])
        assert ranks == {'hostA': 0, 'hostB': 1}

    def test_raises_when_a_node_never_published(self, tmp_path):
        """Fail loudly rather than guess.

        Continuing with an assumed rank produces a job that hangs on collective
        init until the timeout, which is far harder to diagnose than this error.
        """
        topology = str(tmp_path / 't')
        _publish_manifest(topology, {0: 'hostA'})
        with pytest.raises(RuntimeError, match='hostB'):
            lsf_executor.read_rank_manifest(topology, ['hostA', 'hostB'],
                                            timeout=0.2)

    def test_raises_when_the_manifest_is_missing_entirely(self, tmp_path):
        with pytest.raises(RuntimeError, match='Incomplete rank manifest'):
            lsf_executor.read_rank_manifest(str(tmp_path / 'absent'), ['hostA'],
                                            timeout=0.2)

    def test_ignores_unrelated_files(self, tmp_path):
        topology = str(tmp_path / 't')
        _publish_manifest(topology, {0: 'hostA'})
        for junk in ('master', 'rank-notanumber', '.hidden'):
            with open(os.path.join(topology, junk), 'w', encoding='utf-8') as f:
                f.write('ignore me')
        assert lsf_executor.read_rank_manifest(topology, ['hostA']) == {
            'hostA': 0
        }


class TestLogFilename:
    """Every node logs into one shared directory, so names must not collide."""

    def test_multi_node_names_carry_rank_and_role(self):
        assert lsf_executor.log_filename(0, False, False) == '0-head.log'
        assert lsf_executor.log_filename(2, False, False) == '2-worker2.log'

    def test_setup_names_carry_role(self):
        assert lsf_executor.log_filename(0, True, False) == 'setup-head.log'
        assert lsf_executor.log_filename(1, True, False) == 'setup-worker1.log'

    def test_single_node_keeps_the_conventional_name(self):
        assert lsf_executor.log_filename(0, False, True) == 'run.log'

    def test_matches_slurm_naming(self):
        """Log-consuming tooling should not need a per-cloud case."""
        from sky.skylet.executor import slurm  # pylint: disable=unused-import
        assert lsf_executor.node_display_name(0) == 'head'
        assert lsf_executor.node_display_name(3) == 'worker3'


class TestStreamingPrefix:
    """The prefix format is consumed by tooling, not just read by humans."""

    # gbserver's SkyPilot monitor anchors step-metadata capture on exactly this.
    GBSERVER_ANCHOR = re.compile(r'^(\([^)]*\)\s+)?')

    def _plain(self, prefix):
        """Strip ANSI colour, as a log consumer reading a captured file does."""
        return re.sub(r'\x1b\[[0-9;]*m', '', prefix)

    def test_head_shows_no_ip(self):
        assert self._plain(
            lsf_executor.streaming_prefix(0, '10.0.0.1', 'train', False,
                                          False)) == '(head, rank=0) '

    def test_worker_shows_its_ip(self):
        assert self._plain(
            lsf_executor.streaming_prefix(
                1, '10.0.0.2', 'train', False,
                False)) == '(worker1, rank=1, ip=10.0.0.2) '

    def test_single_node_uses_the_task_name(self):
        assert self._plain(
            lsf_executor.streaming_prefix(0, '10.0.0.1', 'train', False,
                                          True)) == '(train) '

    def test_setup_prefixes(self):
        assert self._plain(
            lsf_executor.streaming_prefix(0, '10.0.0.1', None, True,
                                          False)) == '(setup) '
        assert self._plain(
            lsf_executor.streaming_prefix(1, '10.0.0.2', None, True,
                                          False)) == '(setup, ip=10.0.0.2) '

    @pytest.mark.parametrize('rank,is_setup,single', [
        (0, False, False),
        (1, False, False),
        (0, True, False),
        (1, True, False),
        (0, False, True),
    ])
    def test_every_prefix_stays_parseable_by_the_monitor(
            self, rank, is_setup, single):
        """A prefix change would stop gbserver's capture silently, not visibly."""
        line = self._plain(
            lsf_executor.streaming_prefix(rank, '10.0.0.2', 'train', is_setup,
                                          single)) + 'GB_MESSAGE: hello'
        stripped = self.GBSERVER_ANCHOR.sub('', line)
        assert stripped == 'GB_MESSAGE: hello'


class TestWriteCommand:
    """The dispatcher globs for cmd_*.sh, so a partial file must not be visible."""

    def test_command_appears_atomically(self, tmp_path):
        d = str(tmp_path / 'h1')
        lsf_executor.write_command(d, 'abc', {'FOO': 'bar'}, 'echo hi')
        assert sorted(os.listdir(d)) == ['cmd_abc.sh']

    def test_env_is_exported_by_the_script(self, tmp_path):
        """Nothing is inherited: the dispatcher runs `bash <file>` in the
        container, with no connection to the driver's environment."""
        d = str(tmp_path / 'h1')
        lsf_executor.write_command(d, 'abc', {'FOO': 'bar'}, 'echo $FOO')
        body = open(os.path.join(d, 'cmd_abc.sh'), encoding='utf-8').read()
        assert body.startswith('export FOO="bar"\n')
        assert body.endswith('echo $FOO')


class TestRunOnAllNodes:
    """End-to-end against fake dispatchers implementing the real protocol."""

    def _run(self, cluster, script, **kwargs):
        params = dict(
            script=script,
            env_vars={},
            dispatch_root=cluster['dispatch_root'],
            topology_dir=cluster['topology_dir'],
            nodes=cluster['hosts'],
            node_ips=cluster['ips'],
            log_dir=cluster['log_dir'],
            num_gpus_per_node=8,
            job_id='7',
            task_name='gold',
            is_setup=False,
        )
        params.update(kwargs)
        return lsf_executor.run_on_all_nodes(**params)

    def test_script_runs_on_every_node(self, cluster):
        rc = self._run(cluster, 'echo ran')
        assert rc == 0
        assert all(d.ran for d in cluster['dispatchers']), (
            'every node should have received a command')

    def test_each_node_gets_its_own_rank(self, cluster):
        assert self._run(cluster, 'echo rank=$SKYPILOT_NODE_RANK') == 0
        ranks = set()
        for d in cluster['dispatchers']:
            body = d.ran[0]
            ranks.add(
                re.search(r'export SKYPILOT_NODE_RANK="(\d+)"', body).group(1))
        assert ranks == {'0', '1'
                        }, ('the defect this replaces gave every node rank 0')

    def test_num_nodes_reflects_the_allocation(self, cluster):
        self._run(cluster, 'true')
        for d in cluster['dispatchers']:
            assert 'export SKYPILOT_NUM_NODES="2"' in d.ran[0]

    def test_node_ips_lists_peers_in_rank_order(self, cluster):
        """Rank 0 must be first: consumers treat entry 0 as the head."""
        self._run(cluster, 'true')
        for d in cluster['dispatchers']:
            match = re.search(r'export SKYPILOT_NODE_IPS="([^"]*)"', d.ran[0])
            assert match.group(1) == '10.0.0.1\n10.0.0.2'

    def test_gpus_per_node_is_passed_through(self, cluster):
        self._run(cluster, 'true', num_gpus_per_node=8)
        for d in cluster['dispatchers']:
            assert 'export SKYPILOT_NUM_GPUS_PER_NODE="8"' in d.ran[0]

    def test_setup_does_not_set_per_node_task_vars(self, cluster):
        """Matches Slurm: setup env is set by the backend, not here."""
        self._run(cluster, 'true', is_setup=True)
        for d in cluster['dispatchers']:
            assert 'SKYPILOT_NODE_RANK' not in d.ran[0]
            assert 'SKYPILOT_NODE_IPS' not in d.ran[0]

    def test_failure_on_any_node_is_reported(self, cluster):
        assert self._run(cluster, 'exit 3') == 3

    def test_writes_one_log_file_per_node(self, cluster):
        self._run(cluster, 'echo hello')
        assert sorted(os.listdir(
            cluster['log_dir'])) == ['0-head.log', '1-worker1.log']

    def test_log_contents_are_per_node(self, cluster):
        self._run(cluster, 'echo I am rank $SKYPILOT_NODE_RANK')
        head = open(os.path.join(cluster['log_dir'], '0-head.log'),
                    encoding='utf-8').read()
        worker = open(os.path.join(cluster['log_dir'], '1-worker1.log'),
                      encoding='utf-8').read()
        assert 'I am rank 0' in head
        assert 'I am rank 1' in worker

    def test_output_is_streamed_with_a_node_prefix(self, cluster, capsys):
        self._run(cluster, 'echo marker')
        out = re.sub(r'\x1b\[[0-9;]*m', '', capsys.readouterr().out)
        assert '(head, rank=0) marker' in out
        assert '(worker1, rank=1, ip=10.0.0.2) marker' in out

    def test_user_env_vars_reach_every_node(self, cluster):
        self._run(cluster, 'true', env_vars={'MODEL': 'granite'})
        for d in cluster['dispatchers']:
            assert 'export MODEL="granite"' in d.ran[0]
