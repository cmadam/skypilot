"""Tests for LSF cluster feature detection (enroot, FUSE).

Feature probes must run on a compute node, not on the submit (login) host:
on clusters like BlueVela enroot is installed only on the compute nodes, so
probing the login host reports a missing feature and blocks every container
launch.
"""

from unittest import mock

import pytest

from sky.adaptors import lsf as lsf_adaptor
from sky.adaptors.lsf import NodeInfo
from sky.provision.lsf import utils as lsf_utils


def _node(name, status='ok', gpus=8):
    return NodeInfo(node=name,
                    status=status,
                    max_slots=96,
                    cpus=96,
                    memory_gb=2015.0,
                    gpus=gpus,
                    gpu_model='NVIDIAH10080GBHBM3' if gpus else '')


# A realistic BlueVela-shaped host list: admin-closed masters with no GPUs
# first, then the compute nodes.
_NODES = [
    _node('lsfn-master1', status='closed_Adm', gpus=0),
    _node('lsfn-master2', status='closed_Adm', gpus=0),
    _node('p1-r01-n1'),
    _node('p1-r01-n2'),
]


class TestPickProbeNode:
    """Tests for _pick_probe_node."""

    def test_prefers_ok_gpu_node(self):
        with mock.patch.object(lsf_utils,
                               'get_lsf_nodes_info',
                               return_value=_NODES):
            assert lsf_utils._pick_probe_node('c') == 'p1-r01-n1'

    def test_skips_closed_nodes(self):
        nodes = [
            _node('p1-r01-n1', status='closed_Excl'),
            _node('p1-r01-n2'),
        ]
        with mock.patch.object(lsf_utils,
                               'get_lsf_nodes_info',
                               return_value=nodes):
            assert lsf_utils._pick_probe_node('c') == 'p1-r01-n2'

    def test_falls_back_to_cpu_only_node(self):
        nodes = [_node('cpu1', gpus=0), _node('cpu2', gpus=0)]
        with mock.patch.object(lsf_utils,
                               'get_lsf_nodes_info',
                               return_value=nodes):
            assert lsf_utils._pick_probe_node('c') == 'cpu1'

    @pytest.mark.parametrize('side_effect,nodes', [
        (None, []),
        (RuntimeError('ssh failed'), None),
    ])
    def test_returns_none_when_undeterminable(self, side_effect, nodes):
        with mock.patch.object(lsf_utils,
                               'get_lsf_nodes_info',
                               side_effect=side_effect,
                               return_value=nodes):
            assert lsf_utils._pick_probe_node('c') is None


class TestProbeOnNode:
    """Tests for LsfClient._probe_on_node."""

    def _client(self, stdout, rc=0):
        client = mock.Mock(spec=lsf_adaptor.LsfClient)
        client._run_lsf_cmd = mock.Mock(return_value=(rc, stdout, ''))
        client._probe_on_node = (
            lambda node, test_cmd: lsf_adaptor.LsfClient._probe_on_node(
                client, node, test_cmd))
        return client

    def test_no_node_does_not_probe(self):
        client = self._client('')
        assert client._probe_on_node(None, 'command -v enroot') is None
        client._run_lsf_cmd.assert_not_called()

    def test_dispatches_to_named_node(self):
        client = self._client(f'{lsf_adaptor._PROBE_OK}\n')
        assert client._probe_on_node('p1-r01-n1', 'command -v enroot') is True
        cmd = client._run_lsf_cmd.call_args[0][0]
        assert cmd.startswith('lsrun -m p1-r01-n1 ')
        assert 'command -v enroot' in cmd

    def test_missing_marker_means_absent(self):
        # A non-zero rc alongside the marker still means "answered".
        client = self._client(f'{lsf_adaptor._PROBE_MISSING}\n', rc=1)
        assert client._probe_on_node('p1-r01-n1', 'command -v enroot') is False

    @pytest.mark.parametrize(
        'stdout', ['', 'lsrun: command not found\n', 'unrelated output\n'])
    def test_no_marker_is_inconclusive(self, stdout):
        client = self._client(stdout, rc=1)
        assert client._probe_on_node('p1-r01-n1', 'command -v enroot') is None


class TestCheckEnrootEnabled:
    """Tests for check_enroot_enabled."""

    def test_config_disabled_short_circuits(self):
        with mock.patch.object(lsf_utils,
                               'get_enroot_config',
                               return_value={'enabled': False}), \
             mock.patch.object(lsf_utils, '_create_lsf_client') as create:
            assert lsf_utils.check_enroot_enabled('c') is False
            create.assert_not_called()

    def test_probes_compute_node_when_config_enabled(self):
        client = mock.Mock()
        client.check_enroot_available.return_value = True
        with mock.patch.object(lsf_utils,
                               'get_enroot_config',
                               return_value={'enabled': True}), \
             mock.patch.object(lsf_utils, '_create_lsf_client',
                               return_value=client), \
             mock.patch.object(lsf_utils, 'get_lsf_nodes_info',
                               return_value=_NODES), \
             mock.patch.object(lsf_utils.kv_cache, 'get_cache_entry',
                               return_value=None), \
             mock.patch.object(lsf_utils.kv_cache,
                               'add_or_update_cache_entry') as add_entry:
            assert lsf_utils.check_enroot_enabled('c') is True
            client.check_enroot_available.assert_called_once_with(
                node='p1-r01-n1')
            assert add_entry.call_args[0][0] == 'lsf:enroot_compute_enabled:c'
            assert add_entry.call_args[0][1] == 'true'

    def test_inconclusive_probe_trusts_config_and_is_not_cached(self):
        client = mock.Mock()
        client.check_enroot_available.return_value = None
        with mock.patch.object(lsf_utils,
                               'get_enroot_config',
                               return_value={'enabled': True}), \
             mock.patch.object(lsf_utils, '_create_lsf_client',
                               return_value=client), \
             mock.patch.object(lsf_utils, 'get_lsf_nodes_info',
                               return_value=_NODES), \
             mock.patch.object(lsf_utils.kv_cache, 'get_cache_entry',
                               return_value=None), \
             mock.patch.object(lsf_utils.kv_cache,
                               'add_or_update_cache_entry') as add_entry:
            assert lsf_utils.check_enroot_enabled('c') is True
            add_entry.assert_not_called()

    def test_missing_enroot_on_compute_node_reports_false(self):
        client = mock.Mock()
        client.check_enroot_available.return_value = False
        with mock.patch.object(lsf_utils,
                               'get_enroot_config',
                               return_value={'enabled': True}), \
             mock.patch.object(lsf_utils, '_create_lsf_client',
                               return_value=client), \
             mock.patch.object(lsf_utils, 'get_lsf_nodes_info',
                               return_value=_NODES), \
             mock.patch.object(lsf_utils.kv_cache, 'get_cache_entry',
                               return_value=None), \
             mock.patch.object(lsf_utils.kv_cache,
                               'add_or_update_cache_entry'):
            assert lsf_utils.check_enroot_enabled('c') is False


class TestCheckFuseEnabled:
    """Tests for check_fuse_enabled."""

    def test_probes_compute_node(self):
        client = mock.Mock()
        client.check_fuse_enabled.return_value = True
        with mock.patch.object(lsf_utils, '_create_lsf_client',
                               return_value=client), \
             mock.patch.object(lsf_utils, 'get_lsf_nodes_info',
                               return_value=_NODES), \
             mock.patch.object(lsf_utils.kv_cache, 'get_cache_entry',
                               return_value=None), \
             mock.patch.object(lsf_utils.kv_cache,
                               'add_or_update_cache_entry') as add_entry:
            assert lsf_utils.check_fuse_enabled('c') is True
            client.check_fuse_enabled.assert_called_once_with(
                node='p1-r01-n1')
            assert add_entry.call_args[0][0] == 'lsf:fuse_compute_enabled:c'
