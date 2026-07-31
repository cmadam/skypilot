"""Tests for LSF launch-phase timeout resolution.

The LSF backend always pins a queue (SkyPilot's `zone`), so there is no zone
failover and giving up on a pending job only forfeits its place in the queue.
The defaults are therefore generous, and both timeouts are overridable per
cluster and per queue.
"""

from unittest import mock

from sky.provision.lsf import instance as lsf_instance
from sky.provision.lsf import utils as lsf_utils


def _config(mapping):
    """Patch skypilot_config.get_nested with a dict keyed by key-tuple."""

    def _get_nested(keys, default=None, **kwargs):
        del kwargs
        return mapping.get(tuple(keys), default)

    return mock.patch.object(lsf_utils.skypilot_config,
                             'get_nested',
                             side_effect=_get_nested)


class TestGetProvisionTimeout:
    """Tests for get_provision_timeout / get_ready_timeout."""

    def test_defaults_when_unconfigured(self):
        with _config({}):
            assert (lsf_utils.get_provision_timeout('bluevela', 'normal') ==
                    lsf_utils.DEFAULT_PROVISION_TIMEOUT)
            assert (lsf_utils.get_ready_timeout('bluevela', 'normal') ==
                    lsf_utils.DEFAULT_READY_TIMEOUT)

    def test_default_is_24h(self):
        assert lsf_utils.DEFAULT_PROVISION_TIMEOUT == 24 * 60 * 60
        assert lsf_utils.DEFAULT_READY_TIMEOUT == 60 * 60

    def test_global_level(self):
        with _config({('lsf', 'provision_timeout'): 111}):
            assert lsf_utils.get_provision_timeout('bluevela', 'normal') == 111

    def test_cluster_overrides_global(self):
        with _config({
                ('lsf', 'provision_timeout'):
                    111,
                ('lsf', 'cluster_configs', 'bluevela', 'provision_timeout'):
                    222,
        }):
            assert lsf_utils.get_provision_timeout('bluevela', 'normal') == 222

    def test_queue_overrides_cluster(self):
        with _config({
                ('lsf', 'provision_timeout'):
                    111,
                ('lsf', 'cluster_configs', 'bluevela', 'provision_timeout'):
                    222,
                ('lsf', 'cluster_configs', 'bluevela', 'queue_configs',
                 'preemptable', 'provision_timeout'):
                    333,
        }):
            assert (lsf_utils.get_provision_timeout('bluevela',
                                                    'preemptable') == 333)
            # A different queue is unaffected by the preemptable override.
            assert lsf_utils.get_provision_timeout('bluevela', 'normal') == 222

    def test_no_queue_ignores_queue_configs(self):
        with _config({
                ('lsf', 'cluster_configs', 'bluevela', 'provision_timeout'):
                    222,
        }):
            assert lsf_utils.get_provision_timeout('bluevela', None) == 222

    def test_negative_means_indefinite(self):
        with _config({('lsf', 'cluster_configs', 'bluevela', 'ready_timeout'):
                          -1}):
            assert lsf_utils.get_ready_timeout('bluevela', 'normal') == -1

    def test_non_integer_falls_back_to_default(self):
        with _config({
                ('lsf', 'cluster_configs', 'bluevela', 'provision_timeout'):
                    'soon',
        }):
            assert (lsf_utils.get_provision_timeout('bluevela', 'normal') ==
                    lsf_utils.DEFAULT_PROVISION_TIMEOUT)

    def test_ready_and_provision_are_independent(self):
        with _config({
                ('lsf', 'cluster_configs', 'bluevela', 'provision_timeout'):
                    222,
        }):
            assert (lsf_utils.get_ready_timeout('bluevela', 'normal') ==
                    lsf_utils.DEFAULT_READY_TIMEOUT)


class TestProviderConfigTimeout:
    """Tests for the provisioner-side reader.

    Values arrive as strings via the Jinja-rendered cluster YAML, and clusters
    provisioned before these keys existed have no entry at all.
    """

    def test_reads_string_value(self):
        assert lsf_instance._get_timeout({'provision_timeout': '900'},
                                         'provision_timeout', 7) == 900

    def test_negative_passes_through(self):
        assert lsf_instance._get_timeout({'ready_timeout': '-1'},
                                        'ready_timeout', 7) == -1

    def test_missing_key_uses_default(self):
        assert lsf_instance._get_timeout({}, 'provision_timeout', 7) == 7

    def test_empty_string_uses_default(self):
        assert lsf_instance._get_timeout({'ready_timeout': ''},
                                        'ready_timeout', 7) == 7

    def test_garbage_uses_default(self):
        assert lsf_instance._get_timeout({'ready_timeout': 'later'},
                                        'ready_timeout', 7) == 7

    def test_wait_str(self):
        assert lsf_instance._wait_str(-1) == 'indefinitely'
        assert lsf_instance._wait_str(30) == 'up to 30s'


class TestWaitLoopsHonorNegativeTimeout:
    """A negative timeout must not exit the loop on the first iteration."""

    def test_job_nodes_loop_keeps_polling(self):
        client = mock.MagicMock()
        # PEND twice, then allocated: a zero/negative timeout must not make
        # the loop bail out before the nodes appear.
        client.get_job_state.side_effect = [
            'PEND', 'PEND', lsf_instance.lsf_adaptor.LSF_STATE_RUN
        ]
        client.check_job_has_nodes.return_value = True
        client.get_job_nodes.return_value = (['p1-r01-n1'], None)
        with mock.patch.object(lsf_instance.time, 'sleep'):
            nodes = lsf_instance._wait_for_job_nodes(client,
                                                     '1',
                                                     'cluster',
                                                     timeout=-1)
        assert nodes == ['p1-r01-n1']
        assert client.get_job_state.call_count == 3
