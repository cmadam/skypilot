"""Tests for parsing the LSF EXEC_HOST field into an allocation's host list.

EXEC_HOST is how the driver learns which nodes an LSF job actually got, so a
misparse is not a cosmetic bug: the driver would provision, dispatch work to, and
wait for readiness from the wrong set of hosts.

The dangerous property is that the tabular `bjobs -o` form truncates each field
to its width and marks the truncation in no way at all — the value just ends
early, and a short host list is indistinguishable from a small allocation. There
is therefore nothing in the output a test can assert on; what can be pinned is
the query itself, which is why the query-shape tests below check the command
rather than its result.
"""

import json
from unittest import mock

import pytest

from sky.adaptors import lsf


class TestParseExecHostField:
    """Unit coverage of the EXEC_HOST value grammar."""

    def test_single_host_single_slot(self):
        assert lsf.parse_exec_host_field('p1-r08-n4') == ['p1-r08-n4']

    def test_repeated_host_form(self):
        """One entry per slot, repeated."""
        assert lsf.parse_exec_host_field('host1:host1:host2') == [
            'host1', 'host2'
        ]

    def test_run_length_encoded_form(self):
        """LSF collapses slots on a host to "N*host" past some width."""
        assert lsf.parse_exec_host_field('4*p1-r01-n1:4*p1-r01-n2') == [
            'p1-r01-n1', 'p1-r01-n2'
        ]

    def test_mixed_forms_in_one_value(self):
        assert lsf.parse_exec_host_field('8*hostA:hostB:hostB:2*hostC') == [
            'hostA', 'hostB', 'hostC'
        ]

    def test_realistic_four_node_bluevela_allocation(self):
        """32 slots over 4 nodes, the reference GOLD run's shape."""
        hosts = ['p1-r08-n4', 'p1-r08-n5', 'p4-r24-n2', 'p4-r24-n3']
        value = ':'.join(h for h in hosts for _ in range(8))
        assert lsf.parse_exec_host_field(value) == hosts

    def test_order_is_preserved_not_sorted(self):
        """Order defines rank 0, and the job derives the same order from
        $LSB_HOSTS. Sorting here would silently disagree with the job about
        which node is the master — a mismatch that hangs NCCL init rather than
        failing fast."""
        value = 'p9-r01-n1:p1-r01-n1:p5-r01-n1'
        assert lsf.parse_exec_host_field(value) == [
            'p9-r01-n1', 'p1-r01-n1', 'p5-r01-n1'
        ]

    def test_fqdn_hosts_are_left_intact(self):
        assert lsf.parse_exec_host_field(
            'n1.bluevela.example.com:n2.bluevela.example.com') == [
                'n1.bluevela.example.com', 'n2.bluevela.example.com'
            ]

    @pytest.mark.parametrize('value', ['', ':', '::', '   ', ' : '])
    def test_separator_only_values_yield_no_hosts(self, value):
        assert lsf.parse_exec_host_field(value) == []

    def test_dash_passes_through_as_a_token(self):
        """`-` is what LSF prints for a job with no allocation yet.

        The parser does not special-case it; get_job_nodes rejects it before
        calling here, and that split is deliberate — this function's job is the
        grammar, not the policy about what counts as a valid allocation.
        """
        assert lsf.parse_exec_host_field('-') == ['-']

    def test_whitespace_around_entries_is_tolerated(self):
        assert lsf.parse_exec_host_field(' host1 : host1 : host2 ') == [
            'host1', 'host2'
        ]


def _client_with_stdout(stdout: str, rc: int = 0):
    """An LsfClient whose LSF command execution is stubbed out."""
    client = lsf.LsfClient.__new__(lsf.LsfClient)
    client._run_lsf_cmd = mock.Mock(return_value=(rc, stdout, ''))
    client._resolve_hostnames = mock.Mock(
        side_effect=lambda hosts:
        [f'10.0.0.{i + 1}' for i in range(len(hosts))])
    return client


def _json_records(exec_host: str) -> str:
    return json.dumps({'RECORDS': [{'EXEC_HOST': exec_host}]})


class TestGetJobNodesQueryShape:
    """Pin the query, because the failure it prevents leaves no trace.

    A truncated EXEC_HOST produces a valid-looking short host list, so no
    assertion on the parsed result can detect the regression. These tests assert
    that the command asks for a form that cannot truncate.
    """

    def test_queries_json_not_the_truncating_tabular_form(self):
        client = _client_with_stdout(_json_records('host1:host1:host2'))
        client.get_job_nodes('12345')
        cmd = client._run_lsf_cmd.call_args[0][0]
        assert '-json' in cmd, (
            'the tabular `-o` form truncates EXEC_HOST to the field width, '
            'with no marker in the output')
        assert '-noheader -o "EXEC_HOST"' not in cmd

    def test_returns_hosts_and_matching_ips(self):
        client = _client_with_stdout(_json_records('8*p1-r08-n4:8*p1-r08-n5'))
        nodes, ips = client.get_job_nodes('12345')
        assert nodes == ['p1-r08-n4', 'p1-r08-n5']
        assert len(ips) == len(nodes)

    def test_raises_when_no_hosts_are_allocated(self):
        client = _client_with_stdout(_json_records('-'))
        with pytest.raises(RuntimeError, match='No hosts allocated'):
            client.get_job_nodes('12345')

    def test_raises_on_empty_records(self):
        client = _client_with_stdout(json.dumps({'RECORDS': []}))
        with pytest.raises(RuntimeError, match='No hosts allocated'):
            client.get_job_nodes('12345')

    def test_raises_on_unparseable_output(self):
        """A truthful error beats a confusing one downstream.

        If `-json` were unsupported, bjobs prints usage text on stdout; failing
        here names the real problem instead of surfacing as an empty allocation.
        """
        client = _client_with_stdout('bjobs: illegal option -- json')
        with pytest.raises(RuntimeError, match='Failed to parse bjobs JSON'):
            client.get_job_nodes('12345')


class TestCheckJobHasNodes:
    """The truthiness probe stays on the tabular form, and that is safe."""

    def test_truncation_cannot_make_a_populated_field_empty(self):
        client = _client_with_stdout('host1:host1:host2')
        assert client.check_job_has_nodes('12345') is True

    def test_dash_means_not_yet_allocated(self):
        client = _client_with_stdout('-')
        assert client.check_job_has_nodes('12345') is False

    def test_empty_means_not_yet_allocated(self):
        client = _client_with_stdout('')
        assert client.check_job_has_nodes('12345') is False
