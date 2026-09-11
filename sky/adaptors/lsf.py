"""LSF adaptor for SkyPilot."""

import ipaddress
import json
import logging
import re
import shlex
from typing import Dict, List, NamedTuple, Optional, Tuple

from sky.utils import command_runner
from sky.utils import subprocess_utils
from sky.utils import timeline

logger = logging.getLogger(__name__)

# Regex to parse job ID from bsub output: "Job <12345> is submitted to ..."
_BSUB_JOB_ID_REGEX = re.compile(r'Job <(\d+)> is submitted')

# LSF job states
LSF_STATE_PEND = 'PEND'
LSF_STATE_RUN = 'RUN'
LSF_STATE_DONE = 'DONE'
LSF_STATE_EXIT = 'EXIT'
LSF_STATE_PSUSP = 'PSUSP'
LSF_STATE_USUSP = 'USUSP'
LSF_STATE_SSUSP = 'SSUSP'
LSF_STATE_WAIT = 'WAIT'
LSF_STATE_UNKWN = 'UNKWN'

# Terminal states (job is no longer running or pending)
LSF_TERMINAL_STATES = {LSF_STATE_DONE, LSF_STATE_EXIT}

# Suspended states
LSF_SUSPENDED_STATES = {LSF_STATE_PSUSP, LSF_STATE_USUSP, LSF_STATE_SSUSP}

# Markers echoed by compute-node feature probes. We match on these instead of
# the exit code because `lsrun` returns a non-zero code both when the probed
# feature is missing and when the dispatch itself failed, and those two cases
# must be told apart.
_PROBE_OK = 'GB_LSF_PROBE_OK'
_PROBE_MISSING = 'GB_LSF_PROBE_MISSING'


class LsfQueue(NamedTuple):
    """Information about an LSF queue."""
    name: str
    is_default: bool
    is_open: bool
    # Maximum runtime in seconds, None if unlimited
    max_runtime: Optional[int]


class NodeInfo(NamedTuple):
    """Information about an LSF host from bhosts/lshosts."""
    node: str
    status: str
    max_slots: int
    cpus: int
    memory_gb: float
    gpus: int
    gpu_model: str


def parse_exec_host_field(exec_host_str: str) -> List[str]:
    """Parse an LSF EXEC_HOST value into an ordered list of unique hosts.

    LSF reports one entry per allocated *slot*, either repeated
    (``host1:host1:host2``) or run-length encoded (``2*host1:2*host2``), and the
    two forms can be mixed in one value.

    Order is preserved and duplicates dropped. Order is load-bearing, not
    cosmetic: the first host is rank 0, and the bsub script derives the same
    ordering independently from ``$LSB_HOSTS``, so sorting here would silently
    disagree with the job's own view of which node is the master.

    Args:
        exec_host_str: the raw EXEC_HOST field value.

    Returns:
        Unique hostnames in the order LSF listed them.
    """
    unique_hosts: List[str] = []
    seen = set()
    for part in exec_host_str.split(':'):
        part = part.strip()
        if not part:
            continue
        # "N*hostname" -> "hostname"
        hostname = part.split('*', 1)[1] if '*' in part else part
        if hostname not in seen:
            seen.add(hostname)
            unique_hosts.append(hostname)
    return unique_hosts


class LsfClient:
    """Client for IBM Spectrum LSF control plane operations."""

    def __init__(
        self,
        ssh_host: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_user: Optional[str] = None,
        ssh_key: Optional[str] = None,
        ssh_proxy_command: Optional[str] = None,
        ssh_proxy_jump: Optional[str] = None,
        is_inside_lsf_cluster: bool = False,
        identities_only: Optional[bool] = None,
    ):
        """Initialize LsfClient.

        Args:
            ssh_host: Hostname of the LSF login node.
            ssh_port: SSH port on the login node.
            ssh_user: SSH username.
            ssh_key: Path to SSH private key, or None for keyless SSH.
            ssh_proxy_command: Optional SSH proxy command.
            ssh_proxy_jump: Optional SSH proxy jump destination.
            is_inside_lsf_cluster: If True, uses local execution mode (for
                when running on the LSF cluster itself). Defaults to False.
            identities_only: If True, only use the specified identity file and
                don't try ssh-agent keys. If None, defaults to False.
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.ssh_proxy_command = ssh_proxy_command
        self.ssh_proxy_jump = ssh_proxy_jump

        self._runner: command_runner.CommandRunner

        if is_inside_lsf_cluster:
            self._runner = command_runner.LocalProcessCommandRunner()
        else:
            assert ssh_host is not None
            assert ssh_port is not None
            assert ssh_user is not None
            self._runner = command_runner.SSHCommandRunner(
                (ssh_host, ssh_port),
                ssh_user,
                ssh_key,
                ssh_proxy_command=ssh_proxy_command,
                ssh_proxy_jump=ssh_proxy_jump,
                enable_interactive_auth=True,
                disable_identities_only=not identities_only,
            )

    def _run_lsf_cmd(self, cmd: str) -> Tuple[int, str, str]:
        return self._runner.run(cmd,
                                require_outputs=True,
                                separate_stderr=True,
                                stream_logs=False)

    def submit_job(
        self,
        queue: Optional[str],
        job_name: str,
        script_path: str,
    ) -> str:
        """Submit an LSF job script.

        Args:
            queue: LSF queue to submit to. If None, uses the default queue.
            job_name: Name to give the job.
            script_path: Remote path to the job script.

        Returns:
            The job ID of the submitted job.
        """
        cmd = f'bsub -J {shlex.quote(job_name)}'
        if queue is not None:
            cmd += f' -q {shlex.quote(queue)}'
        cmd += f' < {shlex.quote(script_path)}'

        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to submit LSF job.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)

        job_id_match = _BSUB_JOB_ID_REGEX.search(stdout)
        if not job_id_match:
            raise RuntimeError(
                f'Failed to parse job ID from bsub output: {stdout}')

        job_id = job_id_match.group(1).strip()
        logger.debug(f'Successfully submitted LSF job {job_id} with name '
                     f'{job_name}: {stdout}')
        return job_id

    def query_jobs(
        self,
        job_name: Optional[str] = None,
        state_filters: Optional[List[str]] = None,
    ) -> List[str]:
        """Query LSF jobs by state and optional name.

        Args:
            job_name: Optional job name to filter by.
            state_filters: List of job states to filter by
                (e.g., ['RUN', 'PEND']). If None, returns all jobs.

        Returns:
            List of job IDs matching the filters.
        """
        cmd = 'bjobs -noheader -o "JOBID"'
        if job_name is not None:
            cmd += f' -J {shlex.quote(job_name)}'

        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            # bjobs returns non-zero when no jobs are found
            if 'No unfinished job found' in stderr or \
               'No job found' in stderr:
                return []
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                'Failed to query LSF jobs.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

        job_ids = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if line and line.isdigit():
                if state_filters is None:
                    job_ids.append(line)
                else:
                    state = self.get_job_state(line)
                    if state in state_filters:
                        job_ids.append(line)
        return job_ids

    def get_job_state(self, job_id: str) -> Optional[str]:
        """Get the state of an LSF job.

        Args:
            job_id: The LSF job ID.

        Returns:
            The job state (e.g., 'PEND', 'RUN', 'DONE', 'EXIT'),
            or None if the job is not found.
        """
        cmd = f'bjobs -noheader -o "STAT" {job_id}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            if 'is not found' in stderr:
                return None
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to get job state for job {job_id}.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

        state = stdout.strip()
        return state if state else None

    def get_job_state_json(self, job_id: str) -> Optional[Dict]:
        """Get detailed job info via bjobs -json.

        Args:
            job_id: The LSF job ID.

        Returns:
            Parsed JSON dict with job details, or None if not found.
        """
        cmd = f'bjobs -json -o "jobid stat exit_code exec_host" {job_id}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            if 'is not found' in stderr:
                return None
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to get job info for job {job_id}.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

        try:
            data = json.loads(stdout)
            records = data.get('RECORDS', [])
            if records:
                return records[0]
        except (json.JSONDecodeError, KeyError, IndexError):
            logger.warning(f'Failed to parse bjobs JSON output: {stdout}')
        return None

    def get_jobs_state_by_name(self, job_name: str) -> List[str]:
        """Get the states of all LSF jobs by name."""
        cmd = f'bjobs -noheader -o "STAT" -J {shlex.quote(job_name)}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            if 'No unfinished job found' in stderr or \
               'No job found' in stderr:
                return []
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to get job state for job {job_name}.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

        return [s.strip() for s in stdout.splitlines() if s.strip()]

    def cancel_jobs_by_name(self, job_name: str,
                            signal: Optional[str] = None) -> None:
        """Cancel LSF job(s) by name.

        Args:
            job_name: Name of the job(s) to cancel.
            signal: Optional signal to send (e.g., 'KILL', 'TERM').
        """
        cmd = f'bkill -J {shlex.quote(job_name)}'
        if signal is not None:
            cmd = f'bkill -s {signal} -J {shlex.quote(job_name)}'

        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        combined = f'{stdout}\n{stderr}'
        if rc != 0:
            if ('not found' in combined or 'already finished' in combined or
                    'No matching job' in combined):
                logger.debug(f'Job {job_name} not found or already done')
                return
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to cancel job {job_name}.',
                stderr=combined,
                stream_logs=False)
        logger.debug(f'Successfully cancelled job {job_name}: {stdout}')

    def cancel_job(self, job_id: str,
                   signal: Optional[str] = None) -> None:
        """Cancel an LSF job by ID.

        Args:
            job_id: The LSF job ID to cancel.
            signal: Optional signal to send.
        """
        cmd = f'bkill {job_id}'
        if signal is not None:
            cmd = f'bkill -s {signal} {job_id}'

        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0 and 'already finished' not in stderr:
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to cancel job {job_id}.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)
        logger.debug(f'Successfully cancelled job {job_id}: {stdout}')

    def info(self) -> str:
        """Get LSF cluster information.

        Returns:
            The stdout output from bhosts.
        """
        cmd = 'bhosts'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to get LSF cluster information.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)
        return stdout

    @staticmethod
    def _parse_mem(mem_str: str) -> float:
        """Parse LSF memory string (e.g. '1.9T', '1006.8G', '1031047M')."""
        if mem_str == '-':
            return 0.0
        if mem_str.endswith('T'):
            return float(mem_str[:-1]) * 1024.0
        if mem_str.endswith('G'):
            return float(mem_str[:-1])
        if mem_str.endswith('M'):
            return float(mem_str[:-1]) / 1024.0
        return float(mem_str) / 1024.0

    def info_nodes(self) -> List[NodeInfo]:
        """Get LSF host information with GPU details.

        Returns node names, statuses, slots, CPUs, memory, and GPU info.
        Uses lshosts for static info and bhosts for dynamic status.
        """
        # Get static host info (CPUs, memory)
        # Try compact -o format first (3 columns: HOST_NAME ncpus maxmem)
        cmd = 'lshosts -o "HOST_NAME ncpus maxmem"'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        use_compact_format = (rc == 0)
        if rc != 0:
            cmd = 'lshosts -w'
            rc, stdout, stderr = self._run_lsf_cmd(cmd)
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                'Failed to get LSF host information.',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

        host_info: Dict[str, Dict] = {}
        lines = stdout.strip().splitlines()
        if use_compact_format:
            # -o format: HOST_NAME ncpus maxmem
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    hostname = parts[0]
                    try:
                        cpus = int(parts[1]) if parts[1] != '-' else 0
                        memory_gb = self._parse_mem(parts[2])
                        host_info[hostname] = {
                            'cpus': cpus,
                            'memory_gb': memory_gb,
                        }
                    except (ValueError, IndexError):
                        continue
        else:
            # lshosts -w format: HOST_NAME type model cpuf ncpus maxmem ...
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    hostname = parts[0]
                    try:
                        cpus = int(parts[4]) if parts[4] != '-' else 0
                        memory_gb = self._parse_mem(parts[5])
                        host_info[hostname] = {
                            'cpus': cpus,
                            'memory_gb': memory_gb,
                        }
                    except (ValueError, IndexError):
                        continue

        # Get dynamic host status
        cmd = 'bhosts -w'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to get LSF host status.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)

        nodes = []
        lines = stdout.strip().splitlines()
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                hostname = parts[0]
                status = parts[1]
                max_slots = int(parts[3]) if len(parts) > 3 else 0
                info = host_info.get(hostname, {})
                nodes.append(NodeInfo(
                    node=hostname,
                    status=status,
                    max_slots=max_slots,
                    cpus=info.get('cpus', 0),
                    memory_gb=info.get('memory_gb', 0.0),
                    gpus=0,  # Populated by GPU discovery
                    gpu_model='',
                ))

        # Try to get GPU info via LSF GPU resource queries
        gpu_info = self._get_gpu_info()
        if gpu_info:
            nodes = [
                node._replace(
                    gpus=gpu_info.get(node.node, {}).get('count', node.gpus),
                    gpu_model=gpu_info.get(node.node, {}).get('model',
                                                              node.gpu_model),
                ) for node in nodes
            ]

        return nodes

    def _get_gpu_info(self) -> Dict[str, Dict]:
        """Get GPU information per host.

        Returns:
            Dict mapping hostname -> {'count': int, 'model': str}

        Parses `lshosts -gpu` output where each GPU gets a line. The
        hostname appears only on the first GPU line for each host;
        subsequent lines for the same host have leading whitespace.
        """
        cmd = 'lshosts -gpu -w'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            return {}

        gpu_info: Dict[str, Dict] = {}
        current_host: Optional[str] = None
        lines = stdout.strip().splitlines()
        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            if not line[0].isspace():
                parts = line.split()
                if len(parts) >= 3:
                    current_host = parts[0]
                    gpu_model = parts[2]
                    gpu_info[current_host] = {
                        'count': 1,
                        'model': gpu_model,
                    }
            else:
                if current_host is not None and current_host in gpu_info:
                    gpu_info[current_host]['count'] += 1
        return gpu_info

    def check_job_has_nodes(self, job_id: str) -> bool:
        """Check if an LSF job has hosts allocated.

        Left on the tabular form deliberately: this only asks whether the field
        is non-empty, and width truncation cannot turn a populated EXEC_HOST
        into an empty one. get_job_nodes(), which needs the whole list, uses
        JSON instead.
        """
        cmd = f'bjobs -noheader -o "EXEC_HOST" {job_id}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            logger.debug(f'Failed to check hosts for job {job_id}: '
                         f'{stdout}\n{stderr}')
            return False
        return bool(stdout.strip()) and stdout.strip() != '-'

    @timeline.event
    def get_job_nodes(self, job_id: str) -> Tuple[List[str], List[str]]:
        """Get the list of nodes and their IPs for a given job ID.

        Args:
            job_id: The LSF job ID.

        Returns:
            A tuple of (nodes, node_ips) where nodes is a list of unique
            hostnames and node_ips is a list of corresponding IP addresses.
        """
        # Queried as JSON rather than `-noheader -o "EXEC_HOST"`: the tabular
        # form truncates each field to its width, and EXEC_HOST is the one field
        # here whose length grows with the job. It holds one entry per allocated
        # slot, so a 4-node job at 8 slots/node is already ~400 characters and a
        # large job runs to kilobytes. Truncation carries no marker — the value
        # simply ends early — so the failure is a short host list that looks
        # entirely valid, and the driver would provision and dispatch to a
        # subset of the allocation. JSON output is not width-limited, and
        # specifying a wide `-o` field instead would pad every reply to that
        # width. get_job_info() already depends on `-json` in this file.
        cmd = f'bjobs -json -o "exec_host" {job_id}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            f'Failed to get hosts for job {job_id}.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)

        try:
            records = json.loads(stdout).get('RECORDS', [])
        except (json.JSONDecodeError, AttributeError) as e:
            raise RuntimeError(
                f'Failed to parse bjobs JSON output for job {job_id}: '
                f'{stdout}') from e

        # LSF EXEC_HOST format: "host1:host1:host2:host2" or
        # "N*host1:M*host2" for multi-slot
        exec_host_str = (records[0].get('EXEC_HOST', '')
                         if records else '').strip()
        if not exec_host_str or exec_host_str == '-':
            raise RuntimeError(f'No hosts allocated for job {job_id}.')

        unique_hosts = parse_exec_host_field(exec_host_str)

        if not unique_hosts:
            raise RuntimeError(
                f'No hosts found for job {job_id}. '
                f'EXEC_HOST output: {exec_host_str}')

        # Resolve hostnames to IPs
        node_ips = self._resolve_hostnames(unique_hosts)

        return unique_hosts, node_ips

    def _resolve_hostnames(self, hostnames: List[str]) -> List[str]:
        """Resolve a list of hostnames to IP addresses.

        Args:
            hostnames: List of hostnames to resolve.

        Returns:
            List of IP addresses in the same order as hostnames.
        """
        ips = []
        to_resolve = []
        ip_map: Dict[str, str] = {}

        for hostname in hostnames:
            try:
                ipaddress.ip_address(hostname)
                ip_map[hostname] = hostname
            except ValueError:
                to_resolve.append(hostname)

        if to_resolve:
            hosts_str = ' '.join(to_resolve)
            resolve_cmd = (
                f'for h in {hosts_str}; do '
                f'ip=$(getent ahostsv4 "$h" | head -1 | '
                f'awk \'{{print $1}}\'); '
                f'if [ -n "$ip" ]; then echo "$h $ip"; '
                f'else echo "$h UNRESOLVED"; fi; '
                f'done')
            rc, stdout, stderr = self._run_lsf_cmd(resolve_cmd)
            subprocess_utils.handle_returncode(
                rc,
                resolve_cmd,
                f'Failed to resolve hostnames: {to_resolve}',
                stderr=f'{stdout}\n{stderr}',
                stream_logs=False)

            unresolved = []
            for line in stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    hostname = parts[0]
                    ip = parts[1]
                    if ip == 'UNRESOLVED':
                        unresolved.append(hostname)
                    else:
                        ip_map[hostname] = ip

            if unresolved:
                raise RuntimeError(
                    f'Failed to resolve hostnames: {unresolved}')

        for hostname in hostnames:
            if hostname not in ip_map:
                raise RuntimeError(
                    f'Failed to resolve hostname: {hostname}')
            ips.append(ip_map[hostname])

        return ips

    def get_queues(self) -> List[LsfQueue]:
        """Get the queue information for the LSF cluster.

        Returns:
            List of LsfQueue objects.

        Uses `bqueues -w` which outputs:
        QUEUE_NAME PRIO STATUS MAX JL/U JL/P JL/H NJOBS PEND RUN SUSP RSV PJOBS
        """
        cmd = 'bqueues -w'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to get LSF queues.',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)

        queues = []
        lines = stdout.strip().splitlines()
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            status = parts[2]  # e.g. "Open:Active", "Closed:Inact_Adm"
            is_open = status.lower().startswith('open')
            queues.append(LsfQueue(
                name=name,
                is_default=False,
                is_open=is_open,
                max_runtime=None,
            ))
        return queues

    def get_default_queue(self) -> Optional[str]:
        """Get the default queue name for the LSF cluster.

        Returns:
            The default queue name, or None if it cannot be determined.
        """
        queues = self.get_queues()
        for queue in queues:
            if queue.is_default:
                return queue.name
        return None

    def get_pending_job_count(self,
                              queue: str,
                              exclude_job_id: Optional[str] = None) -> int:
        """Count pending jobs in a queue, optionally excluding our own.

        Args:
            queue: The LSF queue to query.
            exclude_job_id: Optional job ID to exclude from the count.

        Returns:
            The number of pending jobs, or -1 if the query fails.
        """
        cmd = f'bjobs -noheader -o "JOBID" -q {shlex.quote(queue)} -p'
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            return -1
        job_ids = [j.strip() for j in stdout.strip().splitlines() if j.strip()]
        if exclude_job_id:
            job_ids = [j for j in job_ids if j != exclude_job_id]
        return len(job_ids)

    def _probe_on_node(self, node: Optional[str],
                       test_cmd: str) -> Optional[bool]:
        """Run a feature test on a compute node via `lsrun`.

        LSF commands are issued from the submit (login) host, which on many
        clusters does not have the same software installed as the compute
        nodes that actually run jobs. Feature probes are therefore dispatched
        to a compute node instead of being run locally.

        Args:
            node: The compute node to probe. If None, no probe is attempted.
            test_cmd: A shell command that succeeds iff the feature exists.

        Returns:
            True/False if the probe reported an answer, None if it was
            inconclusive (no node available, `lsrun` unusable, host down).
        """
        if node is None:
            return None
        script = (f'if {test_cmd} >/dev/null 2>&1; then echo {_PROBE_OK}; '
                  f'else echo {_PROBE_MISSING}; fi')
        cmd = (f'lsrun -m {shlex.quote(node)} '
               f'/bin/sh -c {shlex.quote(script)}')
        _, stdout, stderr = self._run_lsf_cmd(cmd)
        if _PROBE_OK in stdout:
            return True
        if _PROBE_MISSING in stdout:
            return False
        logger.debug(f'Inconclusive LSF probe on {node}: {test_cmd!r} '
                     f'(stdout={stdout.strip()!r} stderr={stderr.strip()!r})')
        return None

    def check_enroot_available(
            self, node: Optional[str] = None) -> Optional[bool]:
        """Check if enroot is available on a compute node.

        Args:
            node: The compute node to probe.

        Returns:
            True if enroot is installed and accessible, False if it is
            missing, None if the check was inconclusive.
        """
        return self._probe_on_node(node, 'command -v enroot')

    def check_dir_shared_fs(self, path: str) -> Optional[str]:
        """Check the filesystem type of a directory.

        Args:
            path: The directory path to check.

        Returns:
            The filesystem type string (e.g., 'nfs', 'gpfs'),
            or None if the check could not be performed.
        """
        cmd = f'stat -f -c %T {shlex.quote(path)}'
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            return None
        return stdout.strip().lower()

    def check_homedir_shared_fs(self) -> Optional[str]:
        """Check the filesystem type of the home directory."""
        return self.check_dir_shared_fs('~')

    def get_env(self) -> Dict[str, str]:
        """Fetch environment variables from the remote host.

        Returns:
            Dictionary of environment variable name -> value.
        """
        rc, stdout, stderr = self._run_lsf_cmd('env')
        if rc != 0:
            logger.warning(f'Failed to fetch remote env: {stderr}')
            return {}
        env: Dict[str, str] = {}
        for line in stdout.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                env[key] = value
        return env

    def get_remote_home_dir(self) -> str:
        """Returns the remote user's home directory."""
        return self._runner.get_remote_home_dir()

    def check_file_exists(self, path: str) -> bool:
        """Check if a file exists on the remote host."""
        cmd = f'test -f {shlex.quote(path)}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc not in (0, 1):
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to check for file: {path}',
                stderr=f'{stdout}\n{stderr}')
        return rc == 0

    def check_fuse_enabled(self,
                           node: Optional[str] = None) -> Optional[bool]:
        """Check if FUSE is available on a compute node.

        Args:
            node: The compute node to probe.

        Returns:
            True if FUSE is available, False if it is not, None if the check
            was inconclusive.
        """
        return self._probe_on_node(node, 'test -e /dev/fuse')

    def get_lsf_version(self) -> Optional[str]:
        """Get the LSF version string.

        Returns:
            Version string (e.g., '10.1.0.14'), or None.
        """
        cmd = 'lsid | head -1'
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            return None
        # Output: "IBM Spectrum LSF 10.1.0.14, ..."
        match = re.search(r'(\d+\.\d+\.\d+(?:\.\d+)?)', stdout)
        return match.group(1) if match else None
