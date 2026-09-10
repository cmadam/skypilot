"""LSF distributed task executor for SkyPilot.

Unlike the Slurm executor, this module runs on the **driver** (the login node),
not on the compute nodes:

    python -m sky.skylet.executor.lsf --script=... --dispatch-root=...

Slurm can `srun python -m sky.skylet.executor.slurm` on a compute node because
the shared-filesystem SkyPilot runtime is importable there. On LSF the
compute-node process runs inside the user's own container, which has neither the
sky wheel nor its dependencies, and whose PATH is fixed by the generated enroot
config. So the per-node work that Slurm does *on* each node is done here instead,
and reaches the nodes through the shared-filesystem dispatcher the bsub
script started on each of them: write ``cmd_<seq>.sh`` into a node's dispatch
directory, the dispatcher runs it, then read back ``out_<seq>.log`` and
``rc_<seq>``.

The responsibilities are the same as the Slurm executor's — per-node
environment, unique per-node log filenames on a shared filesystem, streaming
prefixes that identify the node, and an aggregate return code — only the
execution site differs.

Rank comes from the manifest the job itself publishes
(``.sky/topology/rank-N``), never from the order in which the driver learned the
hostnames. Those are two
independent derivations (``bjobs -o EXEC_HOST`` order versus ``$LSB_HOSTS``
order) that LSF does not promise to agree on, and disagreement does not fail
fast: it pairs a rank with the wrong master address and hangs collective
initialization until the timeout.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import uuid

import colorama

from sky.skylet import constants

# How often to poll the shared filesystem for new output or a return code.
# Matches the dispatcher's own loop interval; a shared FS makes tighter polling
# expensive without making it more responsive.
_POLL_INTERVAL = 0.5

# How long to wait for the rank manifest before giving up. The manifest is
# written early in each node's bsub script, well before readiness is signalled,
# so by the time a task runs it should already be present; this covers a node
# that died between signalling ready and publishing.
_MANIFEST_TIMEOUT = 60.0


def read_rank_manifest(topology_dir: str,
                       nodes: List[str],
                       timeout: float = _MANIFEST_TIMEOUT) -> Dict[str, int]:
    """Read the host -> rank mapping published by the job.

    Each node writes its own short hostname into ``<topology_dir>/rank-<N>``.

    Args:
        topology_dir: directory holding the ``rank-*`` files.
        nodes: hostnames the driver expects, short or fully qualified.
        timeout: seconds to wait for every expected host to appear.

    Returns:
        Mapping from short hostname to rank.

    Raises:
        RuntimeError: if the manifest is missing or incomplete after the
            timeout. Failing here is deliberate: continuing with a guessed rank
            produces a job that hangs rather than one that reports an error.
    """
    expected = {n.split('.')[0] for n in nodes}
    deadline = time.time() + timeout
    ranks: Dict[str, int] = {}
    while True:
        ranks = {}
        try:
            entries = os.listdir(topology_dir)
        except OSError:
            entries = []
        for entry in entries:
            if not entry.startswith('rank-'):
                continue
            try:
                rank = int(entry[len('rank-'):])
            except ValueError:
                continue
            try:
                with open(os.path.join(topology_dir, entry),
                          'r',
                          encoding='utf-8') as f:
                    host = f.read().strip().split('.')[0]
            except OSError:
                continue
            if host:
                ranks[host] = rank
        if expected.issubset(ranks.keys()):
            return {h: r for h, r in ranks.items() if h in expected}
        if time.time() >= deadline:
            missing = sorted(expected - set(ranks.keys()))
            raise RuntimeError(
                f'Incomplete rank manifest in {topology_dir} after {timeout}s. '
                f'No rank published by: {", ".join(missing)}. '
                f'Found: {ranks}')
        time.sleep(_POLL_INTERVAL)


def node_display_name(rank: int) -> str:
    """SkyPilot's conventional name for a node, by rank."""
    return 'head' if rank == 0 else f'worker{rank}'


def log_filename(rank: int, is_setup: bool, is_single_node: bool) -> str:
    """Per-node log filename.

    Every node's logs land in one directory on a shared filesystem, so the
    filename has to carry the node's identity or nodes overwrite each other.
    Mirrors the Slurm executor's naming so log-consuming tooling does not need a
    per-cloud case.
    """
    name = node_display_name(rank)
    if is_setup:
        return f'setup-{name}.log'
    if is_single_node:
        return 'run.log'
    return f'{rank}-{name}.log'


def streaming_prefix(rank: int, ip: str, task_name: Optional[str],
                     is_setup: bool, is_single_node: bool) -> str:
    """Per-node prefix prepended to every streamed output line.

    The format is load-bearing beyond cosmetics: gbserver's SkyPilot monitor
    matches step metadata with a regex anchored on exactly this shape
    (``^(\\([^)]*\\)\\s+)?``), so a different prefix silently stops that capture
    rather than breaking visibly. Mirrors the Slurm executor's five cases.
    """
    name = node_display_name(rank)
    if is_setup:
        body = 'setup' if rank == 0 else f'setup, ip={ip}'
    elif is_single_node:
        body = f'{task_name or "task"}'
    elif rank == 0:
        body = f'{name}, rank={rank}'
    else:
        body = f'{name}, rank={rank}, ip={ip}'
    return f'{colorama.Fore.CYAN}({body}){colorama.Style.RESET_ALL} '


def write_command(dispatch_dir: str, seq: str, env_vars: Dict[str, str],
                  script: str) -> None:
    """Place a command in a node's dispatch directory for its dispatcher.

    Environment is exported by the generated script rather than passed out of
    band, because the dispatcher runs it with a plain ``bash <file>`` inside the
    container and inherits nothing from the driver.
    """
    os.makedirs(dispatch_dir, exist_ok=True)
    env_lines = '\n'.join(f'export {k}="{v}"' for k, v in env_vars.items())
    # Write to a temporary name and rename into place: the dispatcher globs for
    # cmd_*.sh and would happily execute a half-written file.
    tmp_path = os.path.join(dispatch_dir, f'.tmp_{seq}')
    cmd_path = os.path.join(dispatch_dir, f'cmd_{seq}.sh')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(env_lines + '\n' + script)
    os.replace(tmp_path, cmd_path)


def stream_until_done(dispatch_dir: str, seq: str, prefix: str,
                      log_path: str) -> int:
    """Stream one node's output as it appears, then return its exit code.

    Returns:
        The command's exit code, or 1 if the dispatcher never wrote one.
    """
    rc_path = os.path.join(dispatch_dir, f'rc_{seq}')
    out_path = os.path.join(dispatch_dir, f'out_{seq}.log')

    def emit(text: str) -> None:
        for line in text.splitlines(keepends=True):
            sys.stdout.write(prefix + line if line.strip() else line)
        sys.stdout.flush()

    last_pos = 0
    while not os.path.exists(rc_path):
        last_pos = _drain(out_path, last_pos, emit)
        time.sleep(_POLL_INTERVAL)
    # The dispatcher writes rc_ after the command exits, but output written just
    # before that may not have been read yet.
    _drain(out_path, last_pos, emit)

    if os.path.exists(out_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        shutil.copy(out_path, log_path)

    try:
        with open(rc_path, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 1


def _drain(out_path: str, last_pos: int, emit) -> int:
    """Emit any output appended since ``last_pos``; return the new position."""
    if not os.path.exists(out_path):
        return last_pos
    try:
        with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(last_pos)
            new_data = f.read()
            if new_data:
                emit(new_data)
            return f.tell()
    except OSError:
        return last_pos


def run_on_all_nodes(
    script: str,
    env_vars: Dict[str, str],
    dispatch_root: str,
    topology_dir: str,
    nodes: List[str],
    node_ips: List[str],
    log_dir: str,
    num_gpus_per_node: int,
    job_id: Optional[str],
    task_name: Optional[str],
    is_setup: bool,
) -> int:
    """Run one script on every node concurrently; return the first failure.

    Each node gets its own rank, log file and streaming prefix, and its own
    dispatch directory keyed by hostname.
    """
    host_to_rank = read_rank_manifest(topology_dir, nodes)
    num_nodes = len(host_to_rank)
    is_single_node = num_nodes == 1

    # Order IPs by rank so SKYPILOT_NODE_IPS is consistent for every node, and
    # so entry 0 is the head. The driver's own host ordering is not trusted for
    # rank, but it is what pairs a host with its IP.
    ip_by_host = {n.split('.')[0]: ip for n, ip in zip(nodes, node_ips)}
    ranked = sorted(host_to_rank.items(), key=lambda kv: kv[1])
    ips_in_rank_order = [ip_by_host.get(h, '') for h, _ in ranked]

    results: Dict[str, int] = {}
    threads: List[threading.Thread] = []

    def run_one(host: str, rank: int) -> None:
        seq = f'{"setup" if is_setup else "run"}-{uuid.uuid4().hex[:8]}'
        node_env = dict(env_vars)
        node_env[constants.SKYPILOT_NUM_GPUS_PER_NODE] = str(num_gpus_per_node)
        if not is_setup:
            # Setup env is set by CloudVmRayBackend._setup, matching Slurm.
            node_env['SKYPILOT_NODE_RANK'] = str(rank)
            node_env[constants.SKYPILOT_NUM_NODES] = str(num_nodes)
            node_env['SKYPILOT_NODE_IPS'] = '\n'.join(ips_in_rank_order)
        if job_id is not None:
            node_env['SKYPILOT_INTERNAL_JOB_ID'] = str(job_id)

        dispatch_dir = os.path.join(dispatch_root, host)
        prefix = streaming_prefix(rank, ip_by_host.get(host, ''), task_name,
                                  is_setup, is_single_node)
        log_path = os.path.join(os.path.expanduser(log_dir),
                                log_filename(rank, is_setup, is_single_node))
        write_command(dispatch_dir, seq, node_env, script)
        results[host] = stream_until_done(dispatch_dir, seq, prefix, log_path)

    for host, rank in ranked:
        thread = threading.Thread(target=run_one,
                                  args=(host, rank),
                                  daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    # Report the first non-zero code in rank order, so a multi-node failure has
    # a stable exit status rather than one that depends on which node lost the
    # race to finish.
    for host, _ in ranked:
        rc = results.get(host, 1)
        if rc != 0:
            return rc
    return 0


def _parse_args() -> Tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(
        description='Run a task on every node of an LSF allocation.')
    parser.add_argument('--script', help='Inline script to run.')
    parser.add_argument('--script-path',
                        help='File holding the script to run. Used instead of '
                        '--script when the script is too long to inline.')
    parser.add_argument('--env-vars',
                        default='{}',
                        help='JSON object of environment variables.')
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--dispatch-root',
                        required=True,
                        help='Directory containing one dispatch directory per '
                        'host, named by short hostname.')
    parser.add_argument('--topology-dir',
                        required=True,
                        help='Directory holding the rank manifest the job '
                        'publishes.')
    parser.add_argument('--nodes',
                        required=True,
                        help='Comma-separated allocated hostnames.')
    parser.add_argument('--node-ips', default='', help='Comma-separated IPs.')
    parser.add_argument('--num-gpus-per-node', type=int, default=0)
    parser.add_argument('--job-id')
    parser.add_argument('--task-name')
    parser.add_argument('--is-setup', action='store_true')
    args = parser.parse_args()

    if not args.script_path and args.script is None:
        parser.error('one of --script or --script-path is required')
    if args.script_path:
        with open(args.script_path, 'r', encoding='utf-8') as f:
            script = f.read()
    else:
        script = args.script
    return args, script


def main() -> None:
    args, script = _parse_args()
    nodes = [n for n in args.nodes.split(',') if n]
    node_ips = [ip for ip in args.node_ips.split(',') if ip]
    returncode = run_on_all_nodes(
        script=script,
        env_vars=json.loads(args.env_vars),
        dispatch_root=args.dispatch_root,
        topology_dir=args.topology_dir,
        nodes=nodes,
        node_ips=node_ips,
        log_dir=args.log_dir,
        num_gpus_per_node=args.num_gpus_per_node,
        job_id=args.job_id,
        task_name=args.task_name,
        is_setup=args.is_setup,
    )
    sys.exit(returncode)


if __name__ == '__main__':
    main()
