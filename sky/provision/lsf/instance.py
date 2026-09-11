"""LSF instance provisioner for SkyPilot.

Manages the lifecycle of LSF jobs as virtual instances:
submit (run_instances) → poll (wait_instances/query_instances) →
terminate (terminate_instances).
"""
import base64
import hashlib
import logging
import os
import shlex
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.utils import status_lib
from sky.adaptors import lsf as lsf_adaptor
from sky.provision import common
from sky.provision import constants as provision_constants
from sky.provision.lsf import utils as lsf_utils
from sky.skylet import constants
from sky.utils import command_runner
from sky.utils import subprocess_utils
from sky.utils import timeline

logger = sky_logging.init_logger(__name__)

_POLL_INTERVAL = 5

# Built-in shared network-filesystem roots bind-mounted *identity* into the
# enroot container (see the mount lines in _build_enroot_block). Passed to the
# runner as shared_fs_roots and surfaced to the backend via
# LsfCommandRunner.get_unwrapped_mount_prefixes() (which documents why these are
# exempt from symlink-wrapping).
#
# User-configured enroot_mounts are folded in on top of this built-in list by
# _derive_shared_fs_roots(); sky/clouds/lsf.py and lsf-ray.yml.j2 propagate them
# into provider_config. The built-in entries remain because they are mounted
# unconditionally by _build_enroot_block regardless of configuration.
#
# The classification is fail-closed: a root that is not recognized as shared is
# symlink-wrapped (a loud sudo failure), never silently redirected.
#
# TODO(dawood): this built-in list is maintained by hand alongside the identity
# mount lines in _build_enroot_block; if those change without updating this,
# file_mounts to a *built-in* root can silently break. Follow-up: derive both
# from one structured source (e.g. (path, is_shared) tuples).
_SHARED_FS_ROOTS = ['/proj', '/opt/share']

# Prefixes bind-mounted *identity* into the container but node-local (NOT shared
# across the login/compute split), so a login-node write is not visible to the
# job. Used to filter user-configured enroot_mounts when deriving shared roots
# (e.g. device mounts like /dev/shm, plus node-local scratch), preventing them
# from being wrongly exempted from the backend's symlink-wrap.
#
# NOTE: _is_shared_identity_mount treats this as a denylist -- an identity mount
# whose path is outside every prefix here is assumed shared. That fails open for
# an unrecognized node-local path (silent empty read in the job rather than a
# loud error), so keep this list conservative. /home is included because the
# container sets ENROOT_MOUNT_HOME=false (see _build_enroot_block) -- HOME is
# deliberately not the shared, container-visible path, so a /home identity mount
# must not be exempted.
_NODE_LOCAL_MOUNT_PREFIXES = ('/tmp', '/opt/nvme', '/dev', '/run', '/proc',
                              '/sys', '/var/tmp', '/home')


def _is_shared_identity_mount(mount_spec: str) -> bool:
    """Return whether an enroot mount spec is a shared, identity bind-mount.

    A spec qualifies only if its host and container paths are identical (an
    identity mount, so a login-node write is visible to the job at the same
    path) AND the path is absolute and not under a known node-local prefix
    (identity-mounted but not shared across the login/compute split).

    Args:
        mount_spec: an enroot mount string such as ``"/gpfs /gpfs"`` or
            ``"/dev/shm /dev/shm"`` (optionally followed by mount flags).

    Returns:
        True iff the spec is a shared, identity bind-mount usable for
        file_mount wrap-exemption.
    """
    parts = mount_spec.split()
    if len(parts) < 2 or parts[0] != parts[1]:
        return False
    path = parts[1]
    if not path.startswith('/'):
        return False
    return not any(path == p or path.startswith(p + '/')
                   for p in _NODE_LOCAL_MOUNT_PREFIXES)


def _derive_shared_fs_roots(enroot_mounts: List[str]) -> List[str]:
    """Derive the shared-FS wrap-exemption roots from the container's mounts.

    Unions the built-in ``_SHARED_FS_ROOTS`` with any of ``enroot_mounts`` that
    are shared identity mounts (see ``_is_shared_identity_mount``), so a user who
    bind-mounts an extra shared filesystem (e.g. ``/gpfs``) also gets their
    file_mounts to that root left un-wrapped. Node-local device/scratch mounts
    are excluded. Order is preserved and duplicates removed.

    ``enroot_mounts`` must be the list the container is actually built from
    (``provider_config['enroot_mounts']``, frozen into the bsub script at
    provision time and consumed by ``_build_enroot_block``). Deriving the
    exemption from that same source keeps it from ever claiming a root the
    container does not mount -- an unmounted root would silently redirect a
    file_mount to an empty path -- and, unlike live sky config, it cannot drift
    after launch.

    Args:
        enroot_mounts: the enroot bind-mount specs frozen into the container at
            provision time (empty when none are configured).

    Returns:
        The ordered, de-duplicated list of shared-FS root prefixes.
    """
    roots = list(_SHARED_FS_ROOTS)
    roots += [
        spec.split()[1]
        for spec in enroot_mounts
        if _is_shared_identity_mount(spec)
    ]
    return list(dict.fromkeys(roots))  # de-dupe, preserving order


# Precedence for #BSUB directives, lowest to highest:
#
#   1. derived defaults      — values the provisioner computes from a synthesized
#                              instance type the user never asked for
#   2. bsub_options          — cluster/queue config in ~/.sky/config.yaml, i.e. an
#                              environment-wide policy
#   3. explicit resources    — cpus/memory/accelerators the task actually
#                              requested; the most specific statement of intent
#
# Level 3 above level 2 matters: an environment setting `M: 64G` as its default
# must not cap a task that asks for 256G. Level 2 above level 1 matters because a
# synthesized default (LSF's catalog gives 16GB when nothing is requested) should
# never beat deliberate configuration.
#
# bsub options that LSF ACCUMULATES rather than resolving to a single value, so a
# user-supplied one must not suppress the derived one.
#
# -R is the case that matters: LSF ANDs multiple -R expressions, so a user adding
# `-R "rusage[mem=...]"` must not remove our `-R "span[ptile=N]"` — losing the
# span term would let LSF satisfy the slot count from fewer hosts than requested,
# silently collapsing a multi-node job. Every other option here is single-valued,
# where LSF honours the first occurrence and the user's value must therefore
# replace ours rather than follow it.
_ACCUMULATING_BSUB_FLAGS = frozenset({'R'})


def _get_client(provider_config: Dict[str, Any]) -> lsf_adaptor.LsfClient:
    """Create an LsfClient from provider config."""
    ssh_config = provider_config.get('ssh', {})
    is_local = provider_config.get('is_inside_lsf_cluster', False)

    if is_local:
        return lsf_adaptor.LsfClient(is_inside_lsf_cluster=True)

    return lsf_adaptor.LsfClient(
        ssh_host=ssh_config.get('hostname'),
        ssh_port=int(ssh_config.get('port', 22)),
        ssh_user=ssh_config.get('user'),
        ssh_key=ssh_config.get('private_key'),
        ssh_proxy_command=ssh_config.get('proxy_command'),
        ssh_proxy_jump=ssh_config.get('proxy_jump'),
        identities_only=ssh_config.get('identities_only', False),
    )


def _image_hash(image_id: str) -> str:
    """Generate a short hash for a Docker image URI (for cache filenames)."""
    return hashlib.sha256(image_id.encode()).hexdigest()[:16]


def _build_blaunch_dispatch(share_path: str, is_multinode: bool) -> str:
    """Build the blaunch dispatch block for multi-node jobs.

    When multi-node, the master node imports/flattens the sqsh, then uses
    blaunch to re-run the script on all allocated nodes. Workers (BV_WORKER=1)
    skip import and go directly to container create/start.
    """
    if not is_multinode:
        return ''
    return textwrap.dedent(f"""\
        # ── Multi-node blaunch dispatch (master only) ─────────────────────
        if [[ "${{BV_WORKER}}" != "1" && ${{TOTAL_NODES:-1}} -gt 1 ]]; then
            echo "[$(date)] Launching workers on $TOTAL_NODES nodes via blaunch"
            SHARED_SCRIPT="{share_path}/tmp/sky-worker-$LSB_JOBID.sh"
            mkdir -p "$(dirname "$SHARED_SCRIPT")"
            cp "$(realpath "${{BASH_SOURCE[0]}}")" "$SHARED_SCRIPT"
            export BV_WORKER=1
            # -z targets the deduplicated host list, one task per host. Bare
            # `blaunch` launches once per *slot* in $LSB_HOSTS, so a job that
            # requests several slots per host would start several dispatchers
            # on the same node, all racing over one dispatch directory.
            blaunch -z "${{UNIQUE_HOSTS[*]}}" bash "$SHARED_SCRIPT"
            exit $?
        fi
    """)


def _build_dispatcher_block(dispatch_root: str) -> str:
    """Build the per-host command dispatcher block.

    The dispatcher is the LSF substitute for ``srun``: the driver writes
    ``cmd_<seq>.sh`` into a directory on the shared filesystem, the dispatcher
    executes it and writes ``out_<seq>.log`` plus ``rc_<seq>``.

    The directory is keyed by short hostname, so every node gets its own. A
    single shared directory cannot work for multi-node: each blaunch task runs
    this same block, so each would ``rm -rf`` a directory the others are already
    using, and whichever dispatcher happened to notice a ``cmd_*.sh`` first would
    execute it — making the node that runs a given command nondeterministic.

    Keyed by hostname rather than by rank deliberately. Rank is derived
    independently on the two sides of this boundary (the driver from
    ``bjobs -o EXEC_HOST`` order, the job from ``$LSB_HOSTS`` order) and LSF
    guarantees no correspondence between them, whereas a short hostname means
    the same thing to both.
    """
    return textwrap.dedent(f"""\
        # ── Step 4: Command dispatcher (per host) ──────────────────────────────
        DISPATCH_DIR="{dispatch_root}/$(hostname -s)"
        rm -rf "$DISPATCH_DIR"
        mkdir -p "$DISPATCH_DIR"
        cat > "$DISPATCH_DIR/dispatcher.sh" << 'DISPATCH_EOF'
        #!/bin/bash
        DDIR="$1"
        touch "$DDIR/.ready"
        while true; do
            for cmd_file in "$DDIR"/cmd_*.sh; do
                [ -f "$cmd_file" ] || continue
                seq="${{cmd_file##*/cmd_}}"; seq="${{seq%.sh}}"
                /bin/bash "$cmd_file" > "$DDIR/out_${{seq}}.log" 2>&1
                echo $? > "$DDIR/rc_${{seq}}"
                mv "$cmd_file" "$DDIR/done_${{seq}}.sh"
            done
            [ -f "$DDIR/.shutdown" ] && break
            sleep 0.5
        done
        DISPATCH_EOF
        chmod +x "$DISPATCH_DIR/dispatcher.sh"
    """)


def _build_baremetal_exec_block(shared_dir: str, dispatch_root: str,
                                is_multinode: bool) -> str:
    """Build the execution block for a job with no container.

    Containerized jobs get their fan-out and dispatcher from
    ``_build_enroot_block``. Without a container there was nothing at all: the
    script computed topology, signalled ready, and slept on the first host, so a
    multi-node bare-metal allocation left every other node idle while the task
    ran on the login node instead.

    This emits the two pieces the container path relies on — ``blaunch`` to re-run
    the script on every host, and a per-host dispatcher to execute what the driver
    writes — without enroot.

    Args:
        shared_dir: directory on the shared filesystem for the blaunch worker
            script copy. The container path uses the enroot share_path; there is
            no enroot config here, so the caller passes the cluster home.
        dispatch_root: parent of the per-host dispatch directories.
        is_multinode: whether to fan out at all.

    Returns:
        The shell block; the dispatcher alone when single-node.
    """
    return f"""\
# === Bare-metal execution (no container) ===
{_build_blaunch_dispatch(shared_dir, is_multinode)}
{_build_dispatcher_block(dispatch_root)}
echo "[$(date)] Starting command dispatcher on $(hostname -s)"
bash "$DISPATCH_DIR/dispatcher.sh" "$DISPATCH_DIR" &
DISPATCH_PID=$!

echo "[$(date)] Waiting for dispatcher to be ready..."
READY_WAIT=0
while [ ! -f "$DISPATCH_DIR/.ready" ]; do
    sleep 0.5
    READY_WAIT=$((READY_WAIT + 1))
    if [ $READY_WAIT -gt 120 ]; then
        echo "ERROR: dispatcher did not become ready in 60s"
        exit 1
    fi
done
echo "[$(date)] Dispatcher ready (PID=$DISPATCH_PID), dispatch_dir=$DISPATCH_DIR"
"""


def _build_topology_block(num_nodes: int, sky_cluster_home: str) -> str:
    """Build the topology block: rank, world size, master address and port.

    Emitted for every job, containerized or not. Bare-metal multi-node jobs need
    the same values, and the rank manifest this writes is what lets the driver
    map a host to the rank the job actually assigned itself.

    The manifest is the authority on rank. The driver must not infer rank from
    the order of ``bjobs -o EXEC_HOST``: that is a separate derivation from the
    ``$LSB_HOSTS`` order used here, and a mismatched rank/master pairing does not
    fail fast — it hangs NCCL initialization until the collective timeout.
    """
    rank_detection = ''
    if num_nodes > 1:
        rank_detection = textwrap.dedent("""\
            # Deduplicate while preserving order (first host = rank 0 = master).
            # Order matters and must not be sorted: rank 0 is defined as the
            # first host LSF listed.
            UNIQUE_HOSTS=($(echo "$LSB_HOSTS" | tr ' ' '\\n' | awk '!seen[$0]++'))
            if [[ -n "$LSB_HOSTS" ]]; then
                for i in "${!UNIQUE_HOSTS[@]}"; do
                    if [[ "${UNIQUE_HOSTS[$i]}" == "$LOCAL_HOST" ]]; then
                        RANK=$i
                        break
                    fi
                done
            fi
        """)
    return textwrap.dedent("""\
        # === Compute topology ===
        NUM_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
        TOTAL_NODES=$(echo "${{LSB_HOSTS:-$(hostname)}}" | tr ' ' '\\n' | sort -u | wc -l)
        LOCAL_HOST=$(hostname -s)
        MASTER_HOST=$(echo "${{LSB_HOSTS:-$(hostname -s)}}" | awk '{{print $1}}')
        MASTER_PORT=$((29500 + (${{LSB_JOBID:-0}} % 1000)))

        RANK=0
        WORLD_SIZE=$TOTAL_NODES
        LOCAL_RANK=0
        {rank_detection}
        export MASTER_ADDR="$MASTER_HOST"
        export MASTER_PORT RANK WORLD_SIZE LOCAL_RANK
        export NUM_GPUS_PER_NODE TOTAL_NODES
        echo "[$(date)] Topology: node=$LOCAL_HOST rank=$RANK/$WORLD_SIZE gpus=$NUM_GPUS_PER_NODE master=$MASTER_HOST:$MASTER_PORT"

        # Publish this node's rank so the driver can map host -> rank without
        # re-deriving it from a different source.
        TOPOLOGY_DIR="{sky_cluster_home}/.sky/topology"
        mkdir -p "$TOPOLOGY_DIR"
        echo "$LOCAL_HOST" > "$TOPOLOGY_DIR/rank-$RANK"
        if [[ "$RANK" == "0" ]]; then
            echo "$MASTER_HOST:$MASTER_PORT" > "$TOPOLOGY_DIR/master"
        fi
    """).format(rank_detection=rank_detection,
                sky_cluster_home=sky_cluster_home)


def _convert_to_enroot_uri(image_id: str) -> str:
    """Convert Docker image URI to enroot format (registry/path → registry#path)."""
    registry = image_id.split('/')[0]
    if '/' in image_id and '.' in registry:
        remainder = image_id[len(registry) + 1:]
        return f'{registry}#{remainder}'
    return image_id


def _sqsh_name_from_image(image_id: str) -> str:
    """Derive sqsh filename from image name (shared across all jobs using same image).

    e.g. us.icr.io/cil15-shared-registry/sage-py311:0.025
      -> us.icr.io-cil15-shared-registry-sage-py311-0.025.sqsh
    """
    return image_id.replace('/', '-').replace(':', '-') + '.sqsh'


def _build_enroot_block(image_id: str, container_name: str,
                        enroot_config: Dict[str, Any],
                        env_vars: Dict[str, str],
                        mounts: List[str],
                        dispatch_root: str,
                        is_multinode: bool = False,
                        inject_topology: bool = True) -> str:
    """Build the full enroot setup block with BlueVela workarounds.

    Container isolation strategy (per gbansible guidelines):
    - Sqsh file: derived from IMAGE NAME, shared on shared FS (all jobs
      using the same image reuse the same sqsh)
    - Container name: derived from JOB NAME (cluster_name_on_cloud),
      unique per job — prevents concurrent jobs from destroying each
      other's containers via enroot create -f or enroot remove -f
    - inject_topology: when True, passes NUM_GPUS_PER_NODE, WORLD_SIZE,
      RANK, MASTER_ADDR, MASTER_PORT, and LSF job vars into the container
      so training scripts can derive distributed config
    """
    share_path = enroot_config.get('share_path', '/tmp')
    squash_options = enroot_config.get('squash_options',
                                       '-comp lz4 -Xhc -no-xattrs')
    use_nvme = enroot_config.get('use_local_nvme', False)

    # Strip the docker scheme so downstream callers (sqsh filename,
    # enroot URI conversion, and the `docker://` prefix added in the bsub
    # script) don't end up with `docker://docker://...` or filenames
    # starting with `docker---`.
    # SkyPilot passes image_id as 'docker:registry/...' (single colon)
    # while user YAML uses 'docker://registry/...' (double slash).
    if image_id.startswith('docker://'):
        image_id = image_id[len('docker://'):]
    elif image_id.startswith('docker:'):
        image_id = image_id[len('docker:'):]

    enroot_uri = _convert_to_enroot_uri(image_id)
    # Container name is job-specific (prevents concurrent job interference)
    container_name_safe = container_name.replace('/', '-').replace(':', '-')
    # Sqsh file is image-specific (shared across all jobs using same image)
    sqsh_filename = _sqsh_name_from_image(image_id)
    sqsh_file = f'{share_path}/enroot/{sqsh_filename}'

    enroot_data_path = ('/opt/nvme/$USER/enroot-data'
                        if use_nvme else f'{share_path}/user-$(id -u)/enroot-data')

    # Static env lines (written via quoted heredoc — no shell expansion)
    static_env_lines = '    echo "NVIDIA_VISIBLE_DEVICES=all"\n'
    static_env_lines += '    echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility"\n'
    static_env_lines += ('    echo "LD_LIBRARY_PATH='
                         '/opt/share/mpich-4.2.2/lib'
                         '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')
    for key, val in env_vars.items():
        static_env_lines += f'    echo "{key}={val}"\n'

    # Dynamic env lines (written via unquoted heredoc — vars expand at
    # config-write time from the topology_block shell vars).
    dynamic_env_lines = ''
    if inject_topology:
        dynamic_env_lines = (
            '    echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"\n'
            '    echo "TOTAL_NODES=${TOTAL_NODES}"\n'
            '    echo "RANK=${RANK}"\n'
            '    echo "WORLD_SIZE=${WORLD_SIZE}"\n'
            '    echo "LOCAL_RANK=${LOCAL_RANK}"\n'
            '    echo "MASTER_ADDR=${MASTER_HOST}"\n'
            '    echo "MASTER_PORT=${MASTER_PORT:-29500}"\n'
            '    echo "LSB_JOBID=${LSB_JOBID:-}"\n'
            '    echo "LSB_HOSTS=${LSB_HOSTS:-$(hostname)}"\n'
        )

    # Build mount lines for enroot config
    mount_lines = '    echo "/proj /proj"\n'
    mount_lines += '    echo "/tmp /tmp"\n'
    mount_lines += '    echo "/opt/nvme /opt/nvme"\n'
    mount_lines += '    echo "/opt/share /opt/share"\n'
    for m in mounts:
        mount_lines += f'    echo "{m}"\n'

    dispatcher_block = _build_dispatcher_block(dispatch_root)

    block = f"""\
# === Enroot container setup ===

# ── BlueVela workarounds ──────────────────────────────────────────────
BV_WRAPPER_DIR=$(mktemp -d -t bv-enroot-wrappers.XXXXXX)
export PATH="${{BV_WRAPPER_DIR}}:${{PATH}}"

if ! command -v fusermount &>/dev/null && command -v fusermount3 &>/dev/null; then
    ln -sf "$(command -v fusermount3)" "${{BV_WRAPPER_DIR}}/fusermount"
fi

printf '#!/bin/bash\\n/usr/bin/enroot-aufs2ovlfs "$@" || true\\n' \\
    > "${{BV_WRAPPER_DIR}}/enroot-aufs2ovlfs"
chmod +x "${{BV_WRAPPER_DIR}}/enroot-aufs2ovlfs"

cat > "${{BV_WRAPPER_DIR}}/enroot-mksquashovlfs" << 'WRAPPER'
#!/bin/bash
LAYERS="$1"; OUTFILE="$2"; shift 2
/usr/bin/enroot-mksquashovlfs "$LAYERS" "$OUTFILE" "$@" 2>/dev/null
if [ $? -eq 0 ] && [ -f "$OUTFILE" ]; then exit 0; fi
IFS=':' read -ra LAYER_DIRS <<< "$LAYERS"
mksquashfs "${{LAYER_DIRS[@]}}" "$OUTFILE" "$@" -no-xattrs
WRAPPER
chmod +x "${{BV_WRAPPER_DIR}}/enroot-mksquashovlfs"

# ── Enroot path setup ─────────────────────────────────────────────────
export ENROOT_DATA_PATH="{enroot_data_path}"
export ENROOT_CACHE_PATH="{share_path}/user-$(id -u)/enroot-cache"
export ENROOT_SQUASH_OPTIONS='{squash_options}'
export ENROOT_MOUNT_HOME=false
export ENROOT_RUNTIME_PATH="/tmp/user-$(id -u)/enroot-runtime"
export ENROOT_TEMP_PATH="/tmp/user-$(id -u)/enroot-tmp"
export XDG_RUNTIME_DIR="/tmp/user-$(id -u)/xdg-runtime"

mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" \\
         "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH" "$XDG_RUNTIME_DIR"

SQSH_FILE="{sqsh_file}"
CONTAINER_NAME="{container_name_safe}"
mkdir -p "$(dirname "$SQSH_FILE")"

# ── Helper: flatten layered sqsh ──────────────────────────────────────
flatten_sqsh_if_needed() {{
    local sqsh_file="$1"
    local mount_dir="/tmp/user-$(id -u)/sqsh-check"
    mkdir -p "$mount_dir"
    squashfuse "$sqsh_file" "$mount_dir" 2>/dev/null || return 0
    local is_layered=0
    if [[ -d "$mount_dir/0" ]] && [[ ! -d "$mount_dir/bin" ]]; then
        is_layered=1
    fi
    fusermount3 -u "$mount_dir" 2>/dev/null || fusermount -u "$mount_dir" 2>/dev/null || true

    if [[ $is_layered -eq 0 ]]; then
        echo "[$(date)] Sqsh is already flat"
        return 0
    fi

    echo "[$(date)] Sqsh has layered OCI structure, flattening..."
    local work_dir="/opt/nvme/$USER/flatten-work"
    local local_flat="/opt/nvme/$USER/$(basename "$sqsh_file" .sqsh)-flat.sqsh"
    rm -rf "$work_dir" "$local_flat"
    mkdir -p "$work_dir"/{{layers,merged,upper,work}}

    squashfuse "$sqsh_file" "$work_dir/layers"
    local lowerdir
    lowerdir=$(ls -d "$work_dir/layers"/*/ | sort -t/ -k7 -n -r | tr '\\n' ':' | sed 's/:$//')
    fuse-overlayfs -o "lowerdir=${{lowerdir}},upperdir=$work_dir/upper,workdir=$work_dir/work" "$work_dir/merged"

    echo "[$(date)] Creating flat sqsh on local NVME..."
    mksquashfs "$work_dir/merged" "$local_flat" -comp lz4 -Xhc -noappend >/dev/null 2>&1

    echo "[$(date)] Copying flat sqsh to shared filesystem..."
    cp "$local_flat" "$sqsh_file"
    chmod g+rw "$sqsh_file" 2>/dev/null || true
    echo "[$(date)] Flatten complete: $(du -h "$sqsh_file" | cut -f1)"

    fusermount3 -u "$work_dir/merged" 2>/dev/null || true
    fusermount3 -u "$work_dir/layers" 2>/dev/null || true
    rm -rf "$work_dir" "$local_flat"
}}

# ── Step 1: Import + Flatten (master only for multi-node) ─────────────
# Uses flock to prevent concurrent imports of the same image by
# multiple jobs. Only one job proceeds with import; others wait.
if [[ "${{BV_WORKER}}" != "1" ]]; then
    LOCK_FILE="${{SQSH_FILE}}.lock"
    (
        flock -x 200
        if [[ -f "$SQSH_FILE" ]]; then
            echo "[$(date)] Squash file exists: $SQSH_FILE ($(du -h "$SQSH_FILE" | cut -f1)), skipping import"
        else
            echo "[$(date)] Importing docker://{enroot_uri} → $SQSH_FILE"
            if ! enroot import -o "$SQSH_FILE" "docker://{enroot_uri}"; then
                if [[ -f "$SQSH_FILE" ]] && [[ -s "$SQSH_FILE" ]]; then
                    echo "[$(date)] Import completed with warnings (sqsh file was created)"
                else
                    echo "ERROR: enroot import failed for {image_id}"
                    exit 1
                fi
            fi
            chmod g+rw "$SQSH_FILE" 2>/dev/null || true
            echo "[$(date)] Import complete: $(du -h "$SQSH_FILE" | cut -f1)"
            flatten_sqsh_if_needed "$SQSH_FILE"
        fi
    ) 200>"$LOCK_FILE"
fi
{_build_blaunch_dispatch(share_path, is_multinode)}
# ── NVME pre-flight check ─────────────────────────────────────────────
NVME_USAGE=$(df /opt/nvme 2>/dev/null | awk 'NR==2 {{print $5}}' | sed 's/%//')
if [[ -n "$NVME_USAGE" && $NVME_USAGE -gt 85 ]]; then
    echo "[$(date)] WARNING: /opt/nvme is ${{NVME_USAGE}}% full"
fi

# ── Step 2: Create container (per-node) ───────────────────────────────
echo "[$(date)] Creating container '$CONTAINER_NAME' from $SQSH_FILE"
enroot create -f -n "$CONTAINER_NAME" "$SQSH_FILE" || {{
    echo "ERROR: enroot create failed"
    exit 1
}}
chmod -R a+rw "$ENROOT_DATA_PATH/$CONTAINER_NAME" 2>/dev/null || true

# Kill catatonit orphans spawned by THIS container's create only.
# Scoped to children of this shell to avoid killing other jobs'
# catatonit processes (which would destroy their containers).
pkill -9 -P $$ -x catatonit 2>/dev/null || true

# Replace entrypoint with passthrough
CONTAINER_RC="$ENROOT_DATA_PATH/$CONTAINER_NAME/etc/rc"
if [[ -f "$CONTAINER_RC" ]]; then
    printf '#!/bin/sh\\nexec "$@"\\n' > "$CONTAINER_RC"
fi

# Validate container filesystem
CONTAINER_SHELL="$ENROOT_DATA_PATH/$CONTAINER_NAME/bin/sh"
if [[ ! -f "$CONTAINER_SHELL" ]]; then
    echo "[$(date)] ERROR: Container filesystem incomplete (missing /bin/sh)"
    enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
    rm -rf "$ENROOT_DATA_PATH/$CONTAINER_NAME" 2>/dev/null || true
    exit 1
fi

# ── Step 3: Generate enroot config and start ──────────────────────────
ENROOT_CONFIG_FILE=$(mktemp -t enroot.config.XXXXXX)
# Static part (quoted heredoc — no shell expansion)
cat > "$ENROOT_CONFIG_FILE" << 'ENROOT_CFG_STATIC'
environ() {{
    env | grep -v '^PATH=\\|^HOME=\\|^LANG=\\|^HOSTNAME='
    echo "HOME=/"
    echo "PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
{static_env_lines}ENROOT_CFG_STATIC
# Dynamic part (unquoted heredoc — RANK/WORLD_SIZE expand now)
cat >> "$ENROOT_CONFIG_FILE" << ENROOT_CFG_DYNAMIC
{dynamic_env_lines}}}
mounts() {{
{mount_lines}}}
ENROOT_CFG_DYNAMIC

{dispatcher_block}
echo "[$(date)] Starting container with command dispatcher"
enroot start --conf "$ENROOT_CONFIG_FILE" --rw "$CONTAINER_NAME" \\
    bash "$DISPATCH_DIR/dispatcher.sh" "$DISPATCH_DIR" &
ENROOT_PID=$!

# Wait for container to signal readiness
echo "[$(date)] Waiting for container dispatcher to be ready..."
READY_WAIT=0
while [ ! -f "$DISPATCH_DIR/.ready" ]; do
    sleep 0.5
    READY_WAIT=$((READY_WAIT + 1))
    if [ $READY_WAIT -gt 120 ]; then
        echo "ERROR: Container dispatcher did not become ready in 60s"
        exit 1
    fi
done
echo "[$(date)] Enroot container ready (PID=$ENROOT_PID), dispatch_dir=$DISPATCH_DIR"
"""
    return block


def _build_bsub_script(
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    num_nodes: int,
) -> str:
    """Build a bsub script for provisioning virtual instances."""
    cluster = lsf_utils.get_lsf_cluster_from_config(provider_config)
    queue = lsf_utils.get_queue_from_config(provider_config)

    # Resource configuration
    cpus = provider_config.get('cpus', '4')
    memory = provider_config.get('memory', '16')
    acc_count = provider_config.get('accelerator_count', '0')
    acc_type = provider_config.get('accelerator_type', '')
    image_id = provider_config.get('image_id', '')
    # Normalize: strip docker scheme prefix (SkyPilot passes 'docker:...',
    # user YAML may pass 'docker://...'). Enroot import adds its own prefix.
    if image_id.startswith('docker://'):
        image_id = image_id[len('docker://'):]
    elif image_id.startswith('docker:'):
        image_id = image_id[len('docker:'):]
    enroot_enabled = provider_config.get('enroot_enabled', 'False') == 'True'

    # Directories
    workdir = provider_config.get('workdir', '')
    tmpdir = provider_config.get('tmpdir', '')
    if not workdir:
        workdir = f'~/sky_workdir/{cluster_name_on_cloud}'
    if not tmpdir:
        tmpdir = f'/tmp/skypilot/{cluster_name_on_cloud}'

    sky_cluster_home = f'{workdir}/{cluster_name_on_cloud}'

    # Derived directives, as (flag, value) pairs rather than pre-rendered text,
    # so a user-supplied bsub_options entry can REPLACE one instead of being
    # appended after it. LSF honours the FIRST occurrence of a single-valued
    # option, so appending a user's value after ours silently ignored it — a
    # 16G default -M shadowed an environment's `M: 64G` and got a real training
    # job killed with TERM_MEMLIMIT at 46.5G of usage.
    # (flag, value, explicit): `explicit` marks a value the task actually
    # requested, which outranks bsub_options. Bookkeeping directives are never
    # explicit, so an operator can always override them.
    derived: List[Tuple[str, Optional[str], bool]] = [
        ('J', cluster_name_on_cloud, False),
        ('o', f'{sky_cluster_home}/sky_logs/%J.out', False),
        ('e', f'{sky_cluster_home}/sky_logs/%J.err', False),
    ]

    memory_explicit = provider_config.get('memory_explicit', 'False') == 'True'
    cpus_explicit = provider_config.get('cpus_explicit', 'False') == 'True'

    if queue:
        derived.append(('q', queue, False))

    # Slots and their distribution. LSF's -n counts *slots*, not hosts, so a
    # request for N nodes with C CPUs each is N*C slots pinned to C per host by
    # span[ptile=C]. The ptile term is what makes the node count real: without
    # it LSF is free to satisfy -n from any mix of hosts, so `-n 4` could land
    # as four slots on one machine.
    #
    # cpus was previously read and then never used in any directive: -n carried
    # the node count alone, so every node got exactly one slot no matter what
    # was requested.
    try:
        cpus_per_node = max(1, int(float(cpus)))
    except (TypeError, ValueError):
        logger.warning(f'Could not parse cpus={cpus!r} for '
                       f'{cluster_name_on_cloud}; requesting 1 slot per node.')
        cpus_per_node = 1

    derived.append(('n', str(num_nodes * cpus_per_node), cpus_explicit))
    if cpus_per_node > 1 or num_nodes > 1:
        derived.append(('R', f'"span[ptile={cpus_per_node}]"', cpus_explicit))
    if num_nodes > 1:
        # Host-level limits: resource limits apply per host rather than to the
        # job as a whole. NOTE this is also what makes -M enforced — without it
        # LSF treats the memory limit as advisory, which is why a single-node job
        # can exceed it and survive while a multi-node one is killed.
        derived.append(('hl', None, False))

    # GPU allocation. `num=` is per *task* by default, and a task is a slot, so
    # once there is more than one slot per host a per-task count multiplies:
    # num=8 with 4 slots/host would ask for 32 GPUs on every host and the job
    # would never schedule. Say /host explicitly in that case to keep the
    # request per node. At one slot per host the two are equivalent, and the
    # bare form is kept so existing clusters see a byte-identical directive.
    if int(acc_count) > 0:
        per = '/host' if cpus_per_node > 1 else ''
        # Accelerators are only ever present because the task asked for them.
        derived.append(
            ('gpu', f'"num={acc_count}{per}:mode=exclusive_process"', True))

    mem_gb = int(float(memory))
    if mem_gb > 0:
        derived.append(('M', f'{mem_gb}G', memory_explicit))

    # Custom bsub options from config (per-queue overrides take precedence)
    bsub_options = dict(provider_config.get('bsub_options', {}))
    if queue:
        queue_configs = provider_config.get('queue_configs', {})
        if queue in queue_configs:
            bsub_options.update(queue_configs[queue].get('bsub_options', {}))

    # Flags whose explicit request beats bsub_options; the bsub_options entry is
    # then dropped rather than emitted alongside, since LSF would honour whichever
    # came first and two values for one flag is never what was meant.
    explicit_wins = {
        flag
        for flag, _, explicit in derived
        if explicit and flag not in _ACCUMULATING_BSUB_FLAGS
    }

    bsub_directives = []
    for flag, value, explicit in derived:
        overridden = (flag in bsub_options and
                      flag not in _ACCUMULATING_BSUB_FLAGS and not explicit)
        if overridden:
            logger.debug(
                f'LSF: bsub_options -{flag}={bsub_options[flag]!r} overrides the '
                f'derived default -{flag} {value!r} for {cluster_name_on_cloud}.')
            continue
        bsub_directives.append(f'#BSUB -{flag}' +
                               (f' {value}' if value is not None else ''))
    for flag, value in bsub_options.items():
        if flag in explicit_wins:
            logger.info(
                f'LSF: the task explicitly requested -{flag}, so it takes '
                f'precedence over bsub_options -{flag}={value!r} for '
                f'{cluster_name_on_cloud}.')
            continue
        bsub_directives.append(f'#BSUB -{flag} {value}')

    directives_str = '\n'.join(bsub_directives)

    # NCCL tuning
    nccl_tuning = provider_config.get('nccl_tuning_file', '')
    nccl_block = ''
    if nccl_tuning:
        nccl_block = f'[ -f "{nccl_tuning}" ] && source "{nccl_tuning}"'

    # Container block. dispatch_root is a directory *of* per-host dispatch
    # directories; the bsub script appends $(hostname -s) to it.
    dispatch_root = f'{sky_cluster_home}/.sky/dispatch'
    container_block = ''
    if image_id and enroot_enabled:
        enroot_config = {
            'share_path': provider_config.get('enroot_share_path', '/tmp'),
            'squash_options': provider_config.get('enroot_squash_options',
                                                   '-comp lz4 -Xhc -no-xattrs'),
            'use_local_nvme': (
                provider_config.get('enroot_use_local_nvme', 'False') == 'True'
            ),
        }
        extra_mounts = provider_config.get('enroot_mounts', [])
        container_block = _build_enroot_block(
            image_id=image_id,
            container_name=cluster_name_on_cloud,
            enroot_config=enroot_config,
            env_vars={},
            mounts=extra_mounts,
            dispatch_root=dispatch_root,
            is_multinode=(num_nodes > 1),
            inject_topology=True,
        )

    if not container_block:
        # No container: fan out and run a dispatcher on each host, so the task
        # executes on the allocated nodes rather than the login node. The cluster
        # home stands in for the enroot share_path as the shared location for the
        # blaunch worker-script copy.
        container_block = _build_baremetal_exec_block(
            shared_dir=sky_cluster_home,
            dispatch_root=dispatch_root,
            is_multinode=(num_nodes > 1),
        )

    # Topology is computed for every job, not just containerized ones: a
    # bare-metal multi-node job needs the same rank/master values, and the rank
    # manifest it publishes is what the driver reads to map host -> rank.
    topology_block = _build_topology_block(num_nodes, sky_cluster_home)

    # Marker file and ready signals. Each node signals separately, so the
    # driver can wait for the whole allocation rather than for whichever node
    # happened to finish first — with one shared file, a 4-node job reports
    # ready as soon as one node's container is up.
    marker_file = f'{sky_cluster_home}/{lsf_utils.LSF_MARKER_FILE}'
    ready_signal = f'{sky_cluster_home}/.sky_ready'
    ready_signal_host = f'{ready_signal}.$(hostname -s)'

    # NOTE: this template is written flush-left on purpose, and does NOT go
    # through textwrap.dedent. dedent computes the longest common leading
    # whitespace over all lines, and the interpolated blocks below
    # (directives_str, topology_block, container_block) contribute
    # column-0 lines, which drives that common prefix to "" and makes
    # dedent a silent no-op. Three things here require column 0:
    #   - the `#!` line, which is only a shebang at the start of a line;
    #   - `#BSUB` directives, which LSF ignores unless unindented;
    #   - heredoc terminators (the block builders emit `<< 'EOF'`, not `<<-`).
    # _build_enroot_block builds its block flush-left for the same reason.
    script = f"""\
#!/bin/bash
{directives_str}

# === SkyPilot LSF provisioner ===
set -e

cleanup() {{
    local exit_code=$?
    set +e
    echo "[$(date)] Cleaning up SkyPilot LSF instance..."
    # Signal this node's dispatcher to shut down gracefully
    [ -n "${{DISPATCH_DIR:-}}" ] && [ -d "${{DISPATCH_DIR}}" ] && \
        touch "${{DISPATCH_DIR}}/.shutdown"
    # Kill background processes (enroot dispatcher, etc.)
    kill $(jobs -p) 2>/dev/null || true
    # Kill catatonit orphans (scoped to this job's process tree)
    pkill -9 -P $$ -x catatonit 2>/dev/null || true
    # Remove enroot container if it exists
    if command -v enroot &>/dev/null && [[ -n "${{CONTAINER_NAME:-}}" ]]; then
        enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
    # Remove temp wrapper directory
    [[ -n "${{BV_WRAPPER_DIR:-}}" && -d "${{BV_WRAPPER_DIR:-}}" ]] && rm -rf "$BV_WRAPPER_DIR"
    echo "[$(date)] Cleanup done (exit code: $exit_code)"
    exit $exit_code
}}
trap cleanup EXIT
trap 'exit 0' TERM

# Create directories
mkdir -p "{sky_cluster_home}/sky_logs" "{sky_cluster_home}/.sky"
mkdir -p "{tmpdir}"

# Remove this node's stale ready signal from previous runs. Scoped to
# this host: a worker must not delete a peer's fresh signal.
rm -f "{ready_signal_host}"

# Write marker file
touch "{marker_file}"

{nccl_block}

{topology_block}

{container_block}

# Signal ready: this node always, plus the legacy shared path from
# rank 0 so an older driver still sees a cluster come up.
touch "{ready_signal_host}"
if [[ "${{RANK:-0}}" == "0" ]]; then
    touch "{ready_signal}"
fi
echo "SkyPilot LSF instance ready: {cluster_name_on_cloud} (node $(hostname -s), rank ${{RANK:-0}})"

# Keep job alive until terminated. Both modes run a dispatcher in the background
# and wait on it; the sleep is a fallback for a job that has neither.
if [[ -n "${{ENROOT_PID:-}}" ]]; then
    wait $ENROOT_PID
elif [[ -n "${{DISPATCH_PID:-}}" ]]; then
    wait $DISPATCH_PID
else
    sleep infinity
fi
"""

    return script


@timeline.event
def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Submit an LSF job as a virtual instance."""
    provider_config = config.provider_config
    num_nodes = config.count

    client = _get_client(provider_config)
    queue = lsf_utils.get_queue_from_config(provider_config)

    # Check for existing job with same name
    existing_states = client.get_jobs_state_by_name(cluster_name_on_cloud)
    running_states = [s for s in existing_states
                      if s in (lsf_adaptor.LSF_STATE_RUN,
                               lsf_adaptor.LSF_STATE_PEND)]
    if running_states:
        # Resume existing job
        job_ids = client.query_jobs(job_name=cluster_name_on_cloud,
                                    state_filters=[lsf_adaptor.LSF_STATE_RUN,
                                                   lsf_adaptor.LSF_STATE_PEND])
        if job_ids:
            job_id = job_ids[0]
            logger.info(f'Resuming existing LSF job {job_id} for '
                        f'{cluster_name_on_cloud}')
            nodes, _ = client.get_job_nodes(job_id)
            instance_ids = [lsf_utils.instance_id(job_id, n) for n in nodes]
            return common.ProvisionRecord(
                provider_name='lsf',
                region=region,
                zone=queue,
                cluster_name=cluster_name,
                head_instance_id=instance_ids[0],
                resumed_instance_ids=instance_ids,
                created_instance_ids=[],
            )

    # Build and upload job script
    script_content = _build_bsub_script(
        cluster_name_on_cloud, provider_config, num_nodes)

    # Write script to a temp file and upload
    cluster = lsf_utils.get_lsf_cluster_from_config(provider_config)
    workdir = provider_config.get('workdir', '') or f'~/sky_workdir'
    remote_script_dir = f'{workdir}/{cluster_name_on_cloud}/.sky'
    remote_script_path = f'{remote_script_dir}/provision.sh'

    # Create remote dir and write script
    ssh_config = provider_config.get('ssh', {})
    runner = client._runner
    rc, _, stderr = runner.run(
        f'mkdir -p {shlex.quote(remote_script_dir)}',
        require_outputs=True, separate_stderr=True, stream_logs=False)
    if rc != 0:
        raise RuntimeError(f'Failed to create script directory: {stderr}')

    # Write script via pipe (rsync fails on systems with login banners)
    encoded = base64.b64encode(script_content.encode()).decode()
    rc, _, stderr = runner.run(
        f'echo {shlex.quote(encoded)} | base64 -d > '
        f'{shlex.quote(remote_script_path)} && '
        f'chmod +x {shlex.quote(remote_script_path)}',
        require_outputs=True, separate_stderr=True, stream_logs=False)
    if rc != 0:
        raise RuntimeError(f'Failed to write provision script: {stderr}')

    # Remove stale ready signals and the rank manifest before submitting, so
    # the waiter cannot be satisfied by leftovers from a previous run of the
    # same cluster name. Covers both the shared file and the per-host ones.
    sky_cluster_home = f'{workdir}/{cluster_name_on_cloud}'
    ready_file = f'{sky_cluster_home}/.sky_ready'
    topology_dir = f'{sky_cluster_home}/.sky/topology'
    runner.run(
        f'rm -f {shlex.quote(ready_file)} {shlex.quote(ready_file)}.* ; '
        f'rm -rf {shlex.quote(topology_dir)}',
        require_outputs=True, separate_stderr=True, stream_logs=False)

    # Submit job
    job_id = client.submit_job(
        queue=queue,
        job_name=cluster_name_on_cloud,
        script_path=remote_script_path,
    )
    logger.info(f'Submitted LSF job {job_id} for {cluster_name_on_cloud}')

    # Wait for job to get nodes allocated. The timeouts are resolved by
    # sky/clouds/lsf.py (which knows the queue) and passed through the
    # provider config; fall back to the defaults for clusters provisioned by
    # an older config that predates these keys.
    provision_timeout = _get_timeout(provider_config, 'provision_timeout',
                                     lsf_utils.DEFAULT_PROVISION_TIMEOUT)
    nodes = _wait_for_job_nodes(client, job_id, cluster_name_on_cloud,
                                provision_timeout)
    instance_ids = [lsf_utils.instance_id(job_id, n) for n in nodes]

    # Wait for bsub script to signal readiness (includes container setup)
    image_id = provider_config.get('image_id', '')
    enroot_enabled = provider_config.get('enroot_enabled', 'False') == 'True'
    if image_id and enroot_enabled:
        ready_timeout = _get_timeout(provider_config, 'ready_timeout',
                                     lsf_utils.DEFAULT_READY_TIMEOUT)
        _wait_for_all_ready_signals(runner, ready_file, cluster_name_on_cloud,
                                    nodes, ready_timeout)

    return common.ProvisionRecord(
        provider_name='lsf',
        region=region,
        zone=queue,
        cluster_name=cluster_name,
        head_instance_id=instance_ids[0],
        resumed_instance_ids=[],
        created_instance_ids=instance_ids,
    )


def _get_timeout(provider_config: Dict[str, Any], key: str,
                 default: int) -> int:
    """Read a timeout (seconds) from the provider config.

    The value arrives as a string via the Jinja-rendered cluster YAML.
    """
    value = provider_config.get(key)
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f'Ignoring non-integer {key} {value!r} in LSF provider '
                       f'config; using {default}s.')
        return default


def _wait_str(timeout: int) -> str:
    return 'indefinitely' if timeout < 0 else f'up to {timeout}s'


def _wait_for_job_nodes(client: lsf_adaptor.LsfClient,
                        job_id: str,
                        cluster_name: str,
                        timeout: int = lsf_utils.DEFAULT_PROVISION_TIMEOUT
                        ) -> List[str]:
    """Wait until an LSF job has nodes allocated.

    A negative timeout waits indefinitely. Note that a pending job is the
    normal state on a busy queue, so this can legitimately block for hours.
    """
    logger.info(f'Waiting {_wait_str(timeout)} for LSF job {job_id} '
                f'({cluster_name}) to be allocated nodes.')
    start_time = time.time()
    while timeout < 0 or time.time() - start_time < timeout:
        state = client.get_job_state(job_id)
        if state is None:
            raise RuntimeError(
                f'LSF job {job_id} for {cluster_name} not found.')

        if state in lsf_adaptor.LSF_TERMINAL_STATES:
            raise RuntimeError(
                f'LSF job {job_id} for {cluster_name} terminated with '
                f'state {state} before nodes were allocated.')

        if state == lsf_adaptor.LSF_STATE_RUN:
            if client.check_job_has_nodes(job_id):
                nodes, _ = client.get_job_nodes(job_id)
                logger.info(f'Job {job_id} running on nodes: {nodes}')
                return nodes

        time.sleep(_POLL_INTERVAL)

    raise TimeoutError(
        f'Timed out waiting for LSF job {job_id} ({cluster_name}) '
        f'to get nodes allocated after {timeout}s.')


def _wait_for_all_ready_signals(
        runner,
        ready_file: str,
        cluster_name: str,
        nodes: List[str],
        timeout: int = lsf_utils.DEFAULT_READY_TIMEOUT) -> None:
    """Wait until every allocated node has signalled readiness.

    Each node touches ``<ready_file>.<short hostname>``; this waits for all of
    them. Waiting on a single shared file would return as soon as the fastest
    node was up, so the driver could start dispatching work to nodes whose
    container had not been created yet.

    Hostnames come from the LSF allocation, which may report fully-qualified
    names while the job writes the short form (the script uses ``hostname -s``),
    so compare on the first label only.

    A negative timeout waits indefinitely. This window covers ``enroot import``,
    which is slow on a cold cache.
    """
    short_names = [n.split('.')[0] for n in nodes]
    expected = [f'{ready_file}.{n}' for n in short_names]
    logger.info(f'Waiting {_wait_str(timeout)} for {len(expected)} node(s) to '
                f'signal readiness: {ready_file}.<host>')
    start_time = time.time()
    while timeout < 0 or time.time() - start_time < timeout:
        test_expr = ' && '.join(f'test -f {shlex.quote(f)}' for f in expected)
        rc, _, _ = runner.run(test_expr,
                              require_outputs=True,
                              separate_stderr=True,
                              stream_logs=False)
        if rc == 0:
            logger.info(f'All {len(expected)} node(s) ready for {cluster_name}')
            return
        time.sleep(_POLL_INTERVAL)

    # Name the outstanding nodes: with a multi-node job the useful question is
    # never "did it time out" but "which host is stuck".
    missing = []
    for path, name in zip(expected, short_names):
        rc, _, _ = runner.run(f'test -f {shlex.quote(path)}',
                              require_outputs=True,
                              separate_stderr=True,
                              stream_logs=False)
        if rc != 0:
            missing.append(name)
    detail = (f'Node(s) that never signalled: {", ".join(missing)}' if missing
              else 'all signals appeared during the final probe (the timeout is '
              'too tight for this cluster)')
    raise TimeoutError(
        f'Timed out waiting for container readiness for {cluster_name} '
        f'after {timeout}s. {detail}')


def wait_instances(
    region: str,
    cluster_name_on_cloud: str,
    state: Optional['status_lib.ClusterStatus'],
) -> None:
    """Wait for instances — no-op since run_instances already waits."""
    del region, cluster_name_on_cloud, state


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
) -> common.ClusterInfo:
    """Get information about the running LSF cluster."""
    client = _get_client(provider_config)
    ssh_config = provider_config.get('ssh', {})

    # Find the running job
    job_ids = client.query_jobs(
        job_name=cluster_name_on_cloud,
        state_filters=[lsf_adaptor.LSF_STATE_RUN])

    if not job_ids:
        return common.ClusterInfo(
            instances={},
            head_instance_id=None,
            provider_name='lsf',
            provider_config=provider_config,
        )

    job_id = job_ids[0]
    nodes, node_ips = client.get_job_nodes(job_id)

    instances = {}
    for i, (node, ip) in enumerate(zip(nodes, node_ips)):
        inst_id = lsf_utils.instance_id(job_id, node)
        instances[inst_id] = [
            common.InstanceInfo(
                instance_id=inst_id,
                internal_ip=ip,
                external_ip=ssh_config.get('hostname'),
                tags={
                    provision_constants.TAG_SKYPILOT_CLUSTER_NAME:
                        cluster_name_on_cloud,
                    'job_id': job_id,
                    'node': node,
                    'rank': str(i),
                },
                ssh_port=int(ssh_config.get('port', 22)),
            )
        ]

    head_instance_id = lsf_utils.instance_id(job_id, nodes[0])

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name='lsf',
        provider_config=provider_config,
        ssh_user=ssh_config.get('user'),
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Optional[Tuple[Optional['status_lib.ClusterStatus'],
                               Optional[str]]]]:
    """Query instance statuses."""
    client = _get_client(provider_config)

    # Map LSF states to SkyPilot ClusterStatus
    state_map = {
        lsf_adaptor.LSF_STATE_PEND: status_lib.ClusterStatus.INIT,
        lsf_adaptor.LSF_STATE_RUN: status_lib.ClusterStatus.UP,
        lsf_adaptor.LSF_STATE_WAIT: status_lib.ClusterStatus.INIT,
    }

    # Query jobs by name
    all_job_ids = client.query_jobs(job_name=cluster_name_on_cloud)
    if not all_job_ids:
        return {}

    result = {}
    for job_id in all_job_ids:
        state = client.get_job_state(job_id)
        if state is None:
            continue

        cluster_status = state_map.get(state)

        if non_terminated_only and cluster_status is None:
            continue

        # Try to get nodes for running jobs
        if state == lsf_adaptor.LSF_STATE_RUN:
            try:
                nodes, _ = client.get_job_nodes(job_id)
                for node in nodes:
                    inst_id = lsf_utils.instance_id(job_id, node)
                    result[inst_id] = (cluster_status, None)
            except Exception:  # pylint: disable=broad-except
                result[job_id] = (cluster_status, None)
        else:
            result[job_id] = (cluster_status, state)

    return result


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    worker_only: bool = False,
) -> None:
    """Stop is not supported for LSF."""
    raise NotImplementedError('LSF does not support stopping instances. '
                              'Use terminate_instances instead.')


@timeline.event
def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    worker_only: bool = False,
) -> None:
    """Terminate LSF job(s) for the cluster."""
    del worker_only  # LSF jobs are all-or-nothing

    client = _get_client(provider_config)
    client.cancel_jobs_by_name(cluster_name_on_cloud)
    logger.info(f'Terminated LSF jobs for {cluster_name_on_cloud}')


def cleanup_cluster_resources(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Cleanup cluster resources. No-op for LSF (no auxiliary resources)."""
    pass


def _sky_cluster_home_dir(base_dir: str, cluster_name_on_cloud: str) -> str:
    """Returns SkyPilot's home directory for this cluster on the LSF node."""
    return f'{base_dir}/.sky_clusters/{cluster_name_on_cloud}'


def _skypilot_runtime_dir(sky_base_dir: str,
                          cluster_name_on_cloud: str) -> str:
    """Returns the SkyPilot runtime directory on the LSF cluster.

    Uses the shared workdir (not tmpdir) so it's accessible from login nodes.
    """
    return os.path.join(sky_base_dir, '.sky_clusters',
                        cluster_name_on_cloud)


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Any,
) -> List[command_runner.LsfCommandRunner]:
    """Get command runners for each instance in the cluster.

    For LSF, commands are routed through the login node via SSH. Uses
    LsfCommandRunner, which handles the banned rsync wrapper via --rsync-path
    and is given this cluster's shared-FS roots (built-in _SHARED_FS_ROOTS plus
    shared identity enroot_mounts, resolved by _derive_shared_fs_roots) for
    file_mount wrap-exemption.
    """
    del credentials  # Use provider_config SSH info instead

    assert cluster_info.provider_config is not None, cluster_info
    provider_config = cluster_info.provider_config
    ssh_config = provider_config.get('ssh', {})

    if cluster_info.head_instance_id is None:
        return []

    head_instance = cluster_info.get_head_instance()
    assert head_instance is not None, 'Head instance not found'
    cluster_name_on_cloud = head_instance.tags.get(
        provision_constants.TAG_SKYPILOT_CLUSTER_NAME, None)
    assert cluster_name_on_cloud is not None, cluster_info

    instances = [
        instance_infos[0] for instance_infos in cluster_info.instances.values()
    ]

    login_node_ssh_hostname = ssh_config.get('hostname', '')
    login_node_ssh_port = int(ssh_config.get('port', 22))
    login_node_ssh_user = ssh_config.get('user', '')
    login_node_ssh_private_key = ssh_config.get('private_key', None)
    login_node_ssh_proxy_command = ssh_config.get('proxy_command', None)
    login_node_ssh_proxy_jump = ssh_config.get('proxy_jump', None)

    ssh_control_name = command_runner.DEFAULT_SSH_CONTROL_NAME

    lsf_cluster_name = provider_config.get('cluster')
    workdir = lsf_utils.get_workdir(lsf_cluster_name) if lsf_cluster_name else None
    tmpdir = lsf_utils.get_tmpdir(lsf_cluster_name) if lsf_cluster_name else None

    # Expand $USER in paths
    if tmpdir and '$USER' in tmpdir:
        tmpdir = tmpdir.replace('$USER', login_node_ssh_user)

    sky_base_dir = workdir if workdir is not None else f'/home/{login_node_ssh_user}'
    sky_cluster_home_dir = _sky_cluster_home_dir(sky_base_dir,
                                                  cluster_name_on_cloud)

    # Enable dispatch mode when container is active — routes commands
    # to the compute node via the shared-FS dispatcher in the container.
    image_id = provider_config.get('image_id')
    enroot_enabled = provider_config.get('enroot', {}).get('enabled', False)
    dispatch_dir = None
    if image_id and enroot_enabled:
        dispatch_dir = f'{sky_cluster_home_dir}/.sky/dispatch'

    # Shared-FS roots whose file_mounts the backend must not symlink-wrap.
    # Homogeneous per cluster, so derive once and log it: a misplaced payload
    # (e.g. an enroot_mount wrongly classified as shared) otherwise leaves no
    # trace to debug.
    shared_fs_roots = _derive_shared_fs_roots(
        provider_config.get('enroot_mounts', []))
    logger.debug(f'LSF file_mount wrap-exemption roots: {shared_fs_roots}')

    runners = [
        command_runner.LsfCommandRunner(
            (instance_info.external_ip or login_node_ssh_hostname,
             instance_info.ssh_port),
            login_node_ssh_user,
            login_node_ssh_private_key,
            sky_dir=sky_cluster_home_dir,
            skypilot_runtime_dir=_skypilot_runtime_dir(
                sky_base_dir, cluster_name_on_cloud),
            ssh_proxy_command=login_node_ssh_proxy_command,
            ssh_proxy_jump=login_node_ssh_proxy_jump,
            ssh_control_name=ssh_control_name,
            disable_identities_only=True,
            dispatch_dir=dispatch_dir,
            shared_fs_roots=shared_fs_roots,
        ) for instance_info in instances
    ]

    return runners


def open_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Open ports — not supported on LSF."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Cleanup ports — not supported on LSF."""
    del cluster_name_on_cloud, ports, provider_config
