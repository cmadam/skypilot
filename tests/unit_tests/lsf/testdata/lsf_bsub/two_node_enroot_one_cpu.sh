#!/bin/bash
#BSUB -J sky-gold-kd-abc123
#BSUB -o /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.out
#BSUB -e /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.err
#BSUB -n 2
#BSUB -R "span[ptile=1]"
#BSUB -hl
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -G grp_granite_dot_build
#BSUB -M 64G

# === SkyPilot LSF provisioner ===
set -e

cleanup() {
    local exit_code=$?
    set +e
    echo "[$(date)] Cleaning up SkyPilot LSF instance..."
    # Signal this node's dispatcher to shut down gracefully
    [ -n "${DISPATCH_DIR:-}" ] && [ -d "${DISPATCH_DIR}" ] &&         touch "${DISPATCH_DIR}/.shutdown"
    # Kill background processes (enroot dispatcher, etc.)
    kill $(jobs -p) 2>/dev/null || true
    # Kill catatonit orphans (scoped to this job's process tree)
    pkill -9 -P $$ -x catatonit 2>/dev/null || true
    # Remove enroot container if it exists
    if command -v enroot &>/dev/null && [[ -n "${CONTAINER_NAME:-}" ]]; then
        enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
    # Remove temp wrapper directory
    [[ -n "${BV_WRAPPER_DIR:-}" && -d "${BV_WRAPPER_DIR:-}" ]] && rm -rf "$BV_WRAPPER_DIR"
    echo "[$(date)] Cleanup done (exit code: $exit_code)"
    exit $exit_code
}
trap cleanup EXIT
trap 'exit 0' TERM

# Create directories
mkdir -p "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs" "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky"
mkdir -p "/opt/nvme/$USER/skypilot-tmp"

# Remove this node's stale ready signal from previous runs. Scoped to
# this host: a worker must not delete a peer's fresh signal.
rm -f "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_ready.$(hostname -s)"

# Write marker file
touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_lsf_cluster"

[ -f "/proj/granite-build/g4os/bv-nccl-tuning.sh" ] && source "/proj/granite-build/g4os/bv-nccl-tuning.sh"

# === Compute topology ===
NUM_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
TOTAL_NODES=$(echo "${LSB_HOSTS:-$(hostname)}" | tr ' ' '\n' | sort -u | wc -l)
LOCAL_HOST=$(hostname -s)
MASTER_HOST=$(echo "${LSB_HOSTS:-$(hostname -s)}" | awk '{print $1}')
MASTER_PORT=$((29500 + (${LSB_JOBID:-0} % 1000)))

RANK=0
WORLD_SIZE=$TOTAL_NODES
LOCAL_RANK=0
# Deduplicate while preserving order (first host = rank 0 = master).
# Order matters and must not be sorted: rank 0 is defined as the
# first host LSF listed.
UNIQUE_HOSTS=($(echo "$LSB_HOSTS" | tr ' ' '\n' | awk '!seen[$0]++'))
if [[ -n "$LSB_HOSTS" ]]; then
    for i in "${!UNIQUE_HOSTS[@]}"; do
        if [[ "${UNIQUE_HOSTS[$i]}" == "$LOCAL_HOST" ]]; then
            RANK=$i
            break
        fi
    done
fi

export MASTER_ADDR="$MASTER_HOST"
export MASTER_PORT RANK WORLD_SIZE LOCAL_RANK
export NUM_GPUS_PER_NODE TOTAL_NODES
echo "[$(date)] Topology: node=$LOCAL_HOST rank=$RANK/$WORLD_SIZE gpus=$NUM_GPUS_PER_NODE master=$MASTER_HOST:$MASTER_PORT"

# Publish this node's rank so the driver can map host -> rank without
# re-deriving it from a different source.
TOPOLOGY_DIR="/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky/topology"
mkdir -p "$TOPOLOGY_DIR"
echo "$LOCAL_HOST" > "$TOPOLOGY_DIR/rank-$RANK"
if [[ "$RANK" == "0" ]]; then
    echo "$MASTER_HOST:$MASTER_PORT" > "$TOPOLOGY_DIR/master"
fi


# === Enroot container setup ===

# ── BlueVela workarounds ──────────────────────────────────────────────
BV_WRAPPER_DIR=$(mktemp -d -t bv-enroot-wrappers.XXXXXX)
export PATH="${BV_WRAPPER_DIR}:${PATH}"

if ! command -v fusermount &>/dev/null && command -v fusermount3 &>/dev/null; then
    ln -sf "$(command -v fusermount3)" "${BV_WRAPPER_DIR}/fusermount"
fi

printf '#!/bin/bash\n/usr/bin/enroot-aufs2ovlfs "$@" || true\n' \
    > "${BV_WRAPPER_DIR}/enroot-aufs2ovlfs"
chmod +x "${BV_WRAPPER_DIR}/enroot-aufs2ovlfs"

cat > "${BV_WRAPPER_DIR}/enroot-mksquashovlfs" << 'WRAPPER'
#!/bin/bash
LAYERS="$1"; OUTFILE="$2"; shift 2
/usr/bin/enroot-mksquashovlfs "$LAYERS" "$OUTFILE" "$@" 2>/dev/null
if [ $? -eq 0 ] && [ -f "$OUTFILE" ]; then exit 0; fi
IFS=':' read -ra LAYER_DIRS <<< "$LAYERS"
mksquashfs "${LAYER_DIRS[@]}" "$OUTFILE" "$@" -no-xattrs
WRAPPER
chmod +x "${BV_WRAPPER_DIR}/enroot-mksquashovlfs"

# ── Enroot path setup ─────────────────────────────────────────────────
export ENROOT_DATA_PATH="/opt/nvme/$USER/enroot-data"
export ENROOT_CACHE_PATH="/proj/granite-build/g4os/user-$(id -u)/enroot-cache"
export ENROOT_SQUASH_OPTIONS='-comp lz4 -Xhc -no-xattrs'
export ENROOT_MOUNT_HOME=false
export ENROOT_RUNTIME_PATH="/tmp/user-$(id -u)/enroot-runtime"
export ENROOT_TEMP_PATH="/tmp/user-$(id -u)/enroot-tmp"
export XDG_RUNTIME_DIR="/tmp/user-$(id -u)/xdg-runtime"

mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" \
         "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH" "$XDG_RUNTIME_DIR"

SQSH_FILE="/proj/granite-build/g4os/enroot/us.icr.io-cil15-shared-registry-kd-sandbox-distill-0.1.0-uv.sqsh"
CONTAINER_NAME="sky-gold-kd-abc123"
mkdir -p "$(dirname "$SQSH_FILE")"

# ── Helper: flatten layered sqsh ──────────────────────────────────────
flatten_sqsh_if_needed() {
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
    mkdir -p "$work_dir"/{layers,merged,upper,work}

    squashfuse "$sqsh_file" "$work_dir/layers"
    local lowerdir
    lowerdir=$(ls -d "$work_dir/layers"/*/ | sort -t/ -k7 -n -r | tr '\n' ':' | sed 's/:$//')
    fuse-overlayfs -o "lowerdir=${lowerdir},upperdir=$work_dir/upper,workdir=$work_dir/work" "$work_dir/merged"

    echo "[$(date)] Creating flat sqsh on local NVME..."
    mksquashfs "$work_dir/merged" "$local_flat" -comp lz4 -Xhc -noappend >/dev/null 2>&1

    echo "[$(date)] Copying flat sqsh to shared filesystem..."
    cp "$local_flat" "$sqsh_file"
    chmod g+rw "$sqsh_file" 2>/dev/null || true
    echo "[$(date)] Flatten complete: $(du -h "$sqsh_file" | cut -f1)"

    fusermount3 -u "$work_dir/merged" 2>/dev/null || true
    fusermount3 -u "$work_dir/layers" 2>/dev/null || true
    rm -rf "$work_dir" "$local_flat"
}

# ── Step 1: Import + Flatten (master only for multi-node) ─────────────
# Uses flock to prevent concurrent imports of the same image by
# multiple jobs. Only one job proceeds with import; others wait.
if [[ "${BV_WORKER}" != "1" ]]; then
    LOCK_FILE="${SQSH_FILE}.lock"
    (
        flock -x 200
        if [[ -f "$SQSH_FILE" ]]; then
            echo "[$(date)] Squash file exists: $SQSH_FILE ($(du -h "$SQSH_FILE" | cut -f1)), skipping import"
        else
            echo "[$(date)] Importing docker://us.icr.io#cil15-shared-registry/kd-sandbox-distill:0.1.0-uv → $SQSH_FILE"
            if ! enroot import -o "$SQSH_FILE" "docker://us.icr.io#cil15-shared-registry/kd-sandbox-distill:0.1.0-uv"; then
                if [[ -f "$SQSH_FILE" ]] && [[ -s "$SQSH_FILE" ]]; then
                    echo "[$(date)] Import completed with warnings (sqsh file was created)"
                else
                    echo "ERROR: enroot import failed for us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv"
                    exit 1
                fi
            fi
            chmod g+rw "$SQSH_FILE" 2>/dev/null || true
            echo "[$(date)] Import complete: $(du -h "$SQSH_FILE" | cut -f1)"
            flatten_sqsh_if_needed "$SQSH_FILE"
        fi
    ) 200>"$LOCK_FILE"
fi
# ── Multi-node blaunch dispatch (master only) ─────────────────────
if [[ "${BV_WORKER}" != "1" && ${TOTAL_NODES:-1} -gt 1 ]]; then
    echo "[$(date)] Launching workers on $TOTAL_NODES nodes via blaunch"
    SHARED_SCRIPT="/proj/granite-build/g4os/tmp/sky-worker-$LSB_JOBID.sh"
    mkdir -p "$(dirname "$SHARED_SCRIPT")"
    cp "$(realpath "${BASH_SOURCE[0]}")" "$SHARED_SCRIPT"
    export BV_WORKER=1
    # -z targets the deduplicated host list, one task per host. Bare
    # `blaunch` launches once per *slot* in $LSB_HOSTS, so a job that
    # requests several slots per host would start several dispatchers
    # on the same node, all racing over one dispatch directory.
    blaunch -z "${UNIQUE_HOSTS[*]}" bash "$SHARED_SCRIPT"
    exit $?
fi

# ── NVME pre-flight check ─────────────────────────────────────────────
NVME_USAGE=$(df /opt/nvme 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//')
if [[ -n "$NVME_USAGE" && $NVME_USAGE -gt 85 ]]; then
    echo "[$(date)] WARNING: /opt/nvme is ${NVME_USAGE}% full"
fi

# ── Step 2: Create container (per-node) ───────────────────────────────
echo "[$(date)] Creating container '$CONTAINER_NAME' from $SQSH_FILE"
enroot create -f -n "$CONTAINER_NAME" "$SQSH_FILE" || {
    echo "ERROR: enroot create failed"
    exit 1
}
chmod -R a+rw "$ENROOT_DATA_PATH/$CONTAINER_NAME" 2>/dev/null || true

# Kill catatonit orphans spawned by THIS container's create only.
# Scoped to children of this shell to avoid killing other jobs'
# catatonit processes (which would destroy their containers).
pkill -9 -P $$ -x catatonit 2>/dev/null || true

# Replace entrypoint with passthrough
CONTAINER_RC="$ENROOT_DATA_PATH/$CONTAINER_NAME/etc/rc"
if [[ -f "$CONTAINER_RC" ]]; then
    printf '#!/bin/sh\nexec "$@"\n' > "$CONTAINER_RC"
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
environ() {
    env | grep -v '^PATH=\|^HOME=\|^LANG=\|^HOSTNAME='
    echo "HOME=/"
    echo "PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    echo "NVIDIA_VISIBLE_DEVICES=all"
    echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    echo "LD_LIBRARY_PATH=/opt/share/mpich-4.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENROOT_CFG_STATIC
# Dynamic part (unquoted heredoc — RANK/WORLD_SIZE expand now)
cat >> "$ENROOT_CONFIG_FILE" << ENROOT_CFG_DYNAMIC
    echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
    echo "TOTAL_NODES=${TOTAL_NODES}"
    echo "RANK=${RANK}"
    echo "WORLD_SIZE=${WORLD_SIZE}"
    echo "LOCAL_RANK=${LOCAL_RANK}"
    echo "MASTER_ADDR=${MASTER_HOST}"
    echo "MASTER_PORT=${MASTER_PORT:-29500}"
    echo "LSB_JOBID=${LSB_JOBID:-}"
    echo "LSB_HOSTS=${LSB_HOSTS:-$(hostname)}"
}
mounts() {
    echo "/proj /proj"
    echo "/tmp /tmp"
    echo "/opt/nvme /opt/nvme"
    echo "/opt/share /opt/share"
}
ENROOT_CFG_DYNAMIC

# ── Step 4: Command dispatcher (per host) ──────────────────────────────
DISPATCH_DIR="/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky/dispatch/$(hostname -s)"
rm -rf "$DISPATCH_DIR"
mkdir -p "$DISPATCH_DIR"
cat > "$DISPATCH_DIR/dispatcher.sh" << 'DISPATCH_EOF'
#!/bin/bash
DDIR="$1"
touch "$DDIR/.ready"
while true; do
    for cmd_file in "$DDIR"/cmd_*.sh; do
        [ -f "$cmd_file" ] || continue
        seq="${cmd_file##*/cmd_}"; seq="${seq%.sh}"
        /bin/bash "$cmd_file" > "$DDIR/out_${seq}.log" 2>&1
        echo $? > "$DDIR/rc_${seq}"
        mv "$cmd_file" "$DDIR/done_${seq}.sh"
    done
    [ -f "$DDIR/.shutdown" ] && break
    sleep 0.5
done
DISPATCH_EOF
chmod +x "$DISPATCH_DIR/dispatcher.sh"

echo "[$(date)] Starting container with command dispatcher"
enroot start --conf "$ENROOT_CONFIG_FILE" --rw "$CONTAINER_NAME" \
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


# Signal ready: this node always, plus the legacy shared path from
# rank 0 so an older driver still sees a cluster come up.
touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_ready.$(hostname -s)"
if [[ "${RANK:-0}" == "0" ]]; then
    touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_ready"
fi
echo "SkyPilot LSF instance ready: sky-gold-kd-abc123 (node $(hostname -s), rank ${RANK:-0})"

# Keep job alive until terminated. Both modes run a dispatcher in the background
# and wait on it; the sleep is a fallback for a job that has neither.
if [[ -n "${ENROOT_PID:-}" ]]; then
    wait $ENROOT_PID
elif [[ -n "${DISPATCH_PID:-}" ]]; then
    wait $DISPATCH_PID
else
    sleep infinity
fi
