#!/bin/bash
#BSUB -J sky-gold-kd-abc123
#BSUB -o /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.out
#BSUB -e /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.err
#BSUB -n 4
#BSUB -R "span[ptile=4]"
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



# === Compute topology ===
NUM_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
TOTAL_NODES=$(echo "${LSB_HOSTS:-$(hostname)}" | tr ' ' '\n' | sort -u | wc -l)
LOCAL_HOST=$(hostname -s)
MASTER_HOST=$(echo "${LSB_HOSTS:-$(hostname -s)}" | awk '{print $1}')
MASTER_PORT=$((29500 + (${LSB_JOBID:-0} % 1000)))

RANK=0
WORLD_SIZE=$TOTAL_NODES
LOCAL_RANK=0

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


# === Bare-metal execution (no container) ===

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
