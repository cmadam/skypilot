        #!/bin/bash
        #BSUB -J sky-gold-kd-abc123
#BSUB -o /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.out
#BSUB -e /proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/sky_logs/%J.err
#BSUB -n 1
#BSUB -M 64G

        # === SkyPilot LSF provisioner ===
        set -e

        cleanup() {
            local exit_code=$?
            set +e
            echo "[$(date)] Cleaning up SkyPilot LSF instance..."
            # Signal dispatcher to shut down gracefully
            [ -d "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky/dispatch" ] && touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky/dispatch/.shutdown"
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

        # Remove stale ready signal from previous runs
        rm -f "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_ready"

        # Write marker file
        touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_lsf_cluster"







        # Signal ready
        touch "/proj/granite-build/g4os/skypilot/sky-gold-kd-abc123/.sky_ready"
        echo "SkyPilot LSF instance ready: sky-gold-kd-abc123"

        # Keep job alive until terminated
        if [[ -n "${ENROOT_PID:-}" ]]; then
            # Container mode: wait for dispatcher to exit (or be killed)
            wait $ENROOT_PID
        else
            # Bare-metal mode: sleep forever
            sleep infinity
        fi
