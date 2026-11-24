#!/usr/bin/env bash
set -euo pipefail

TIMEOUT_SEC=1800
INTERVAL_SEC=""
GPU_THRESH=5
PROC_PATTERN="python.*countzes_"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ec2_idle_stop.sh [options]

Auto-stop EC2 instance (sudo shutdown -h now) when both matched processes
and GPU utilization stay idle for TIMEOUT_SEC consecutive seconds.

Options:
  --timeout <sec>      Idle duration before shutdown (default: 1800 = 30m)
  --interval <sec>     Poll interval in seconds (default: 5 if timeout < 60, else 60)
  --gpu-thresh <pct>   GPU utilization % considered busy (default: 5)
  --pattern <regex>    Process regex passed to pgrep -af (default: python.*countzes_)
  --dry-run            Log shutdown action instead of powering off
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      TIMEOUT_SEC="${2:?missing value for --timeout}"
      shift 2
      ;;
    --interval)
      INTERVAL_SEC="${2:?missing value for --interval}"
      shift 2
      ;;
    --gpu-thresh)
      GPU_THRESH="${2:?missing value for --gpu-thresh}"
      shift 2
      ;;
    --pattern)
      PROC_PATTERN="${2:?missing value for --pattern}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INTERVAL_SEC" ]]; then
  if [[ "$TIMEOUT_SEC" -lt 60 ]]; then
    INTERVAL_SEC=5
  else
    INTERVAL_SEC=60
  fi
fi

if [[ "$DRY_RUN" -eq 0 ]] && command -v sudo >/dev/null 2>&1; then
  if ! sudo -n true 2>/dev/null; then
    echo "[WARN] Passwordless sudo is not available; 'sudo shutdown -h now' may fail or prompt." >&2
  fi
fi

echo "[ec2_idle_stop] Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ec2_idle_stop] timeout=${TIMEOUT_SEC}s | interval=${INTERVAL_SEC}s | gpu_thresh=${GPU_THRESH}% | pattern='${PROC_PATTERN}' | dry_run=${DRY_RUN}"

idle_sec=0
first_check=1
while true; do
  gpu_util=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
      raw_gpu="$(timeout 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
    else
      raw_gpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
    fi
    if [[ "$raw_gpu" =~ ^[0-9]+$ ]]; then
      gpu_util="$raw_gpu"
    fi
  fi

  has_proc=0
  if [[ -n "$PROC_PATTERN" ]] && command -v pgrep >/dev/null 2>&1; then
    has_proc="$( (pgrep -af "$PROC_PATTERN" 2>/dev/null || true) | awk -v self="$$" '$1 != self && $0 !~ /ec2_idle_stop/' | wc -l | tr -d '[:space:]' )"
  fi

  if [[ "${has_proc:-0}" -gt 0 || "${gpu_util:-0}" -ge "$GPU_THRESH" ]]; then
    if [[ "$idle_sec" -gt 0 || "$first_check" -eq 1 ]]; then
      echo "[$(date '+%H:%M:%S')] Busy (gpu=${gpu_util}%, proc=${has_proc}) -> idle counter=0s."
    fi
    idle_sec=0
  else
    echo "[$(date '+%H:%M:%S')] Idle ${idle_sec}/${TIMEOUT_SEC}s (gpu=${gpu_util}%, proc=${has_proc})"
    if [[ "$idle_sec" -ge "$TIMEOUT_SEC" ]]; then
      echo "[$(date '+%H:%M:%S')] Idle timeout reached (${TIMEOUT_SEC}s)."
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Would execute: sudo -n shutdown -h now"
      else
        echo "Executing: sudo -n shutdown -h now"
        sudo -n shutdown -h now || sudo shutdown -h now
      fi
      break
    fi
    idle_sec=$((idle_sec + INTERVAL_SEC))
  fi

  first_check=0
  sleep "$INTERVAL_SEC"
done
