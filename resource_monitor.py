"""Lightweight CPU, RAM, Process RSS, and CUDA VRAM profiler with Drive logging."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import platform
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional fallback for minimal environments
    psutil = None


def _read_linux_mem_mb() -> Dict[str, Optional[float]]:
    used_mb: Optional[float] = None
    total_mb: Optional[float] = None
    rss_mb: Optional[float] = None

    if os.path.exists("/proc/meminfo"):
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    val_str = parts[1].strip().split()[0]
                    if val_str.isdigit():
                        info[parts[0]] = int(val_str)
        if "MemTotal" in info:
            total_kb = info["MemTotal"]
            avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
            total_mb = total_kb / 1024.0
            used_mb = (total_kb - avail_kb) / 1024.0

    if os.path.exists("/proc/self/status"):
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    val_str = line.split(":", 1)[1].strip().split()[0]
                    if val_str.isdigit():
                        rss_mb = int(val_str) / 1024.0
                    break

    return {
        "ram_used_mb": used_mb,
        "ram_total_mb": total_mb,
        "ram_pct": (100.0 * used_mb / total_mb) if (used_mb is not None and total_mb) else None,
        "rss_mb": rss_mb,
    }


def _read_proc_stat_cpu() -> Optional[tuple]:
    if not os.path.exists("/proc/stat"):
        return None
    with open("/proc/stat", "r", encoding="utf-8") as f:
        first = f.readline().strip().split()
    if len(first) < 5 or first[0] != "cpu":
        return None
    vals = [int(x) for x in first[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    return idle, total


def _sample_cpu_ram(proc_handle, prev_proc_stat: Optional[tuple]) -> Dict[str, Any]:
    cpu_pct: Optional[float] = None
    next_proc_stat = prev_proc_stat

    if psutil is not None:
        cpu_pct = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        ram_total_mb = float(vm.total) / (1024.0 * 1024.0)
        ram_used_mb = float(vm.used) / (1024.0 * 1024.0)
        ram_pct = float(vm.percent)
        rss_mb = float(proc_handle.memory_info().rss) / (1024.0 * 1024.0) if proc_handle else None
    else:
        mem = _read_linux_mem_mb()
        ram_total_mb = mem["ram_total_mb"]
        ram_used_mb = mem["ram_used_mb"]
        ram_pct = mem["ram_pct"]
        rss_mb = mem["rss_mb"]

        cur_stat = _read_proc_stat_cpu()
        if cur_stat is not None:
            if prev_proc_stat is not None:
                d_idle = cur_stat[0] - prev_proc_stat[0]
                d_total = cur_stat[1] - prev_proc_stat[1]
                if d_total > 0:
                    cpu_pct = 100.0 * (1.0 - d_idle / float(d_total))
            next_proc_stat = cur_stat

    return {
        "cpu_pct": cpu_pct,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "ram_pct": ram_pct,
        "rss_mb": rss_mb,
        "_proc_stat": next_proc_stat,
    }


def _sample_vram(device_index: int = 0) -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "vram_torch_alloc_mb": None,
            "vram_torch_reserved_mb": None,
            "vram_dev_used_mb": None,
            "vram_dev_total_mb": None,
        }

    mb = 1024.0 * 1024.0
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    return {
        "gpu_name": torch.cuda.get_device_name(device_index),
        "vram_torch_alloc_mb": torch.cuda.memory_allocated(device_index) / mb,
        "vram_torch_reserved_mb": torch.cuda.memory_reserved(device_index) / mb,
        "vram_dev_used_mb": (total_b - free_b) / mb,
        "vram_dev_total_mb": total_b / mb,
    }


def _max_valid(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def _avg_valid(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return (sum(valid) / len(valid)) if valid else None


def _round_or_none(val: Optional[float], digits: int = 2) -> Optional[float]:
    return round(float(val), digits) if val is not None else None


def _fmt_mb(val: Optional[float]) -> str:
    return f"{val:,.1f} MB" if val is not None else "N/A"


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val:.1f}%" if val is not None else "N/A"


def get_hardware_summary(device_index: int = 0) -> Dict[str, Any]:
    """Return current machine CPU, RAM, and GPU specifications."""
    proc_handle = psutil.Process(os.getpid()) if psutil is not None else None
    mem = _sample_cpu_ram(proc_handle, None)
    vram = _sample_vram(device_index)
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": _round_or_none(mem["ram_total_mb"], 1),
        "gpu_name": vram["gpu_name"],
        "vram_total_mb": _round_or_none(vram["vram_dev_total_mb"], 1),
        "psutil_enabled": psutil is not None,
    }


def format_record_line(record: Dict[str, Any]) -> str:
    """Format a single resource usage record into a readable summary block."""
    meta = record.get("metadata") or {}
    meta_str = " | ".join(f"{k}={v}" for k, v in meta.items())
    header = (
        f"[{record['timestamp']}] stage={record['stage']} | "
        f"time={record['duration_sec']:.2f}s | status={record['status']}"
    )
    if meta_str:
        header += f" | {meta_str}"

    cpu_line = (
        f"  CPU : avg={_fmt_pct(record['cpu_avg_pct'])} | "
        f"peak={_fmt_pct(record['cpu_peak_pct'])} | cores={record['cpu_count']}"
    )
    ram_line = (
        f"  RAM : sys { _fmt_mb(record['ram_before_mb']) } -> { _fmt_mb(record['ram_after_mb']) } "
        f"(peak { _fmt_mb(record['ram_peak_mb']) } / { _fmt_mb(record['ram_total_mb']) }) | "
        f"RSS { _fmt_mb(record['rss_before_mb']) } -> { _fmt_mb(record['rss_after_mb']) } "
        f"(peak { _fmt_mb(record['rss_peak_mb']) })"
    )
    if record["vram_total_mb"] is not None:
        vram_line = (
            f"  VRAM: dev { _fmt_mb(record['vram_dev_before_mb']) } -> { _fmt_mb(record['vram_dev_after_mb']) } "
            f"(peak { _fmt_mb(record['vram_dev_peak_mb']) } / { _fmt_mb(record['vram_total_mb']) }) | "
            f"torch alloc { _fmt_mb(record['vram_torch_before_mb']) } -> { _fmt_mb(record['vram_torch_after_mb']) } "
            f"(peak alloc { _fmt_mb(record['vram_torch_peak_mb']) }, "
            f"peak reserved { _fmt_mb(record['vram_torch_reserved_peak_mb']) })"
        )
    else:
        vram_line = "  VRAM: CUDA not available"

    return "\n".join([header, cpu_line, ram_line, vram_line])


@contextmanager
def monitor_resources(
    stage: str,
    log_dir: Optional[str] = None,
    interval: float = 0.25,
    metadata: Optional[Dict[str, Any]] = None,
    device_index: int = 0,
    verbose: bool = True,
):
    """Profile wall time, CPU %, System RAM, Process RSS, and CUDA VRAM peaks."""
    proc_handle = psutil.Process(os.getpid()) if psutil is not None else None
    if psutil is not None:
        # Prime psutil internal CPU counter so first sampled delta is meaningful
        psutil.cpu_percent(interval=None)
    prev_stat = _read_proc_stat_cpu() if psutil is None else None

    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if has_cuda:
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

    before_cr = _sample_cpu_ram(proc_handle, prev_stat)
    prev_stat = before_cr["_proc_stat"]
    before_vr = _sample_vram(device_index)

    samples: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    lock = threading.Lock()

    def _poll():
        nonlocal prev_stat
        while not stop_event.wait(interval):
            cr = _sample_cpu_ram(proc_handle, prev_stat)
            prev_stat = cr["_proc_stat"]
            vr = _sample_vram(device_index)
            with lock:
                samples.append({**cr, **vr})

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    status = "ok"
    t0 = time.perf_counter()
    result_holder: Dict[str, Any] = {}
    try:
        yield result_holder
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        stop_event.set()
        poller.join(timeout=2.0)
        if has_cuda:
            torch.cuda.synchronize(device_index)
        duration = time.perf_counter() - t0

        after_cr = _sample_cpu_ram(proc_handle, prev_stat)
        after_vr = _sample_vram(device_index)

        mb = 1024.0 * 1024.0
        torch_peak_alloc = (
            torch.cuda.max_memory_allocated(device_index) / mb if has_cuda else None
        )
        torch_peak_res = (
            torch.cuda.max_memory_reserved(device_index) / mb if has_cuda else None
        )

        with lock:
            all_cpu = [s["cpu_pct"] for s in samples] + [after_cr["cpu_pct"]]
            all_ram = [before_cr["ram_used_mb"]] + [s["ram_used_mb"] for s in samples] + [after_cr["ram_used_mb"]]
            all_rss = [before_cr["rss_mb"]] + [s["rss_mb"] for s in samples] + [after_cr["rss_mb"]]
            all_vram_dev = [before_vr["vram_dev_used_mb"]] + [s["vram_dev_used_mb"] for s in samples] + [after_vr["vram_dev_used_mb"]]

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "stage": stage,
            "status": status,
            "duration_sec": round(duration, 3),
            "cpu_count": os.cpu_count(),
            "cpu_avg_pct": _round_or_none(_avg_valid(all_cpu), 1),
            "cpu_peak_pct": _round_or_none(_max_valid(all_cpu), 1),
            "ram_total_mb": _round_or_none(after_cr["ram_total_mb"], 1),
            "ram_before_mb": _round_or_none(before_cr["ram_used_mb"], 1),
            "ram_after_mb": _round_or_none(after_cr["ram_used_mb"], 1),
            "ram_peak_mb": _round_or_none(_max_valid(all_ram), 1),
            "rss_before_mb": _round_or_none(before_cr["rss_mb"], 1),
            "rss_after_mb": _round_or_none(after_cr["rss_mb"], 1),
            "rss_peak_mb": _round_or_none(_max_valid(all_rss), 1),
            "gpu_name": after_vr["gpu_name"],
            "vram_total_mb": _round_or_none(after_vr["vram_dev_total_mb"], 1),
            "vram_dev_before_mb": _round_or_none(before_vr["vram_dev_used_mb"], 1),
            "vram_dev_after_mb": _round_or_none(after_vr["vram_dev_used_mb"], 1),
            "vram_dev_peak_mb": _round_or_none(_max_valid(all_vram_dev), 1),
            "vram_torch_before_mb": _round_or_none(before_vr["vram_torch_alloc_mb"], 1),
            "vram_torch_after_mb": _round_or_none(after_vr["vram_torch_alloc_mb"], 1),
            "vram_torch_peak_mb": _round_or_none(torch_peak_alloc, 1),
            "vram_torch_reserved_peak_mb": _round_or_none(torch_peak_res, 1),
            "metadata": metadata or {},
        }
        result_holder["record"] = record

        formatted = format_record_line(record)
        if verbose:
            print("=== RESOURCE MONITOR ===")
            print(formatted)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_txt = os.path.join(log_dir, "resource_usage.log")
            log_jsonl = os.path.join(log_dir, "resource_usage.jsonl")
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(formatted + "\n" + "-" * 60 + "\n")
            with open(log_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_log_history(log_dir: str, last_n: int = 10) -> None:
    """Print the most recent resource usage records saved in log_dir."""
    log_jsonl = os.path.join(log_dir, "resource_usage.jsonl")
    if not os.path.exists(log_jsonl):
        print(f"Chưa có file log tại: {log_jsonl}")
        return
    with open(log_jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Hiển thị {min(last_n, len(lines))}/{len(lines)} bản ghi gần nhất từ {log_jsonl}:")
    for line in lines[-last_n:]:
        rec = json.loads(line)
        print(format_record_line(rec))
        print("-" * 60)
