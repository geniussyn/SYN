"""Logging, disk guard and provenance stamping."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SYN_TRAIN_ROOT", Path(__file__).resolve().parents[1]))
MIN_FREE_GB = float(os.environ.get("SYN_MIN_FREE_GB", "3.0"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def disk_guard(need_gb: float = 0.0, path: Path = ROOT, label: str = "") -> float:
    """Abort before writing if free space is below need_gb + MIN_FREE_GB."""
    free = free_gb(path)
    required = need_gb + MIN_FREE_GB
    if free < required:
        raise SystemExit(f"[disk] {label or path}: {free:.1f} GB free < "
                         f"{required:.1f} GB required")
    log(f"disk {label}: {free:.1f} GB free (need {required:.1f} GB)")
    return free


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _hardware() -> dict:
    hw = {"cpu_count": os.cpu_count()}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            hw.update(gpu_name=torch.cuda.get_device_name(0),
                      gpu_mem_gb=round(p.total_memory / 1e9, 1),
                      gpu_capability=f"{p.major}.{p.minor}",
                      cuda=torch.version.cuda,
                      bf16_supported=bool(torch.cuda.is_bf16_supported()))
        else:
            hw["gpu_name"] = None
    except Exception:
        pass
    return hw


def provenance(**extra) -> dict:
    import numpy
    pkgs = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    for name in ("torch", "transformers", "peft", "faiss", "pyarrow"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pass
    return {"git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "host_os": platform.platform(), "hardware": _hardware(),
            "packages": pkgs, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}


def write_json(path: Path, obj: dict) -> None:
    """Append-only: an existing file is kept and the new record timestamped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    log(f"wrote {path}")
