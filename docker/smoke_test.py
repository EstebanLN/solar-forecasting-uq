"""Quick check that an environment can run the experiments.

Checks the frozen package versions, that CUDA works, and which data directories are
present. It uses a few hundred MB of GPU memory at most, so it is safe to run next to
training jobs.

    python docker/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

EXPECTED = {
    "torch": "2.11.0.dev20260206+cu130",
    "torchvision": "0.26.0.dev20260206+cu130",
    "optuna": None,
    "numpy": None,
    "pandas": None,
    "xarray": None,
    "netCDF4": None,
}


def main() -> int:
    ok = True
    print(f"python {sys.version.split()[0]}")

    for name, want in EXPECTED.items():
        try:
            mod = __import__(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] import {name}: {exc}")
            ok = False
            continue
        have = getattr(mod, "__version__", "?")
        flag = "ok" if want is None or have == want else "MISMATCH"
        ok &= flag == "ok"
        print(f"[{flag}] {name} {have}" + ("" if flag == "ok" else f" (expected {want})"))

    import torch

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        x = torch.randn(1024, 1024, device=dev)
        y = (x @ x.T).sum().item()
        print(f"[ok] cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)} "
              f"(matmul {y:.1f})")
        del x
        torch.cuda.empty_cache()
    else:
        print("[FAIL] CUDA not available (missing --gpus all or driver < 580?)")
        ok = False

    import solar_uq
    from solar_uq.zoo import PROJECT_ROOT

    print(f"[ok] solar_uq from {Path(solar_uq.__file__).parent}, PROJECT_ROOT={PROJECT_ROOT}")
    for sub in ("data/datasets", "data/patches_v1", "data/ground_aligned", "runs"):
        p = PROJECT_ROOT / sub
        print(f"[{'ok' if p.exists() else '--'}] {sub}" + ("" if p.exists() else " (not mounted)"))

    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
