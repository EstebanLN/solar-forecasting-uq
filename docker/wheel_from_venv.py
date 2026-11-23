"""Rebuild a wheel from an installed distribution, byte for byte.

Used when the index now serves a different build under the same version (triton
3.6.0+git9844da95 was republished), so the only copy of what the experiments ran on
is the installed one. Every file is checked against the dist's RECORD before it is
packed, so a corrupted install is refused instead of frozen.

    .venv/bin/python docker/wheel_from_venv.py triton docker/wheelhouse
"""
from __future__ import annotations

import base64
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

SITE = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()


def main(name: str, out_dir: str) -> int:
    (dist,) = SITE.glob(f"{name}-*.dist-info")
    rows = list(csv.reader((dist / "RECORD").open()))
    version = dist.name[len(name) + 1 : -len(".dist-info")]
    tags = [l.split(": ", 1)[1].strip() for l in (dist / "WHEEL").read_text().splitlines() if l.startswith("Tag: ")]
    py = ".".join(sorted({t.split("-")[0] for t in tags}))
    abi = ".".join(sorted({t.split("-")[1] for t in tags}))
    plat = ".".join(sorted({t.split("-")[2] for t in tags}))
    skip = {f"{dist.name}/{n}" for n in ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")}

    bad, files = [], []
    for row in rows:
        rel = row[0]
        if rel in skip or "__pycache__" in rel or rel.startswith(".."):
            continue
        path = SITE / rel
        if row[1] and digest(path) != row[1]:
            bad.append(rel)
        files.append((rel, path, row))
    if bad:
        print(f"refusing: {len(bad)} files do not match RECORD, e.g. {bad[:3]}")
        return 1

    whl = Path(out_dir) / f"{name}-{version}-{py}-{abi}-{plat}.whl"
    record_name = f"{dist.name}/RECORD"
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        out_rows = []
        for rel, path, row in files:
            z.write(path, rel)
            out_rows.append(row[:3])
        out_rows.append([record_name, "", ""])
        lines = "".join(",".join(r) + "\n" for r in out_rows)
        z.writestr(record_name, lines)
    print(f"wrote {whl} ({len(files)} files, all verified against RECORD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
