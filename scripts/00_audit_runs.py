#!/usr/bin/env python
"""00_audit_runs.py - consistency audit of runs/ before trusting any number.

Written after a day of finding these one crash at a time. Every check here
corresponds to a defect that actually occurred in this project:

  DUP       two tuned runs for one (site, horizon, seed); consumers used
            opposite tie-break rules and silently disagreed on the model
  VNET_SRC  a variance net fitted on a mean model that is not the canonical
            one, so its swapped head cannot load into the member
  VNET_OLD  variance nets from before 2026-10-21 fit the squared residual in
            PHYSICAL units; mixing one in scales sigma_a wrong
  FOREIGN   a summary that records an absolute path from the other machine
  DANGLING  a recorded checkpoint that is not on this disk
  STALE     a derived result superseded by one with more ensemble members
  THIN      a derived result built on fewer members than are now on disk, which
            is how the deep ensembles ended up at M=2 for one cell and M=4 for
            the rest after the missing checkpoints were synced
  NESTED    a run directory inside another run directory, which rsync --relative
            produces when the source path carries a ./ marker, and which makes a
            run appear twice in any inventory
  V2GAP     a family not yet complete in the v2 protocol for all four seeds at
            both sites, which is the condition 10_paper_figures.py's site-aware
            v1/v2 selection waits on before it can be removed
  FIGORPHAN a figure cited by a .tex that no script in scripts/ mentions, so the
            repo cannot rebuild it. Dropping 09b/09c as "superseded" left the SGLD
            methodology note citing two such figures
  FIGMISS   a figure cited by a .tex that is not on disk at all
  RAW_SEL   a script that picks a run by sort order instead of run_select

Exit code is 1 if anything was found, so it can gate a results build:
    .venv/bin/python scripts/00_audit_runs.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from solar_uq.run_select import pick_best, scan_cell_runs  # noqa: E402

# Variance nets written before this date fit the residual in physical units.
VNET_UNIT_CUTOFF = 20261021
H_FROM_STEPS = {6: 1.0, 18: 3.0, 36: 6.0}
# ERROR means a number built now would be wrong or irreproducible; WARN means a
# human should know but the canonical rule already resolves it deterministically.
SEVERITY = {
    "VNET_SRC": "ERROR", "VNET_OLD": "ERROR", "DANGLING": "ERROR",
    "FOREIGN": "ERROR", "RAW_SEL": "ERROR", "BADJSON": "ERROR",
    "STALE": "WARN", "DUP": "WARN", "THIN": "WARN", "NESTED": "ERROR", "V2GAP": "WARN",
    "FIGORPHAN": "ERROR", "FIGMISS": "ERROR",
}
findings: list[tuple[str, str]] = []


def add(tag: str, msg: str) -> None:
    findings.append((tag, msg))


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        add("BADJSON", f"{p}: {e}")
        return None


def run_date(name: str) -> int | None:
    m = re.search(r"_(\d{8})_\d{6}$", name)
    return int(m.group(1)) if m else None


def cells_of(optuna_root: Path):
    """{(site, horizon): [run meta...]} over every completed run."""
    out = defaultdict(list)
    if not optuna_root.exists():
        return out
    for d in sorted(optuna_root.iterdir()):
        m = load(d / "summary.json") if (d / "summary.json").exists() else None
        if not m:
            continue
        site, h = m.get("site"), m.get("temporal", {}).get("horizon_hours")
        if site and h is not None:
            out[(site, float(h))].append((d, m))
    return out


def audit_optuna(optuna_root: Path) -> dict:
    """Report duplicate seeds; return {(site,h,seed): canonical ckpt}."""
    canon = {}
    for (site, h), runs in sorted(cells_of(optuna_root).items(), key=str):
        by_seed = defaultdict(list)
        for d, m in runs:
            by_seed[int(m["seed"])].append((d, m))
        for seed, rs in sorted(by_seed.items()):
            if len(rs) > 1:
                detail = ", ".join(
                    f"{d.name} (hidden_t={m['optuna']['best_params'].get('hidden_t')}, "
                    f"val={(m.get('best_model') or {}).get('final_val', {}).get('rmse_day')})"
                    for d, m in rs)
                add("DUP", f"{optuna_root.name} {site} h{h:g} seed {seed}: {len(rs)} runs -> {detail}")
            for d, m in rs:
                ck = (m.get("best_model") or {}).get("best_ckpt_path")
                if ck and not (PROJECT_ROOT / ck).exists() and not Path(ck).exists():
                    add("DANGLING", f"{d.name}: best_ckpt_path missing on disk ({ck})")
        best = {r["seed"]: r for r in
                [pick_best(scan_cell_runs(optuna_root, site, h, seed=s)) or {} for s in by_seed]
                if r}
        for seed, r in best.items():
            canon[(site, h, seed)] = Path(r["ckpt"])
    return canon


def audit_variance(canon: dict) -> None:
    root = PROJECT_ROOT / "runs" / "resnet_lstm_variance"
    if not root.exists():
        return
    newest_per_cell: dict[tuple, tuple[int, Path]] = {}
    for d in sorted(root.iterdir()):
        m = load(d / "summary.json") if (d / "summary.json").exists() else None
        if not m:
            continue
        key = (m.get("site"), float(m.get("horizon_hours", 0)), int(m.get("seed", -1)))
        dt = run_date(d.name) or 0
        prev = newest_per_cell.get(key)
        if prev is None or dt > prev[0]:
            newest_per_cell[key] = (dt, d)
        src = m.get("source_mean_ckpt")
        want = canon.get(key)
        if dt >= VNET_UNIT_CUTOFF and src and want:
            if Path(src).name != want.name or Path(src).parent.name != want.parent.name:
                add("VNET_SRC", f"{d.name}: fitted on {Path(src).parent.name}, canonical member is "
                               f"{want.parent.name} -> retrain: scripts/12_variance_net.py --arch resnet "
                               f"--site {key[0]} --hours_ahead {int(key[1])} --seed {key[2]} "
                               f"--optuna_version v2")
    for key, (dt, d) in sorted(newest_per_cell.items(), key=str):
        if dt < VNET_UNIT_CUTOFF:
            add("VNET_OLD", f"{key[0]} h{key[1]:g} seed {key[2]}: newest variance net is {d.name} "
                            f"(pre-{VNET_UNIT_CUTOFF} physical-unit convention); 08f will refuse it")


def audit_foreign_and_members(canon: dict) -> None:
    here = str(PROJECT_ROOT)
    for fam in ("resnet_lstm_sgld", "resnet_sgld_multichain", "resnet_deep_ensemble",
                "resnet_combined_uq", "resnet_mc_dropout", "resnet_lstm_variance"):
        root = PROJECT_ROOT / "runs" / fam
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = load(d / "summary.json") if (d / "summary.json").exists() else None
            if not m:
                continue
            blob = json.dumps(m)
            foreign = {a for a in re.findall(r'"(/[^"]*?/runs/[^"]*?)"', blob)
                       if not a.startswith(here)}
            if foreign:
                roots = sorted({a.split("/runs/")[0] for a in foreign})
                add("FOREIGN", f"{fam}/{d.name}: {len(foreign)} path(s) recorded from another "
                               f"machine ({', '.join(roots)}); the run is not reproducible here "
                               f"until its artefacts are synced and the paths made relative")
            for ck in (m.get("ensemble") or {}).get("member_ckpts", []) or []:
                if not (PROJECT_ROOT / ck).exists() and not Path(ck).exists():
                    add("DANGLING", f"{fam}/{d.name}: member checkpoint missing ({ck})")


def audit_stale_derived() -> None:
    """A cell with both _M2 and _M4 keeps a result built on fewer members."""
    for fam in ("resnet_deep_ensemble", "resnet_combined_uq"):
        root = PROJECT_ROOT / "runs" / fam
        if not root.exists():
            continue
        groups = defaultdict(list)
        for d in sorted(root.iterdir()):
            mm = re.match(r"(.+)_M(\d+)$", d.name)
            if mm:
                groups[mm.group(1)].append((int(mm.group(2)), d))
        for stem, members in sorted(groups.items()):
            if len(members) > 1:
                best = max(m[0] for m in members)
                for n, d in sorted(members):
                    if n < best:
                        add("STALE", f"{fam}/{d.name}: superseded by M{best}; delete it or a results "
                                     f"table may report the thinner ensemble")


def audit_thin_derived(canon: dict) -> None:
    """A derived result using fewer seeds than the cell now has members for."""
    avail = defaultdict(set)
    for (site, h, seed) in canon:
        avail[(site, h)].add(seed)
    H_OF = {"H6": 1.0, "H18": 3.0, "H36": 6.0}
    for fam, key in (("resnet_deep_ensemble", "ensemble"), ("resnet_combined_uq", "ensemble_seeds")):
        root = PROJECT_ROOT / "runs" / fam
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.name.startswith("_"):
                continue
            m = load(d / "summary.json") if (d / "summary.json").exists() else None
            if not m:
                continue
            used = m.get(key)
            if isinstance(used, dict):
                used = used.get("seeds")
            if not used:
                continue
            site = m.get("site")
            hs = next((v for k, v in H_OF.items() if f"_{k}_" in d.name), None)
            if site is None or hs is None:
                continue
            have = avail.get((site, hs), set())
            missing = sorted(have - set(used))
            if missing:
                add("THIN", f"{fam}/{d.name}: built on {len(used)} member(s) {sorted(used)} but "
                            f"seeds {missing} are now on disk too; rebuild for a symmetric grid")


V2_FAMILIES = ("resnet_lstm_optuna_v2", "graphsage_lstm_optuna_v2",
               "fusion_resnet_lstm", "fusion_graphsage_lstm")
SEEDS = (1, 7, 13, 42)


def audit_v2_coverage() -> None:
    """Which cells are still missing from the v2 protocol.

    10_paper_figures.py still selects Uniandes from the v1 study and El Paso from
    v2 (its src_by_site map). That asymmetry can only be removed once a family is
    complete in v2 for both sites, all three horizons and all four seeds, so make
    the remaining gap explicit instead of relying on memory.
    """
    for fam in V2_FAMILIES:
        root = PROJECT_ROOT / "runs" / fam
        if not root.exists():
            continue
        have = defaultdict(set)
        for d in sorted(root.iterdir()):
            if d.name.startswith("_"):
                continue
            m = load(d / "summary.json") if (d / "summary.json").exists() else None
            if not m:
                continue
            site, h = m.get("site"), m.get("temporal", {}).get("horizon_hours")
            if site and h is not None:
                have[(site, float(h))].add(int(m["seed"]))
        gaps = []
        for site in ("uniandes", "elpaso"):
            for h in (1.0, 3.0, 6.0):
                miss = [s for s in SEEDS if s not in have.get((site, h), set())]
                if miss:
                    gaps.append(f"{site} h{h:g} missing {miss}")
        if gaps:
            add("V2GAP", f"{fam}: {len(gaps)} incomplete cell(s) -> " + "; ".join(gaps))


def audit_nested_runs() -> None:
    """runs/<family>/<run>/summary.json is depth 3; anything deeper is a copy of a
    run inside a run, unless it sits under a retired, underscore-prefixed group."""
    root = PROJECT_ROOT / "runs"
    if not root.exists():
        return
    for sj in sorted(root.rglob("summary.json")):
        rel = sj.relative_to(root)
        if len(rel.parts) <= 3:
            continue
        if any(part.startswith("_") for part in rel.parts[:-1]):
            continue       # retired group, e.g. sarima/_superseded_static_forecast/<run>
        add("NESTED", f"runs/{rel.parent}: a run directory nested inside another; "
                      f"delete the inner copy so inventories do not double-count it")


def audit_figures() -> None:
    """Every figure our own .tex files cite must exist and be rebuildable.

    A public repo for reproducing results cannot cite a figure whose generator was
    deleted; that is how the SGLD methodology note ended up pointing at two PNGs
    nothing in scripts/ could produce.

    Scope is what this repo regenerates: the article and the methodology notes.
    docs/background/ archives earlier theses and docs/presentations/ holds slide
    assets carried over from them; neither is rebuilt here, and including them
    buried the two real findings under 151 false ones. Paths resolve by basename
    anywhere under docs/, which covers \\graphicspath{{figures/}} without parsing
    LaTeX.
    """
    docs = PROJECT_ROOT / "docs"
    if not docs.exists():
        return
    on_disk = {}
    for f in docs.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg", ".eps"):
            on_disk.setdefault(f.stem, f)
    scripts_text = [sp.read_text(encoding="utf-8", errors="replace")
                    for sp in sorted((PROJECT_ROOT / "scripts").glob("*.py"))]
    for tex in sorted(docs.rglob("*.tex")):
        parts = tex.relative_to(docs).parts
        if "background" in parts or "presentations" in parts:
            continue
        try:
            body = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        where = tex.relative_to(PROJECT_ROOT)
        for ref in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))):
            stem = Path(ref).stem
            if stem not in on_disk:
                add("FIGMISS", f"{where} cites {ref}, which is not under docs/")
                continue
            if not any(stem in t for t in scripts_text):
                add("FIGORPHAN", f"{where} cites {stem} ({on_disk[stem].relative_to(PROJECT_ROOT)}), "
                                 f"which no script in scripts/ mentions: the repo cannot rebuild it")


def audit_raw_selection() -> None:
    bad = []
    for f in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        if f.name == "00_audit_runs.py":
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        if "best_ckpt_path" not in t:
            continue
        if "run_select" in t:
            continue
        if re.search(r"sorted\(\s*\w*runs_root\.iterdir\(\)", t) or "match[-1]" in t:
            bad.append(f.name)
    for n in bad:
        add("RAW_SEL", f"{n}: selects a run by sort order and does not use solar_uq.run_select")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Consistency audit of runs/. Exits 1 if any ERROR-level finding is "
                    "present, so it can gate a results build.")
    ap.add_argument("--optuna_root", default="resnet_lstm_optuna_v2",
                    help="Optuna family under runs/ whose cells define the canonical models")
    ap.add_argument("--quiet", action="store_true", help="Print only the summary counts")
    args = ap.parse_args()

    canon = audit_optuna(PROJECT_ROOT / "runs" / args.optuna_root)
    audit_variance(canon)
    audit_foreign_and_members(canon)
    audit_stale_derived()
    audit_thin_derived(canon)
    audit_nested_runs()
    audit_v2_coverage()
    audit_figures()
    audit_raw_selection()

    if not findings:
        print("\nAUDIT CLEAN - no inconsistencies found.\n")
        return 0
    by_tag = defaultdict(list)
    for tag, msg in findings:
        by_tag[tag].append(msg)
    n_err = sum(len(v) for t, v in by_tag.items() if SEVERITY.get(t) == "ERROR")
    n_warn = len(findings) - n_err
    print(f"\nAUDIT: {n_err} error(s), {n_warn} warning(s)\n")
    if args.quiet:
        return 1 if n_err else 0
    for tag in sorted(by_tag, key=lambda t: (SEVERITY.get(t, "ERROR") != "ERROR", t)):
        print(f"[{SEVERITY.get(tag, 'ERROR')}] {tag}  ({len(by_tag[tag])})")
        for msg in by_tag[tag]:
            print(f"   {msg}")
        print()
    if n_err == 0:
        print("No errors: warnings are resolved deterministically by solar_uq.run_select.\n")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
