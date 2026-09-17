#!/usr/bin/env python3
"""
aggregate_causal.py - Agrega todas las corridas de run_kg_causal.py.

    python aggregate_causal.py
    python aggregate_causal.py --dir results_causal --out analysis

Agrupa por (solver, brazo), reporta media +- desviacion, contrasta A frente a B
con Welch y Mann-Whitney, y escribe analysis/causal_summary.csv.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_causal")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "metrics.json"), recursive=True))
    if not paths:
        print(f"Sin metrics.json bajo {args.dir}")
        return

    groups = defaultdict(list)
    rows = []
    for p in paths:
        with open(p) as fh:
            m = json.load(fh)
        key = (m.get("solver", "?"), m.get("arm", "?"))
        groups[key].append(m)
        rows.append({
            "solver": m.get("solver"), "arm": m.get("arm"), "seed": m.get("seed"),
            "t0": m.get("t0"), "t1": m.get("t1"),
            "L2_u_window_pct": m.get("L2_u_window_pct"),
            "acum_interna": m.get("err_t_ratio_last_first_decile"),
            "loss_final": m.get("loss_final"),
            "total_time_s": m.get("total_time_s"),
            "run_dir": os.path.dirname(p),
        })

    with open(os.path.join(args.out, "causal_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{len(paths)} corridas encontradas\n")
    summary = {}
    for key in sorted(groups):
        solver, arm = key
        err = np.array([m["L2_u_window_pct"] for m in groups[key]], dtype=float)
        acc = np.array([m["err_t_ratio_last_first_decile"] for m in groups[key]], dtype=float)
        summary[key] = err
        sd = err.std(ddof=1) if err.size > 1 else 0.0
        sda = acc.std(ddof=1) if acc.size > 1 else 0.0
        print(f"[{solver} brazo {arm}] n={err.size}")
        print(f"   L2_u en la ventana : {err.mean():.3f} +- {sd:.3f} %"
              f"   valores: {np.round(err, 3).tolist()}")
        print(f"   acumulacion interna: {acc.mean():.2f} +- {sda:.2f}")

    print("\n--- contraste A frente a B ---")
    out_rows = []
    for solver in sorted({k[0] for k in summary}):
        A, B = summary.get((solver, "A")), summary.get((solver, "B"))
        if A is None or B is None:
            continue
        ratio = B.mean() / A.mean()
        line = f"[{solver}] A={A.mean():.3f}  B={B.mean():.3f}  razon B/A={ratio:.3f}"
        rec = {"solver": solver, "n_A": A.size, "n_B": B.size,
               "A_mean": A.mean(), "A_std": A.std(ddof=1) if A.size > 1 else 0.0,
               "B_mean": B.mean(), "B_std": B.std(ddof=1) if B.size > 1 else 0.0,
               "ratio_B_over_A": ratio}
        if HAVE_SCIPY and A.size > 1 and B.size > 1:
            t, p = stats.ttest_ind(B, A, equal_var=False)
            u, pu = stats.mannwhitneyu(B, A, alternative="two-sided")
            pooled = np.sqrt((A.var(ddof=1) + B.var(ddof=1)) / 2)
            d = (B.mean() - A.mean()) / pooled if pooled else np.nan
            line += f"   Welch p={p:.3f}  MW p={pu:.3f}  d={d:.2f}"
            rec.update({"welch_t": t, "welch_p": p, "mw_p": pu, "cohen_d": d})
        print("  " + line)
        out_rows.append(rec)

    if out_rows:
        with open(os.path.join(args.out, "causal_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)

    print(f"\nCSV en {args.out}/causal_runs.csv y {args.out}/causal_summary.csv")
    print("Referencia de dominio completo (calculada aparte con --compare-full):")
    print("  QCPINN  temprana 4.150   tardia 6.542   razon 1.58")
    print("  PINN    temprana 3.261   tardia 5.270   razon 1.62")


if __name__ == "__main__":
    main()
