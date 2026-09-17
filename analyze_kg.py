#!/usr/bin/env python3
"""
analyze_kg.py - Agregacion multi-semilla y diagnostico para las corridas de run_kg.py.

Uso tipico (despues de correr 10 semillas de cada modelo):

    python analyze_kg.py \
        --runs "qcpinn_cascade=results/kg_dv_angle_cascade_seed*" \
               "pinn_model2=results/kg_classical_seed*" \
        --out analysis/

Produce en --out:
    summary_runs.csv      una fila por corrida
    summary_groups.csv    media +- desviacion por grupo (protocolo de 10 semillas)
    loss_components.png   L_f y L_u promediadas por grupo, con banda de dispersion
    grad_variance.png     Var[dL/dtheta_q] por epoca (diagnostico de meseta esteril)
    spectral.png          espectro de amplitud en t: prediccion vs exacta (sesgo espectral)
    error_map_<grupo>.png |u_pred - u_exact| sobre el dominio
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEYS = ["n_params", "mean_iter_time_s", "total_time_s", "peak_rss_mb", "loss_final",
        "L2_u_global_pct", "L2_f_global_pct", "L2_u_t0_pct",
        "L2_u_boundary_pct", "L2_u_interior_pct"]


def find_runs(pattern):
    """Devuelve los directorios que contienen metrics.json bajo el patron dado."""
    dirs = []
    for base in glob.glob(pattern):
        for mj in glob.glob(os.path.join(base, "**", "metrics.json"), recursive=True):
            dirs.append(os.path.dirname(mj))
    return sorted(set(dirs))


def load_csv(run_dir):
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for col in rows[0].keys():
        vals = []
        for r in rows:
            v = r[col]
            vals.append(float(v) if v not in ("", None) else np.nan)
        out[col] = np.array(vals)
    return out


def stack_curves(curves, col):
    """Alinea curvas de distinta longitud al minimo comun y devuelve matriz (n_runs, n_pts)."""
    series = [c[col] for c in curves if c is not None and col in c]
    series = [s for s in series if np.isfinite(s).any()]
    if not series:
        return None, None
    n = min(len(s) for s in series)
    M = np.vstack([s[:n] for s in series])
    epochs = curves[0]["epoch"][:n]
    return epochs, M


def band_energy(freqs, amps, cut=1.5):
    lo = amps[freqs < cut].sum()
    hi = amps[freqs >= cut].sum()
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help='pares "etiqueta=patron_glob"')
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--hf-cut", type=float, default=1.5,
                    help="frontera baja/alta frecuencia en ciclos por unidad de t")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups = {}
    for spec in args.runs:
        label, pattern = spec.split("=", 1)
        groups[label] = find_runs(pattern)
        print(f"{label}: {len(groups[label])} corridas")
        if not groups[label]:
            print(f"  aviso: el patron '{pattern}' no encontro metrics.json")

    # ---------- tablas ----------
    rows, grows = [], []
    for label, dirs in groups.items():
        vals = {k: [] for k in KEYS}
        for d in dirs:
            with open(os.path.join(d, "metrics.json")) as fh:
                m = json.load(fh)
            rows.append({"group": label, "run_dir": d, "seed": m.get("seed"),
                         **{k: m.get(k) for k in KEYS}})
            for k in KEYS:
                if isinstance(m.get(k), (int, float)):
                    vals[k].append(m[k])
        if dirs:
            g = {"group": label, "n_runs": len(dirs)}
            for k in KEYS:
                arr = np.array(vals[k], dtype=float)
                g[f"{k}_mean"] = float(arr.mean()) if arr.size else np.nan
                g[f"{k}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            grows.append(g)

    if rows:
        with open(os.path.join(args.out, "summary_runs.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if grows:
        with open(os.path.join(args.out, "summary_groups.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(grows[0].keys()))
            w.writeheader()
            w.writerows(grows)
        print("\n--- media +- std por grupo ---")
        for g in grows:
            print(f"\n[{g['group']}] n={g['n_runs']}")
            for k in KEYS:
                print(f"  {k:22s} {g[k+'_mean']:.4g} +- {g[k+'_std']:.4g}")

    # ---------- curvas L_f vs L_u ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for label, dirs in groups.items():
        curves = [load_csv(d) for d in dirs]
        curves = [c for c in curves if c is not None]
        if not curves:
            continue
        for ax, col, title in zip(axes, ["L_f_raw", "L_u_raw"],
                                  ["$L_f$ (residuo fisico)", "$L_u$ (anclaje IC/BC)"]):
            ep, M = stack_curves(curves, col)
            if M is None:
                continue
            mu, sd = M.mean(0), M.std(0)
            ax.plot(ep, mu, label=label)
            ax.fill_between(ep, np.maximum(mu - sd, 1e-12), mu + sd, alpha=0.2)
            ax.set_yscale("log")
            ax.set_xlabel("epoca")
            ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss_components.png"), dpi=150)
    plt.close(fig)

    # ---------- varianza de gradientes cuanticos ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, dirs in groups.items():
        curves = [load_csv(d) for d in dirs]
        curves = [c for c in curves if c is not None and "grad_var_q" in c]
        if not curves:
            continue
        ep, M = stack_curves(curves, "grad_var_q")
        if M is None:
            continue
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(M, axis=0)
        good = np.isfinite(mu)
        if good.sum() == 0:
            continue
        ax.plot(ep[good], mu[good], label=label)
        plotted = True
    if plotted:
        ax.set_yscale("log")
        ax.set_xlabel("epoca")
        ax.set_ylabel(r"Var$[\partial L/\partial\theta_q]$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "grad_variance.png"), dpi=150)
    plt.close(fig)

    # ---------- sesgo espectral + mapa de error ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    spectral_report = {}
    for label, dirs in groups.items():
        if not dirs:
            continue
        d = dirs[0]
        npz_path = os.path.join(d, "eval.npz")
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path)
        n = int(round(np.sqrt(z["X"].shape[0])))
        U_pred = z["u_pred"].reshape(n, n)      # [indice x, indice t]
        U_exact = z["u_star"].reshape(n, n)

        dt = 1.0 / (n - 1)
        freqs = np.fft.rfftfreq(n, d=dt)
        A_pred = np.abs(np.fft.rfft(U_pred, axis=1)).mean(0)
        A_exact = np.abs(np.fft.rfft(U_exact, axis=1)).mean(0)

        lo_p, hi_p = band_energy(freqs, A_pred, args.hf_cut)
        lo_e, hi_e = band_energy(freqs, A_exact, args.hf_cut)
        spectral_report[label] = {
            "low_band_ratio_pred_over_exact": lo_p / lo_e if lo_e else np.nan,
            "high_band_ratio_pred_over_exact": hi_p / hi_e if hi_e else np.nan,
        }

        ax.semilogy(freqs, A_pred, label=f"{label} (pred)")
        if label == list(groups.keys())[0]:
            ax.semilogy(freqs, A_exact, "k--", label="exacta")

        E = np.abs(U_pred - U_exact)
        f2, a2 = plt.subplots(figsize=(5.2, 4))
        im = a2.imshow(E, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="Reds")
        a2.set_xlabel("t"); a2.set_ylabel("x")
        a2.set_title(f"|u_pred - u_exact| : {label}")
        f2.colorbar(im, ax=a2)
        f2.tight_layout()
        f2.savefig(os.path.join(args.out, f"error_map_{label}.png"), dpi=150)
        plt.close(f2)

    ax.axvline(2.5, color="gray", lw=0.8)
    ax.annotate("modo $5\\pi$ (2.5 ciclos)", xy=(2.5, ax.get_ylim()[1]),
                xytext=(3.0, ax.get_ylim()[1] * 0.5), fontsize=8)
    ax.set_xlabel("frecuencia en t [ciclos por unidad]")
    ax.set_ylabel("amplitud media")
    ax.set_xlim(0, 12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "spectral.png"), dpi=150)
    plt.close(fig)

    if spectral_report:
        print("\n--- razon de amplitud predicha/exacta por banda ---")
        for k, v in spectral_report.items():
            print(f"  {k:20s} baja={v['low_band_ratio_pred_over_exact']:.3f} "
                  f"alta={v['high_band_ratio_pred_over_exact']:.3f}")
        with open(os.path.join(args.out, "spectral_report.json"), "w") as fh:
            json.dump(spectral_report, fh, indent=2)

    print(f"\nSalidas en: {args.out}")


if __name__ == "__main__":
    main()
