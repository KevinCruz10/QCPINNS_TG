#!/usr/bin/env python3
"""
fig_prediccion.py — Predicción frente a solución exacta (cap. 6).

Lee los eval.npz que genera run_kg.py (misma convención que diagnose_kg.py:
fila = x fijo, columna = t) y grafica un corte espacial con media ± desviación
estándar entre semillas.

Uso (desde la raíz del repositorio):
    python fig_prediccion.py \
        --runs "QCPINN Angle–Cascade=results/kg_dv_angle_cascade_seed*" \
               "PINN Model-2=results/kg_classical_seed*" \
        --slice-x 0.5 --out figuras/
Salida: figuras/fig_prediccion_x050.pdf / .png
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_evals(pattern):
    out = []
    for base in glob.glob(pattern):
        out += glob.glob(os.path.join(base, "**", "eval.npz"), recursive=True)
    return sorted(set(out))


def load_group(paths):
    preds, exact = [], None
    for p in paths:
        z = np.load(p)
        n = int(round(np.sqrt(z["X"].shape[0])))
        preds.append(z["u_pred"].reshape(n, n))
        if exact is None:
            exact = z["u_star"].reshape(n, n)
    g = np.linspace(0.0, 1.0, n)
    return g, g, np.stack(preds), exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"Etiqueta=patrón_glob"')
    ap.add_argument("--slice-x", type=float, default=0.5)
    ap.add_argument("--out", default="figuras")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    colores = ["#1f6fb4", "#e07b00", "#2a9d3a", "#9b3fb5"]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.3, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    xs = args.slice_x
    exacta_dibujada = False

    for c, spec in zip(colores, args.runs):
        label, pattern = spec.split("=", 1)
        paths = find_evals(pattern)
        if not paths:
            raise SystemExit(f"Sin eval.npz para {pattern}")
        t, x, P, E = load_group(paths)
        i = int(round(xs * (len(x) - 1)))
        ex, S = E[i, :], P[:, i, :]
        mu, sd = S.mean(0), S.std(0, ddof=1) if len(S) > 1 else np.zeros_like(S[0])
        err = np.abs(S - ex)                       # error por semilla
        emu, esd = err.mean(0), err.std(0, ddof=1) if len(S) > 1 else np.zeros_like(ex)

        if not exacta_dibujada:
            a0.plot(t, ex, "k--", lw=1.3, label=r"Exacta $u^{*}$", zorder=5)
            exacta_dibujada = True
        a0.plot(t, mu, color=c, lw=1.2, label=f"{label} ($n={len(S)}$)")
        a0.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.2, lw=0)
        a1.plot(t, emu, color=c, lw=1.1)
        a1.fill_between(t, np.clip(emu - esd, 0, None), emu + esd, color=c, alpha=0.2, lw=0)
        print(f"{label}: x = {x[i]:.3f}, semillas = {len(S)}, "
              f"error máx. medio = {emu.max():.4f}")

    a0.set_ylabel(rf"$u(t,\,x={xs:g})$".replace(".", "{,}"))
    a0.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    a1.set_ylabel(r"$|u_\theta-u^{*}|$")
    a1.set_xlabel(r"$t$")
    a1.set_xlim(0, 1)
    fig.align_ylabels()
    fig.tight_layout()
    base = os.path.join(args.out, f"fig_prediccion_x{int(round(xs*100)):03d}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    print("Guardado en", base + ".pdf")


if __name__ == "__main__":
    main()
