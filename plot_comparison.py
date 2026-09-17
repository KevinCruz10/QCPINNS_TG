#!/usr/bin/env python3
"""
plot_comparison.py - Figura consolidada del Capitulo 6: precision frente a
costo computacional y frente a grados de libertad, para los cuatro metodos.

    python plot_comparison.py

Lee:
    results/fd_reference/convergence.csv
    results/fem_reference/convergence.csv
    analysis/summary_groups.csv        (generado por analyze_kg.py)

Escribe:
    analysis/comparison_error_vs_cost.png
    analysis/tabla_consolidada.csv

Los metodos clasicos aparecen como curvas parametrizadas por la resolucion de
malla; las redes, como puntos con barra de dispersion entre semillas.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "Diferencias finitas": dict(color="#1f77b4", marker="o"),
    "Elementos finitos P1": dict(color="#2ca02c", marker="s"),
    "PINN (Model-2)": dict(color="#ff7f0e", marker="D"),
    "QCPINN (angle-cascade)": dict(color="#d62728", marker="*"),
}


def read_convergence(path):
    if not os.path.exists(path):
        print(f"  aviso: no se encontro {path}")
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {
        "err": np.array([float(r["L2_u_global_pct"]) for r in rows]),
        "time": np.array([float(r["tiempo_s"]) for r in rows]),
        "dof": np.array([float(r["nx"]) * float(r["nt"]) for r in rows]),
        "label": np.array([r["nx"] for r in rows]),
    }


def read_summary(path):
    if not os.path.exists(path):
        print(f"  aviso: no se encontro {path}")
        return {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[r["group"]] = {
            "err": float(r["L2_u_global_pct_mean"]),
            "err_sd": float(r["L2_u_global_pct_std"]),
            "time": float(r["total_time_s_mean"]),
            "time_sd": float(r["total_time_s_std"]),
            "dof": float(r["n_params_mean"]),
            "n_runs": int(r["n_runs"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="results/fd_reference/convergence.csv")
    ap.add_argument("--fem", default="results/fem_reference/convergence.csv")
    ap.add_argument("--summary", default="analysis/summary_groups.csv")
    ap.add_argument("--qcpinn-group", default="qcpinn_cascade")
    ap.add_argument("--pinn-group", default="pinn_model2")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fd = read_convergence(args.fd)
    fem = read_convergence(args.fem)
    nets = read_summary(args.summary)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---------------- panel (a): error vs tiempo -----------------
    ax = axes[0]
    for data, name, off in ((fd, "Diferencias finitas", (-6, -12)),
                            (fem, "Elementos finitos P1", (6, 7))):
        if data is None:
            continue
        st = STYLE[name]
        ax.plot(data["time"], data["err"], "-", **st, label=name, ms=6, lw=1.4)
        for tt, ee, lb in zip(data["time"], data["err"], data["label"]):
            ax.annotate(f"n={lb}", (tt, ee), textcoords="offset points",
                        xytext=off, fontsize=7, color=st["color"])

    for key, name in ((args.pinn_group, "PINN (Model-2)"),
                      (args.qcpinn_group, "QCPINN (angle-cascade)")):
        if key not in nets:
            continue
        d, st = nets[key], STYLE[name]
        ax.errorbar(d["time"], d["err"], yerr=d["err_sd"], xerr=d["time_sd"],
                    fmt=st["marker"], color=st["color"], ms=10, capsize=3,
                    label=f"{name}, {d['n_runs']} semillas")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tiempo de computo [s]")
    ax.set_ylabel("error relativo $L_2$ en $u$ [%]")
    ax.set_title("(a) Precision frente a costo")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(top=40)

    # ---------------- panel (b): error vs grados de libertad -----------------
    ax = axes[1]
    for data, name in ((fd, "Diferencias finitas"), (fem, "Elementos finitos P1")):
        if data is None:
            continue
        st = STYLE[name]
        ax.plot(data["dof"], data["err"], "-", **st, label=name, ms=6, lw=1.4)

    for key, name in ((args.pinn_group, "PINN (Model-2)"),
                      (args.qcpinn_group, "QCPINN (angle-cascade)")):
        if key not in nets:
            continue
        d, st = nets[key], STYLE[name]
        ax.errorbar(d["dof"], d["err"], yerr=d["err_sd"],
                    fmt=st["marker"], color=st["color"], ms=10, capsize=3,
                    label=name)

    # pendiente de referencia de segundo orden: err ~ dof^{-1}
    if fd is not None:
        xs = np.array([fd["dof"].min(), fd["dof"].max()])
        ys = fd["err"][0] * (xs / fd["dof"][0]) ** -1.0
        ax.plot(xs, ys, "k:", lw=1.0, label="pendiente de 2do orden")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grados de libertad\n(incognitas de malla o parametros entrenables)")
    ax.set_ylabel("error relativo $L_2$ en $u$ [%]")
    ax.set_title("(b) Precision frente a grados de libertad")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_ylim(top=40)

    fig.tight_layout()
    dest = os.path.join(args.out, "comparison_error_vs_cost.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)

    # ---------------- tabla consolidada -----------------
    rows = []
    if fd is not None:
        i = int(np.argmin(fd["err"]))
        rows.append(["Diferencias finitas", f"n={fd['label'][i]}",
                     f"{fd['err'][i]:.4f}", "-", f"{fd['time'][i]:.3f}",
                     f"{fd['dof'][i]:.0f} incognitas"])
    if fem is not None:
        i = int(np.argmin(fem["err"]))
        rows.append(["Elementos finitos P1", f"n={fem['label'][i]}",
                     f"{fem['err'][i]:.4f}", "-", f"{fem['time'][i]:.3f}",
                     f"{fem['dof'][i]:.0f} incognitas"])
    for key, name in ((args.pinn_group, "PINN (Model-2)"),
                      (args.qcpinn_group, "QCPINN (angle-cascade)")):
        if key in nets:
            d = nets[key]
            rows.append([name, f"{d['n_runs']} semillas",
                         f"{d['err']:.3f} +- {d['err_sd']:.3f}", "-",
                         f"{d['time']:.1f}", f"{d['dof']:.0f} parametros"])

    with open(os.path.join(args.out, "tabla_consolidada.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metodo", "Configuracion", "L2_u (%)", "L2_f (%)",
                    "Tiempo (s)", "Grados de libertad"])
        w.writerows(rows)

    print("\n--- tabla consolidada ---")
    for r in rows:
        print(f"  {r[0]:24s} {r[1]:14s} {r[2]:>18s} {r[4]:>10s} s  {r[5]}")

    # razones utiles para la redaccion
    if fd is not None and args.qcpinn_group in nets:
        q = nets[args.qcpinn_group]
        i = int(np.argmin(fd["err"]))
        print(f"\n  QCPINN / FD  -> error x{q['err'] / fd['err'][i]:.0f}, "
              f"tiempo x{q['time'] / fd['time'][i]:.0f}")
    if args.pinn_group in nets and args.qcpinn_group in nets:
        p, q = nets[args.pinn_group], nets[args.qcpinn_group]
        print(f"  QCPINN / PINN -> error x{q['err'] / p['err']:.2f}, "
              f"tiempo x{q['time'] / p['time']:.1f}, "
              f"parametros {100 * q['dof'] / p['dof']:.1f} %")

    print(f"\nFigura: {dest}")


if __name__ == "__main__":
    main()
