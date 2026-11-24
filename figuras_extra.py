#!/usr/bin/env python3
"""
figuras_extra.py - Genera las figuras que faltan en el capitulo de resultados.

    python figuras_extra.py                    # todas
    python figuras_extra.py --solo convergencia campos

Produce en --out (por defecto analysis/):
    fig_convergencia_numerica.png   orden de convergencia de FD y FEM
    fig_campos_numericos.png        solucion exacta, solucion FD y mapa de error
    fig_espacio_diseno.png          barras de error por configuracion del circuito
    fig_pesos_cauchy.png            barrido de la ponderacion de u_t(0,x)
    fig_prueba_causal.png           brazos A y B por arquitectura

Lee de results/fd_reference/, results/fem_reference/ y analysis/.
Cada figura se omite con aviso si le faltan datos, sin abortar el resto.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, VERDE, NARANJA, ROJO, GRIS = "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"


def leer_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def leer_grupos(path):
    filas = leer_csv(path)
    if not filas:
        return {}
    return {f["group"]: f for f in filas}


def val(fila, clave, defecto=np.nan):
    try:
        return float(fila[clave])
    except (KeyError, TypeError, ValueError):
        return defecto


# --------------------------------------------------------------- 1. convergencia

def fig_convergencia(args):
    fd = leer_csv(os.path.join(args.results, "fd_reference", "convergence.csv"))
    fem = leer_csv(os.path.join(args.results, "fem_reference", "convergence.csv"))
    if not fd and not fem:
        print("  omitida convergencia: faltan los convergence.csv")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for datos, nombre, color, marca, off in (
            (fd, "Diferencias finitas", AZUL, "o", (-14, -16)),
            (fem, "Elementos finitos P1", VERDE, "s", (10, 8))):
        if not datos:
            continue
        dx = np.array([float(r["dx"]) for r in datos])
        err = np.array([float(r["L2_u_global_pct"]) for r in datos])
        ax.loglog(dx, err, marker=marca, color=color, label=nombre,
                  lw=1.5, ms=7, alpha=0.85)
        for d, e, r in zip(dx, err, datos):
            o = r.get("orden_observado", "")
            if o:
                ax.annotate(f"$p={o}$", (d, e), textcoords="offset points",
                            xytext=off, fontsize=8, color=color)

    # pendiente de referencia de segundo orden
    base = fd or fem
    dx = np.array([float(r["dx"]) for r in base])
    err = np.array([float(r["L2_u_global_pct"]) for r in base])
    xs = np.array([dx.min(), dx.max()])
    ax.loglog(xs, err[0] * (xs / dx[0]) ** 2, "k--", lw=1.0,
              label=r"pendiente $\mathcal{O}(\Delta x^{2})$")

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"error relativo $L_2$ en $u$ [\%]".replace("\\%", "%"))
    ax.set_title("Convergencia de los métodos numéricos clásicos")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.invert_xaxis()
    ax.margins(y=0.18)
    fig.tight_layout()
    dest = os.path.join(args.out, "fig_convergencia_numerica.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"  {dest}")


# ------------------------------------------------------------------ 2. campos

def fig_campos(args):
    p = os.path.join(args.results, "fd_reference", "eval_fd.npz")
    if not os.path.exists(p):
        print("  omitida campos: falta eval_fd.npz")
        return
    z = np.load(p)
    U, E = z["u_fd"], z["u_exact"]          # fila = x, columna = t
    err = np.abs(U - E)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for ax, campo, titulo, cmap in (
            (axes[0], E, "Solución exacta $u^{*}(t,x)$", "RdYlBu_r"),
            (axes[1], U, "Diferencias finitas $u_h(t,x)$", "RdYlBu_r"),
            (axes[2], err, "$|u_h - u^{*}|$", "Reds")):
        vmax = np.abs(campo).max()
        kw = dict(vmin=0, vmax=vmax) if cmap == "Reds" else dict(vmin=-vmax, vmax=vmax)
        im = ax.imshow(campo, origin="lower", extent=[0, 1, 0, 1],
                       aspect="auto", cmap=cmap, **kw)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_title(titulo, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    dest = os.path.join(args.out, "fig_campos_numericos.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"  {dest}")


# ----------------------------------------------------------- 3. espacio de diseño

NOMBRES = {
    "cascade": "Angle--Cascade",
    "cross_mesh": "Angle--Cross-mesh",
    "alternate": "Angle--Alternate",
    "layered": "Angle--Layered",
    "amp_cascade": "Amplitude--Cascade",
    "q6": "Angle--Cascade (6 qubits)",
}


def fig_espacio_diseno(args):
    g = leer_grupos(os.path.join(args.out, "summary_groups.csv"))
    if not g:
        print("  omitida espacio de diseño: falta summary_groups.csv")
        return
    orden = [k for k in ["cascade", "q6", "cross_mesh", "alternate",
                         "layered", "amp_cascade"] if k in g]
    if not orden:
        print("  omitida espacio de diseño: sin grupos reconocidos")
        return

    medias = [val(g[k], "L2_u_global_pct_mean") for k in orden]
    sds = [val(g[k], "L2_u_global_pct_std", 0.0) for k in orden]
    etiquetas = [f"{NOMBRES.get(k,k)}\n($n$={int(val(g[k],'n_runs',0))})" for k in orden]
    colores = [AZUL if m < 10 else NARANJA for m in medias]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(orden))
    ax.barh(y, medias, xerr=sds, color=colores, alpha=0.85,
            error_kw=dict(ecolor=GRIS, capsize=4, lw=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=8.5)
    ax.invert_yaxis()

    if "pinn" in g:
        ref = val(g["pinn"], "L2_u_global_pct_mean")
        ax.axvline(ref, color=ROJO, ls="--", lw=1.3,
                   label=f"PINN Model-2 ({ref:.2f}\\%)".replace("\\%", "%"))
        ax.legend(fontsize=8.5, loc="lower right")

    for yi, m, s in zip(y, medias, sds):
        ax.text(m + s + 0.4, yi, f"{m:.2f}", va="center", fontsize=8.5)

    ax.set_xlabel("error relativo $L_2$ en $u$ [%]")
    ax.set_title("Espacio de diseño del circuito cuántico")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    dest = os.path.join(args.out, "fig_espacio_diseno.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"  {dest}")


# ------------------------------------------------------------------ 4. pesos

def fig_pesos(args):
    g = leer_grupos(os.path.join(args.out, "summary_groups.csv"))
    claves = [("cascade", 0.1), ("wut1", 1.0), ("wut10", 10.0)]
    disponibles = [(k, w) for k, w in claves if k in g]
    if len(disponibles) < 2:
        print("  omitida pesos: faltan los grupos wut")
        return

    pesos = [w for _, w in disponibles]
    glob = [val(g[k], "L2_u_global_pct_mean") for k, _ in disponibles]
    glob_sd = [val(g[k], "L2_u_global_pct_std", 0.0) for k, _ in disponibles]
    t0 = [val(g[k], "L2_u_t0_pct_mean") for k, _ in disponibles]
    t0_sd = [val(g[k], "L2_u_t0_pct_std", 0.0) for k, _ in disponibles]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(pesos, glob, yerr=glob_sd, marker="o", color=AZUL,
                capsize=4, lw=1.5, label="error global")
    ax.errorbar(pesos, t0, yerr=t0_sd, marker="s", color=VERDE,
                capsize=4, lw=1.5, label="error en $t=0$")
    ax.set_xscale("log")
    ax.set_xticks(pesos)
    ax.set_xticklabels([f"{w:g}" for w in pesos])
    ax.set_xlabel(r"ponderación $\lambda_{u_t}$ de la condición de velocidad inicial")
    ax.set_ylabel("error relativo $L_2$ [%]")
    ax.set_title("Efecto de la ponderación de la segunda condición de Cauchy")
    ax.axvline(0.1, color=GRIS, ls=":", lw=1.0)
    ax.annotate("valor del\nprotocolo original", xy=(0.1, max(glob) * 0.95),
                fontsize=8, color=GRIS, ha="left")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    dest = os.path.join(args.out, "fig_pesos_cauchy.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"  {dest}")


# ------------------------------------------------------------- 5. prueba causal

def fig_causal(args):
    filas = leer_csv(os.path.join(args.out, "causal_runs.csv"))
    if not filas:
        print("  omitida prueba causal: falta causal_runs.csv")
        return

    datos = {}
    for f in filas:
        clave = (f.get("solver"), f.get("arm"))
        datos.setdefault(clave, []).append(float(f["L2_u_window_pct"]))

    arqs = [("Classical", "PINN Model-2"), ("DV", "QCPINN Cascade")]
    arqs = [(k, n) for k, n in arqs if (k, "A") in datos and (k, "B") in datos]
    if not arqs:
        print("  omitida prueba causal: sin brazos completos")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ancho = 0.34
    x = np.arange(len(arqs))
    for i, (brazo, etiqueta, color) in enumerate(
            [("A", r"Brazo A: $t\in[0,\,0{,}5]$", AZUL),
             ("B", r"Brazo B: $t\in[0{,}5,\,1]$", NARANJA)]):
        m = [np.mean(datos[(k, brazo)]) for k, _ in arqs]
        s = [np.std(datos[(k, brazo)], ddof=1) if len(datos[(k, brazo)]) > 1 else 0
             for k, _ in arqs]
        pos = x + (i - 0.5) * ancho
        ax.bar(pos, m, ancho, yerr=s, label=etiqueta, color=color, alpha=0.87,
               error_kw=dict(ecolor=GRIS, capsize=4, lw=1.1))
        for p, mi, si in zip(pos, m, s):
            ax.text(p, mi + si + 0.12, f"{mi:.2f}", ha="center", fontsize=8.5)

    for xi, (k, _) in enumerate(arqs):
        r = np.mean(datos[(k, "B")]) / np.mean(datos[(k, "A")])
        ax.text(xi, -0.55, f"razón B/A = {r:.2f}".replace(".", ","),
                ha="center", fontsize=9,
                color=(VERDE if r < 1 else ROJO))

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in arqs])
    ax.set_ylabel("error relativo $L_2$ en la ventana [%]")
    ax.set_title("Prueba causal: efecto de eliminar el error heredado")
    ax.set_ylim(bottom=-1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    dest = os.path.join(args.out, "fig_prueba_causal.png")
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"  {dest}")


# --------------------------------------------------------------------- main

FIGURAS = {
    "convergencia": fig_convergencia,
    "campos": fig_campos,
    "diseno": fig_espacio_diseno,
    "pesos": fig_pesos,
    "causal": fig_causal,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--solo", nargs="+", choices=sorted(FIGURAS), default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("Generando figuras:")
    for nombre in (a.solo or FIGURAS):
        FIGURAS[nombre](a)
    print("\nCopia a la tesis con:")
    print(f"  cp {a.out}/fig_*.png $TESIS/figuras/")


if __name__ == "__main__":
    main()
