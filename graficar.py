#!/usr/bin/env python3
"""
graficar.py - genera el maximo de figuras para la monografia a partir de los
CSV que ya produjo comparar_revision.py / analizar_perdida.py. No reentrena nada.

    python graficar.py --indir analysis_revision --results results \
                       --out figuras --fmt pdf

Salida: un PDF (o PNG con --fmt png) por figura en --out/. Cada figura se omite
sin error si le faltan sus datos. Colores fijos por modelo en todas las graficas
(paleta Okabe-Ito, segura para daltonismo e impresion).
"""
import argparse, csv, glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- estilo comun ----------
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
    "legend.frameon": False, "lines.linewidth": 2.0,
})
# color fijo por entidad (identidad, nunca por rango)
COL = {
    "pinn": "#0072B2", "qcpinn": "#D55E00", "ctrl_none": "#009E73",
    "ctrl_cos": "#CC79A7", "ctrl_tanh": "#E69F00", "ctrl_linear": "#56B4E9",
}
NOM = {
    "pinn": "PINN (Model-2)", "qcpinn": "QCPINN", "ctrl_none": "sin centro",
    "ctrl_cos": "cos(z)", "ctrl_tanh": "tanh", "ctrl_linear": "lineal 5x5",
}
GRIS_FD, GRIS_FEM = "#444444", "#888888"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_master(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d.setdefault(r["grupo"], []).append(r)
    return d


def col(rows, c):
    return [num(r.get(c)) for r in rows if num(r.get(c)) is not None]


def stats(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if n == 0:
        return (float("nan"), 0.0, 0)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd, n


def medrange(v):
    v = sorted(x for x in v if x is not None)
    n = len(v)
    if n == 0:
        return (float("nan"),) * 3
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return med, v[0], v[-1]


_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
        8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(dof):
    ks = sorted(_T95)
    for k in ks:
        if dof <= k:
            return _T95[k]
    return 1.96


def welch_ci(a, b):
    """IC95 de media(a)-media(b) por Welch. Devuelve (dif, lo, hi, p_aprox)."""
    ma, sa, na = stats(a)
    mb, sb, nb = stats(b)
    if na < 2 or nb < 2:
        return (ma - mb, float("nan"), float("nan"))
    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    if se == 0:
        return (ma - mb, ma - mb, ma - mb)
    num_ = (sa ** 2 / na + sb ** 2 / nb) ** 2
    den = (sa ** 2 / na) ** 2 / (na - 1) + (sb ** 2 / nb) ** 2 / (nb - 1)
    dof = num_ / den if den else 1
    from scipy import stats as _st
    t = _st.t.ppf(0.975, dof)
    d = ma - mb
    return d, d - t * se, d + t * se


def save(fig, out, name, fmt):
    p = os.path.join(out, f"{name}.{fmt}")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")
    return p


# ---------- figuras ----------
def fig_box_controles(M, out, fmt):
    orden = ["pinn", "ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear", "qcpinn"]
    datos = [(k, col(M.get(k, []), "L2_u_global_pct")) for k in orden]
    datos = [(k, v) for k, v in datos if v]
    if len(datos) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, v) in enumerate(datos):
        bp = ax.boxplot(v, positions=[i], widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor=COL[k], alpha=0.6))
        for box in bp["boxes"]:
            box.set(facecolor=COL[k], alpha=0.55, edgecolor=COL[k])
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.12, 0.12, len(v)),
                   v, s=14, color=COL[k], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels([NOM[k] for k, _ in datos], rotation=20, ha="right")
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Distribución del error por tipo de centro (misma forma de red)")
    save(fig, out, "01_box_controles", fmt)


def fig_strip_semillas(M, out, fmt):
    orden = ["pinn", "ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear", "qcpinn"]
    datos = [(k, col(M.get(k, []), "L2_u_global_pct")) for k in orden]
    datos = [(k, v) for k, v in datos if v]
    if len(datos) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, v) in enumerate(datos):
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)),
                   v, s=30, color=COL[k], edgecolor="white", linewidth=0.6, zorder=3)
        med, _, _ = medrange(v)
        ax.plot([i - 0.25, i + 0.25], [med, med], color="black", lw=1.8, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels([NOM[k] for k, _ in datos], rotation=20, ha="right")
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Error por semilla (barra negra = mediana)")
    save(fig, out, "02_strip_semillas", fmt)


def fig_regiones(M, out, fmt):
    orden = ["pinn", "ctrl_none", "ctrl_cos", "qcpinn"]
    orden = [k for k in orden if M.get(k)]
    if len(orden) < 2:
        return
    regs = [("L2_u_global_pct", "global"), ("L2_u_interior_pct", "interior"),
            ("L2_u_boundary_pct", "frontera")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.25
    x = np.arange(len(orden))
    tono = {"global": 1.0, "interior": 0.7, "frontera": 0.4}
    for j, (c, nom) in enumerate(regs):
        vals = [stats(col(M[k], c))[0] for k in orden]
        ax.bar(x + (j - 1) * w, vals, w, label=nom,
               color=[COL[k] for k in orden], alpha=tono[nom], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([NOM[k] for k in orden], rotation=15, ha="right")
    ax.set_ylabel("error L2 (%)")
    ax.set_title("Error por región (global / interior / frontera)")
    # leyenda de tono (no de color, que es identidad del modelo)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=tono[n], label=n)
                       for _, n in regs], loc="upper left")
    save(fig, out, "03_regiones", fmt)


def fig_params(M, out, fmt):
    orden = ["pinn", "ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear", "qcpinn"]
    pts = []
    for k in orden:
        if not M.get(k):
            continue
        np_, _, _ = stats(col(M[k], "n_params"))
        mu, sd, n = stats(col(M[k], "L2_u_global_pct"))
        if np_ == np_ and mu == mu:
            pts.append((k, np_, mu, sd))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, npar, mu, sd in pts:
        lo_ = min(sd, 0.6 * mu)
        ax.errorbar(npar, mu, yerr=[[lo_], [sd]], fmt="o", ms=10, color=COL[k],
                    ecolor=COL[k], elinewidth=1.5, capsize=3, zorder=3)
        off = {"qcpinn": (14, 10), "ctrl_none": (14, -2), "ctrl_cos": (14, -14),
               "pinn": (-30, 12)}.get(k, (14, 6))
        ax.annotate(NOM[k], (npar, mu), textcoords="offset points",
                    xytext=off, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("número de parámetros entrenables")
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Error frente a tamaño de la red (el ahorro es de la forma)")
    save(fig, out, "04_error_vs_parametros", fmt)


def fig_masa(M, out, fmt):
    pares = [("pinn", "beta1_pinn", "PINN"), ("qcpinn", "beta1_qcpinn", "QCPINN"),
             ("ctrl_cos", "beta1_cos", "cos(z)"), ("ctrl_none", "beta1_none", "sin centro")]
    pares = [(a, b, nom) for a, b, nom in pares if M.get(a) and M.get(b)]
    if not pares:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(pares))
    w = 0.38
    for j, (clave, etq) in enumerate([(0, "sin masa (β=0)"), (1, "con masa (β=1)")]):
        vals, errs, cols = [], [], []
        for a, b, nom in pares:
            k = a if clave == 0 else b
            base = a
            mu, sd, _ = stats(col(M[k], "L2_u_global_pct"))
            vals.append(mu); errs.append(sd); cols.append(COL[base])
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=cols, alpha=0.5 if clave == 0 else 0.95, edgecolor="white",
               label=etq, hatch="//" if clave == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([nom for _, _, nom in pares])
    ax.set_ylabel("error L2 global de u (%)")
    ax.set_title("Efecto del término de masa (β=0 vs β=1)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.5, hatch="//", label="beta=0"),
                       Patch(facecolor="gray", alpha=0.95, label="beta=1")])
    save(fig, out, "05_masa_beta", fmt)


def fig_lbfgs(M, out, fmt):
    grupos = [("lbfgs", "clásico 2000 pts."), ("lbfgs_cls800", "clásico 800 pts."),
              ("lbfgs_dv", "híbrido 800 pts.")]
    grupos = [(k, nom) for k, nom in grupos if M.get(k)]
    if not grupos:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(grupos))
    w = 0.38
    for j, (c, etq, hatch, al) in enumerate(
            [("L2_u_global_pct_adam", "Adam", "//", 0.5),
             ("L2_u_global_pct", "Adam + L-BFGS", None, 0.95)]):
        mus = [stats(col(M[k], c))[0] for k, _ in grupos]
        sds = [stats(col(M[k], c))[1] for k, _ in grupos]
        base = ["pinn", "pinn", "qcpinn"]
        ax.bar(x + (j - 0.5) * w, mus, w, yerr=sds, capsize=3,
               color=[COL[b] for b in base[:len(grupos)]], alpha=al,
               hatch=hatch, edgecolor="white")
    # puntos por semilla del resultado final (muestra outliers como seed 8)
    for i, (k, _) in enumerate(grupos):
        v = col(M[k], "L2_u_global_pct")
        ax.scatter(np.full(len(v), i + 0.5 * w) + np.random.uniform(-0.05, 0.05, len(v)),
                   v, s=16, color="black", alpha=0.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([nom for _, nom in grupos])
    ax.set_ylabel("error L2 global de u (%)")
    ax.set_title("Efecto de L-BFGS tras Adam (puntos = semillas)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.5, hatch="//", label="Adam"),
                       Patch(facecolor="gray", alpha=0.95, label="Adam + L-BFGS")])
    save(fig, out, "06_lbfgs_antes_despues", fmt)


def fig_energia(M, R, out, fmt):
    def refjson(name):
        p = os.path.join(R, name, "metrics.json")
        if os.path.exists(p):
            return num(json.load(open(p)).get("drift_rel_ref"))
        return None
    barras = []  # (etiqueta, media, sd, color)
    fd, fem = refjson("energia_fd_beta1") or 2.3e-6, refjson("energia_fem_beta1") or 4.5e-6
    if fd:
        barras.append(("FD", fd, 0, GRIS_FD))
    if fem:
        barras.append(("FEM", fem, 0, GRIS_FEM))
    for k, etq, c in [("energia_cls_lbfgs", "PINN\nAdam", COL["pinn"]),
                      ("energia_cls_lbfgs", "PINN\nL-BFGS", COL["pinn"]),
                      ("energia_dv_lbfgs", "QCPINN\nAdam", COL["qcpinn"]),
                      ("energia_dv_lbfgs", "QCPINN\nL-BFGS", COL["qcpinn"])]:
        campo = "drift_rel_adam" if "Adam" in etq else "drift_rel"
        if M.get(k):
            mu, sd, _ = stats(col(M[k], campo))
            if mu == mu:
                barras.append((etq, mu, sd, c))
    if len(barras) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(barras))
    for i, (etq, mu, sd, c) in enumerate(barras):
        al = 0.55 if "Adam" in etq else (1.0 if etq in ("FD", "FEM") else 0.95)
        hatch = "//" if "Adam" in etq else None
        ax.bar(i, mu, 0.7, yerr=sd if sd else None, capsize=3, color=c,
               alpha=al, hatch=hatch, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([e for e, *_ in barras])
    ax.set_ylabel("deriva relativa de E(t)  [escala log]")
    ax.set_title("Conservación de energía: redes vs métodos numéricos")
    save(fig, out, "07_energia_deriva", fmt)


def fig_pesos(M, out, fmt):
    orden = [("wres0.1", 0.1), ("wres1", 1), ("wres10", 10), ("wres100", 100)]
    orden = [(k, w) for k, w in orden if M.get(k)]
    if len(orden) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for c, nom, mark in [("L2_u_global_pct", "global", "o"),
                         ("L2_u_interior_pct", "interior", "s"),
                         ("L2_u_boundary_pct", "frontera", "^")]:
        ws = [w for _, w in orden]
        ys = [stats(col(M[k], c))[0] for k, _ in orden]
        ax.plot(ws, ys, marker=mark, label=nom)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("peso del residuo  λ_f")
    ax.set_ylabel("error L2 (%)  [escala log]")
    ax.set_title("Barrido del peso del residuo: todo empeora al subirlo")
    ax.legend()
    save(fig, out, "08_barrido_pesos", fmt)


def fig_forest(M, out, fmt):
    contr = []
    def add(nom, a, b, ca="L2_u_global_pct", cb="L2_u_global_pct"):
        if M.get(a) and M.get(b):
            d, lo, hi = welch_ci(col(M[a], ca), col(M[b], cb))
            if lo == lo:
                contr.append((nom, d, lo, hi))
    add("QCPINN - PINN  (β=0)", "qcpinn", "pinn")
    add("QCPINN - PINN  (β=1)", "beta1_qcpinn", "beta1_pinn")
    add("QCPINN - cos(z)  (β=1)", "beta1_qcpinn", "beta1_cos")
    add("Adam 50k - L-BFGS(800)  clásico", "adamext", "lbfgs_cls800")
    add("híbrido - clásico  (L-BFGS 800)", "lbfgs_dv", "lbfgs_cls800")
    if not contr:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(contr) + 1.5))
    for i, (nom, d, lo, hi) in enumerate(contr):
        y = len(contr) - 1 - i
        c = "#b00020" if (lo > 0 or hi < 0) else "#444444"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=2)
        ax.scatter([d], [y], s=55, color=c, zorder=3)
        ax.annotate(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", (d, y),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axvline(0, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.set_yticks(range(len(contr)))
    ax.set_yticklabels([nom for nom, *_ in reversed(contr)])
    ax.set_xlabel("diferencia de error L2 global (pp)   IC 95%")
    ax.set_title("Contrastes clave (rojo = IC excluye 0)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "09_forest_contrastes", fmt)


def fig_trayectoria(indir, M, out, fmt):
    p = os.path.join(indir, "traj_l2_adamext.csv")
    if not os.path.exists(p):
        return
    ep, mu, sd = [], [], []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            ep.append(num(r["epoch"])); mu.append(num(r["L2_u_pct_media"]))
            sd.append(num(r.get("L2_u_pct_sd")) or 0.0)
    if len(ep) < 2:
        return
    ep, mu, sd = np.array(ep), np.array(mu), np.array(sd)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.fill_between(ep, mu - sd, mu + sd, color=COL["pinn"], alpha=0.18)
    ax.plot(ep, mu, color=COL["pinn"], label="Adam (PINN), media +- sd")
    # lineas L-BFGS
    for grp, etq, c in [("lbfgs", "L-BFGS 2000 pts.", "#b00020"),
                        ("lbfgs_cls800", "L-BFGS 800 pts.", "#D55E00")]:
        if M.get(grp):
            v = stats(col(M[grp], "L2_u_global_pct"))[0]
            if v == v:
                ax.axhline(v, color=c, ls="--", lw=1.6, label=f"{etq} = {v:.2f}%")
    ax.set_yscale("log")
    ax.set_xlabel("épocas de Adam")
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Adam no se estabiliza; L-BFGS llega mas bajo con ~20x menos pasos")
    ax.legend()
    save(fig, out, "10_trayectoria_adam_lbfgs", fmt)


def fig_curvas_perdida(indir, out, fmt):
    archivos = sorted(glob.glob(os.path.join(indir, "curva_perdida_*.csv")))
    if not archivos:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    cmap = {"pinn": COL["pinn"], "qcpinn": COL["qcpinn"]}
    algo = False
    for f in archivos:
        etq = os.path.basename(f)[len("curva_perdida_"):-4]
        ep, lo = [], []
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ep.append(num(r["epoch"])); lo.append(num(r["loss_media"]))
        if len(ep) < 2:
            continue
        ax.plot(ep, lo, color=cmap.get(etq, "#666666"), label=NOM.get(etq, etq))
        algo = True
    if not algo:
        plt.close(fig); return
    ax.set_yscale("log")
    ax.set_xlabel("épocas"); ax.set_ylabel("pérdida media  [escala log]")
    ax.set_title("Curvas de pérdida (aún descendiendo al final del entrenamiento)")
    ax.legend()
    save(fig, out, "11_curvas_perdida", fmt)


def fig_box_lbfgs(M, out, fmt):
    orden = [("lbfgs_cls800", "clásico\n800 pts.", "pinn"),
             ("lbfgs_dv", "híbrido\n800 pts.", "qcpinn")]
    datos = [(k, nom, b, col(M.get(k, []), "L2_u_global_pct")) for k, nom, b in orden]
    datos = [t for t in datos if t[3]]
    if len(datos) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (k, nom, b, v) in enumerate(datos):
        bp = ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=5,
                                        markerfacecolor=COL[b], alpha=0.7))
        for box in bp["boxes"]:
            box.set(facecolor=COL[b], alpha=0.55, edgecolor=COL[b])
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.1, 0.1, len(v)),
                   v, s=22, color=COL[b], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels([nom for _, nom, _, _ in datos])
    ax.set_ylabel("error L2 global de u (%) tras L-BFGS")
    ax.set_title("Distribución tras L-BFGS (outliers visibles)")
    save(fig, out, "12_box_lbfgs", fmt)


def fig_cuatro_metodos(M, R, out, fmt):
    """Exactitud de los cuatro métodos de solución (β=1)."""
    barras = []
    for name, etq, c in [("fd_reference_beta1", "FD", GRIS_FD),
                         ("fem_reference_beta1", "FEM", GRIS_FEM)]:
        p = os.path.join(R, name, "metrics.json")
        if os.path.exists(p):
            v = num(json.load(open(p)).get("L2_u_global_pct"))
            if v:
                barras.append((etq, v, 0, c))
    for k, etq, c in [("beta1_pinn", "PINN", COL["pinn"]),
                      ("beta1_qcpinn", "QCPINN", COL["qcpinn"])]:
        if M.get(k):
            mu, sd, _ = stats(col(M[k], "L2_u_global_pct"))
            if mu == mu:
                barras.append((etq, mu, sd, c))
    if len(barras) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (etq, mu, sd, c) in enumerate(barras):
        ax.bar(i, mu, 0.7, yerr=sd if sd else None, capsize=3, color=c,
               edgecolor="white")
        ax.annotate(f"{mu:.3g}", (i, mu), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(barras)))
    ax.set_xticklabels([e for e, *_ in barras])
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Exactitud de los cuatro métodos de solución (β=1)")
    save(fig, out, "13_cuatro_metodos", fmt)


def fig_slope_adam_lbfgs(M, out, fmt):
    """Adam → L-BFGS, una línea por semilla, clásico vs QCPINN."""
    bloques = [("lbfgs_cls800", "clasico", "pinn", 0.0),
               ("lbfgs_dv", "QCPINN", "qcpinn", 2.5)]
    bloques = [b for b in bloques if M.get(b[0])]
    if not bloques:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, nom, base, x0 in bloques:
        c = COL[base]
        rows = M[k]
        for r in rows:
            a, d = num(r.get("L2_u_global_pct_adam")), num(r.get("L2_u_global_pct"))
            if a is None or d is None:
                continue
            ax.plot([x0, x0 + 1], [a, d], color=c, alpha=0.4, lw=1.2, zorder=2)
            ax.scatter([x0, x0 + 1], [a, d], s=18, color=c, zorder=3,
                       edgecolor="white", linewidth=0.4)
        ma = stats(col(rows, "L2_u_global_pct_adam"))[0]
        md = stats(col(rows, "L2_u_global_pct"))[0]
        ax.plot([x0, x0 + 1], [ma, md], color=c, lw=3.2, zorder=4)
        ax.annotate(nom, (x0 + 0.5, max(ma, md)), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=11, color=c, weight="bold")
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["Adam", "L-BFGS", "Adam", "L-BFGS"])
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Adam → L-BFGS por semilla (linea gruesa = media)")
    save(fig, out, "14_slope_adam_lbfgs", fmt)


def fig_factor_lbfgs(M, out, fmt):
    """Cuantas veces reduce el error L-BFGS, por configuracion."""
    grupos = [("lbfgs", "clásico\n2000 pts.", "pinn"),
              ("lbfgs_cls800", "clásico\n800 pts.", "pinn"),
              ("lbfgs_dv", "QCPINN\n800 pts.", "qcpinn")]
    grupos = [g for g in grupos if M.get(g[0])]
    if not grupos:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (k, nom, base) in enumerate(grupos):
        ma = stats(col(M[k], "L2_u_global_pct_adam"))[0]
        md = stats(col(M[k], "L2_u_global_pct"))[0]
        if not (ma == ma and md == md and md > 0):
            continue
        f = ma / md
        ax.bar(i, f, 0.6, color=COL[base], edgecolor="white")
        ax.annotate(f"{f:.1f}x", (i, f), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
    ax.axhline(1, color="#999999", ls="--", lw=1)
    ax.set_xticks(range(len(grupos)))
    ax.set_xticklabels([nom for _, nom, _ in grupos])
    ax.set_ylabel("factor de reducción del error  (Adam / L-BFGS)")
    ax.set_title("Cuánto baja el error L-BFGS, por arquitectura")
    save(fig, out, "15_factor_reduccion_lbfgs", fmt)


def fig_qcpinn_regimenes(M, out, fmt):
    """QCPINN en tres regímenes, mismo eje (L2 %)."""
    casos = [("qcpinn", "L2_u_global_pct", "β=0\nAdam"),
             ("lbfgs_dv", "L2_u_global_pct", "β=0\nL-BFGS"),
             ("beta1_qcpinn", "L2_u_global_pct", "β=1\nAdam")]
    casos = [(k, c, nom) for k, c, nom in casos if M.get(k)]
    if len(casos) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (k, c, nom) in enumerate(casos):
        v = col(M[k], c)
        mu, sd, _ = stats(v)
        ax.bar(i, mu, 0.6, yerr=sd, capsize=3, color=COL["qcpinn"],
               alpha=0.85, edgecolor="white")
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.12, 0.12, len(v)),
                   v, s=16, color="black", alpha=0.5, zorder=3)
    ax.set_xticks(range(len(casos)))
    ax.set_xticklabels([nom for _, _, nom in casos])
    ax.set_ylabel("error L2 global de u (%)")
    ax.set_title("QCPINN: efecto del optimizador y de la masa")
    save(fig, out, "16_qcpinn_regimenes", fmt)


# patrones de carpeta por grupo (para encontrar eval.npz)
PATRON = {
    "qcpinn": "kg_dv_angle_cascade_seed*", "pinn": "kg_classical_seed*",
    "beta1_qcpinn": "kg2_dv_beta1_seed*", "beta1_pinn": "kg2_cls_beta1_seed*",
    "lbfgs_dv": "kg2_dv_lbfgs_seed*", "lbfgs_cls800": "kg2_cls_lbfgs800_seed*",
    "en_fd": "energia_fd_beta1", "en_fem": "energia_fem_beta1",
    "en_cls": "energia_cls_lbfgs_seed*", "en_dv": "energia_dv_lbfgs_seed*",
    "beta1_lbfgs_dv": "kg2_dv_beta1_lbfgs_seed*",
    "beta1_lbfgs_cls": "kg2_cls_beta1_lbfgs_seed*",
}


def find_median_run(R, grupo):
    """Carpeta de la corrida con L2 mediano del grupo. Devuelve (run_dir, seed)."""
    pat = PATRON.get(grupo)
    if not pat:
        return None
    cands = []
    for run in glob.glob(os.path.join(R, pat, "*")):
        mp = os.path.join(run, "metrics.json")
        if os.path.exists(mp):
            try:
                j = json.load(open(mp))
                l2 = num(j.get("L2_u_global_pct")); sd = j.get("seed")
            except Exception:
                continue
            if l2 is not None:
                cands.append((l2, run, sd))
    if not cands:
        return None
    cands.sort()
    _, run, sd = cands[len(cands) // 2]
    return run, sd


def load_field(R, grupo):
    """Campo de la corrida con L2 mediano. dict con t,x,us,up,err,ferr,sd o None."""
    mr = find_median_run(R, grupo)
    if not mr:
        return None
    run, sd = mr
    ep = os.path.join(run, "eval.npz")
    if not os.path.exists(ep):
        return None
    d = np.load(ep)
    X, us, up = d["X"], d["u_star"].ravel(), d["u_pred"].ravel()
    n = int(round(math.sqrt(X.shape[0])))
    if n * n != X.shape[0]:
        return None
    fs = d["f_star"].ravel().reshape(n, n); fp = d["f_pred"].ravel().reshape(n, n)
    t = X[:, 0].reshape(n, n); x = X[:, 1].reshape(n, n)
    US = us.reshape(n, n); UP = up.reshape(n, n)
    return {"t": t, "x": x, "us": US, "up": UP, "err": np.abs(UP - US),
            "ferr": np.abs(fp - fs), "sd": sd}


def _campo(ax, t, x, Z, cmap, vmin=None, vmax=None, diverg=False):
    ext = [t.min(), t.max(), x.min(), x.max()]
    if diverg:
        m = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)))
        vmin, vmax = -m, m
    im = ax.imshow(Z, extent=ext, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.grid(False)
    return im


def fig_campo_error_qcpinn_masa(R, out, fmt):
    f = load_field(R, "beta1_qcpinn")
    if not f:
        return
    t, x, ERR, sd = f["t"], f["x"], f["err"], f["sd"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = _campo(ax, t, x, ERR, "viridis")
    fig.colorbar(im, ax=ax, label="|u_pred - u_exacta|")
    ax.set_title(f"Campo de error de QCPINN con masa (beta=1, semilla {sd})")
    save(fig, out, "17_campo_error_qcpinn_masa", fmt)


def fig_campos_pinn_vs_qcpinn(R, out, fmt):
    fp, fq = load_field(R, "beta1_pinn"), load_field(R, "beta1_qcpinn")
    if not (fp and fq):
        return
    vmax = max(np.nanmax(fp["err"]), np.nanmax(fq["err"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, nom in [(axs[0], fp, "PINN"), (axs[1], fq, "QCPINN")]:
        im = _campo(ax, f["t"], f["x"], f["err"], "viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{nom} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|u_pred - u_exacta|", shrink=0.85)
    fig.suptitle("Campo de error con masa (β=1), misma escala", y=1.02)
    save(fig, out, "18_campos_pinn_vs_qcpinn_masa", fmt)


def fig_campos_adam_vs_lbfgs(R, out, fmt):
    fa, fl = load_field(R, "qcpinn"), load_field(R, "lbfgs_dv")
    if not (fa and fl):
        return
    vmax = max(np.nanmax(fa["err"]), np.nanmax(fl["err"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, nom in [(axs[0], fa, "QCPINN - Adam"), (axs[1], fl, "QCPINN - L-BFGS")]:
        im = _campo(ax, f["t"], f["x"], f["err"], "viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{nom} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|u_pred - u_exacta|", shrink=0.85)
    fig.suptitle("Campo de error QCPINN: Adam vs L-BFGS (misma escala)", y=1.02)
    save(fig, out, "19_campos_qcpinn_adam_vs_lbfgs", fmt)


def fig_error_vs_t(R, out, fmt):
    fp, fq = load_field(R, "beta1_pinn"), load_field(R, "beta1_qcpinn")
    series = [(fp, "PINN", COL["pinn"]), (fq, "QCPINN", COL["qcpinn"])]
    series = [s for s in series if s[0]]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for f, nom, c in series:
        et = f["err"].mean(axis=0)   # error medio sobre x en cada t
        ax.plot(f["t"][0, :], et, color=c, label=nom)
    ax.set_xlabel("t"); ax.set_ylabel("error medio sobre x,  |u_pred - u_exacta|")
    ax.set_title("Crecimiento del error en el tiempo (β=1)")
    ax.legend()
    save(fig, out, "20_error_vs_t_masa", fmt)


def fig_perfiles(R, out, fmt):
    f = load_field(R, "beta1_qcpinn")
    if not f:
        return
    t, x, US, UP, sd = f["t"], f["x"], f["us"], f["up"], f["sd"]
    tv, xv = t[0, :], x[:, 0]
    cortes = [0.25, 0.5, 0.75]
    fig, axs = plt.subplots(1, len(cortes), figsize=(11, 3.6), sharey=True)
    for ax, tc in zip(axs, cortes):
        j = int(np.argmin(np.abs(tv - tc)))
        ax.plot(xv, US[:, j], color="black", lw=2, label="exacta")
        ax.plot(xv, UP[:, j], color=COL["qcpinn"], lw=2, ls="--", label="QCPINN")
        ax.set_title(f"t = {tv[j]:.2f}"); ax.set_xlabel("x")
    axs[0].set_ylabel("u(t, x)"); axs[0].legend()
    fig.tight_layout()
    fig.suptitle(f"QCPINN con masa (beta=1): perfiles predicho vs exacto (semilla {sd})", y=1.04)
    save(fig, out, "21_perfiles_qcpinn_masa", fmt)



def load_metrics_curva(R, grupo):
    """metrics.csv (componentes de perdida vs epoca) de la corrida con L2 mediano."""
    mr = find_median_run(R, grupo)
    if not mr:
        return None
    p = os.path.join(mr[0], "metrics.csv")
    if not os.path.exists(p):
        return None
    cols = {}
    with open(p) as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            for k, v in r.items():
                cols.setdefault(k, []).append(num(v))
    if "epoch" not in cols:
        return None
    return {k: np.array([x for x in v], dtype=float) for k, v in cols.items()}, mr[1]


def load_energia(R, grupo):
    """energia.csv (t,E) de la corrida con deriva mediana. Devuelve (t, E) o None."""
    pat = PATRON.get(grupo)
    if not pat:
        return None
    files = glob.glob(os.path.join(R, pat, "**", "energia.csv"), recursive=True)
    files += glob.glob(os.path.join(R, pat, "energia.csv"))
    curvas = []
    for fp in set(files):
        t, E = [], []
        with open(fp) as fh:
            for r in csv.DictReader(fh):
                tv, ev = num(r.get("t")), num(r.get("E"))
                if tv is not None and ev is not None:
                    t.append(tv); E.append(ev)
        if len(t) >= 2:
            t, E = np.array(t), np.array(E)
            dr = np.max(np.abs(E - E[0])) / abs(E[0]) if E[0] else 0.0
            curvas.append((dr, t, E))
    if not curvas:
        return None
    curvas.sort(key=lambda c: c[0])
    _, t, E = curvas[len(curvas) // 2]
    return t, E


# ---- A. residuo de la PDE ----
def fig_residuo_pinn_vs_qcpinn(R, out, fmt):
    fp, fq = load_field(R, "beta1_pinn"), load_field(R, "beta1_qcpinn")
    if not (fp and fq):
        return
    vmax = max(np.nanmax(fp["ferr"]), np.nanmax(fq["ferr"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, nom in [(axs[0], fp, "PINN"), (axs[1], fq, "QCPINN")]:
        im = _campo(ax, f["t"], f["x"], f["ferr"], "magma", vmin=0, vmax=vmax)
        ax.set_title(f"{nom} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|residuo f|", shrink=0.85)
    fig.suptitle("Residuo de la PDE con masa (β=1): qué tan bien se cumple la ecuación", y=1.02)
    save(fig, out, "22_residuo_pinn_vs_qcpinn", fmt)


def fig_l2_residuo(M, out, fmt):
    orden = ["pinn", "ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear", "qcpinn"]
    datos = [(k, col(M.get(k, []), "L2_f_global_pct")) for k in orden]
    datos = [(k, v) for k, v in datos if v]
    if len(datos) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, v) in enumerate(datos):
        mu, sd, _ = stats(v)
        ax.bar(i, mu, 0.6, yerr=sd, capsize=3, color=COL[k], alpha=0.85, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels([NOM[k] for k, _ in datos], rotation=20, ha="right")
    ax.set_ylabel("L2 del residuo f (%)  [escala log]")
    ax.set_title("Error de la ecuación (residuo) por modelo")
    save(fig, out, "23_l2_residuo", fmt)


def fig_residuo_adam_vs_lbfgs(R, out, fmt):
    fa, fl = load_field(R, "qcpinn"), load_field(R, "lbfgs_dv")
    if not (fa and fl):
        return
    vmax = max(np.nanmax(fa["ferr"]), np.nanmax(fl["ferr"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, nom in [(axs[0], fa, "QCPINN - Adam"), (axs[1], fl, "QCPINN - L-BFGS")]:
        im = _campo(ax, f["t"], f["x"], f["ferr"], "magma", vmin=0, vmax=vmax)
        ax.set_title(f"{nom} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|residuo f|", shrink=0.85)
    fig.suptitle("Residuo de la PDE: Adam vs L-BFGS (misma escala)", y=1.02)
    save(fig, out, "24_residuo_adam_vs_lbfgs", fmt)


# ---- B. convergencia y restricciones ----
def fig_componentes_perdida(R, out, fmt):
    r = load_metrics_curva(R, "qcpinn")
    if not r:
        return
    c, sd = r
    ep = c["epoch"]
    comps = []
    if "loss_res" in c:
        comps.append(("residuo", c["loss_res"]))
    if "loss_bc1" in c and "loss_bc2" in c:
        comps.append(("fronteras", c["loss_bc1"] + c["loss_bc2"]))
    if "loss_ics" in c:
        comps.append(("cond. inicial", c["loss_ics"]))
    if "loss_ut" in c:
        comps.append(("u_t(0)", c["loss_ut"]))
    if not comps:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    paleta = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (nom, y), col_ in zip(comps, paleta):
        ax.plot(ep, np.maximum(y, 1e-12), label=nom, color=col_)
    ax.set_yscale("log")
    ax.set_xlabel("épocas"); ax.set_ylabel("pérdida por componente  [log]")
    ax.set_title(f"Descomposicion de la perdida (QCPINN, semilla {sd})")
    ax.legend()
    save(fig, out, "25_componentes_perdida", fmt)


def fig_error_restricciones(M, out, fmt):
    orden = ["pinn", "ctrl_cos", "qcpinn"]
    orden = [k for k in orden if M.get(k)]
    if len(orden) < 2:
        return
    regs = [("L2_u_t0_pct", "cond. inicial (t=0)"), ("L2_u_boundary_pct", "fronteras"),
            ("L2_u_interior_pct", "interior")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.25; x = np.arange(len(orden)); tono = [1.0, 0.65, 0.35]
    from matplotlib.patches import Patch
    for j, (cc, nom) in enumerate(regs):
        vals = [stats(col(M[k], cc))[0] for k in orden]
        ax.bar(x + (j - 1) * w, vals, w, color=[COL[k] for k in orden], alpha=tono[j],
               edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([NOM[k] for k in orden])
    ax.set_ylabel("error L2 (%)  [log]")
    ax.set_title("Cumplimiento de las restricciones impuestas blandamente")
    ax.legend(handles=[Patch(facecolor="gray", alpha=tono[j], label=n) for j, (_, n) in enumerate(regs)])
    save(fig, out, "26_error_restricciones", fmt)


# ---- C. efecto de la masa en el campo ----
def _campo_b0_b1(R, g0, g1, nom, archivo, out, fmt):
    f0, f1 = load_field(R, g0), load_field(R, g1)
    if not (f0 and f1):
        return
    vmax = max(np.nanmax(f0["err"]), np.nanmax(f1["err"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, et in [(axs[0], f0, "beta=0"), (axs[1], f1, "beta=1")]:
        im = _campo(ax, f["t"], f["x"], f["err"], "viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{et} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|u_pred - u_exacta|", shrink=0.85)
    fig.suptitle(f"Campo de error de {nom}: sin masa vs con masa (misma escala)", y=1.02)
    save(fig, out, archivo, fmt)


def fig_campo_masa_qcpinn(R, out, fmt):
    _campo_b0_b1(R, "qcpinn", "beta1_qcpinn", "QCPINN", "27_campo_masa_qcpinn", out, fmt)


def fig_campo_masa_pinn(R, out, fmt):
    _campo_b0_b1(R, "pinn", "beta1_pinn", "PINN", "28_campo_masa_pinn", out, fmt)


# ---- D. distribucion del error puntual ----
def fig_hist_error(R, out, fmt):
    series = [("pinn", "PINN"), ("qcpinn", "QCPINN"), ("lbfgs_dv", "QCPINN L-BFGS")]
    series = [(load_field(R, g), nom, g) for g, nom in series]
    series = [(f, nom, g) for f, nom, g in series if f]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    cmap = {"pinn": COL["pinn"], "qcpinn": COL["qcpinn"], "lbfgs_dv": "#8c3b00"}
    for f, nom, g in series:
        e = f["err"].ravel()
        e = e[e > 0]
        ax.hist(e, bins=np.logspace(np.log10(max(e.min(), 1e-6)), np.log10(e.max()), 40),
                histtype="step", lw=2, color=cmap.get(g, "#666"), label=nom)
    ax.set_xscale("log")
    ax.set_xlabel("error puntual |u_pred - u_exacta|  [log]")
    ax.set_ylabel("número de puntos de la malla")
    ax.set_title("Distribución del error puntual en el dominio")
    ax.legend()
    save(fig, out, "29_hist_error_puntual", fmt)


# ---- E. energia (visual) ----
def fig_energia_curvas(R, out, fmt):
    series = [("en_fd", "FD", GRIS_FD), ("en_fem", "FEM", GRIS_FEM),
              ("en_cls", "PINN", COL["pinn"]), ("en_dv", "QCPINN", COL["qcpinn"])]
    got = []
    for g, nom, c in series:
        r = load_energia(R, g)
        if r:
            got.append((r, nom, c))
    if not got:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for (t, E), nom, c in got:
        ax.plot(t, E / E[0], color=c, label=nom)
    ax.axhline(1, color="#bbbbbb", ls="--", lw=1)
    ax.set_xlabel("t"); ax.set_ylabel("E(t) / E(0)")
    ax.set_title("Conservación de energía: E(t) normalizada (f=0)")
    ax.legend()
    save(fig, out, "30_energia_curvas", fmt)


def fig_energia_deriva_t(R, out, fmt):
    series = [("en_fd", "FD", GRIS_FD), ("en_fem", "FEM", GRIS_FEM),
              ("en_cls", "PINN", COL["pinn"]), ("en_dv", "QCPINN", COL["qcpinn"])]
    got = []
    for g, nom, c in series:
        r = load_energia(R, g)
        if r:
            got.append((r, nom, c))
    if not got:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for (t, E), nom, c in got:
        ax.plot(t, np.abs(E - E[0]) / abs(E[0]), color=c, label=nom)
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("|E(t)-E(0)| / E(0)  [log]")
    ax.set_title("Deriva acumulada de energía en el tiempo")
    ax.legend()
    save(fig, out, "31_energia_deriva_t", fmt)




def _ref_l2(R, cands):
    for c in cands:
        p = os.path.join(R, c, "metrics.json")
        if os.path.exists(p):
            v = num(json.load(open(p)).get("L2_u_global_pct"))
            if v is not None:
                return v
    return None


def fig_cuatro_metodos_masa(M, R, out, fmt):
    """Los 4 métodos de solución, error L2 sin masa (β=0) vs con masa (β=1)."""
    metodos = [
        ("FD", _ref_l2(R, ["fd_reference_beta0", "fd_reference", "fd_kg", "fd_kg_beta0"]),
         _ref_l2(R, ["fd_reference_beta1"]), GRIS_FD),
        ("FEM", _ref_l2(R, ["fem_reference_beta0", "fem_reference", "fem_kg", "fem_kg_beta0"]),
         _ref_l2(R, ["fem_reference_beta1"]), GRIS_FEM),
        ("PINN", stats(col(M.get("pinn", []), "L2_u_global_pct"))[0],
         stats(col(M.get("beta1_pinn", []), "L2_u_global_pct"))[0], COL["pinn"]),
        ("QCPINN", stats(col(M.get("qcpinn", []), "L2_u_global_pct"))[0],
         stats(col(M.get("beta1_qcpinn", []), "L2_u_global_pct"))[0], COL["qcpinn"]),
    ]
    metodos = [m for m in metodos if (m[1] is not None and m[1] == m[1])
               or (m[2] is not None and m[2] == m[2])]
    if len(metodos) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(metodos)); w = 0.38
    from matplotlib.patches import Patch
    for j, (idx, al, hatch) in enumerate([(1, 0.5, "//"), (2, 0.95, None)]):
        for i, m in enumerate(metodos):
            v = m[idx]
            if v is None or v != v:
                continue
            ax.bar(i + (j - 0.5) * w, v, w, color=m[3], alpha=al, hatch=hatch,
                   edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metodos])
    ax.set_ylabel("error L2 global de u (%)  [escala log]")
    ax.set_title("Los cuatro métodos de solución: sin masa vs con masa")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.5, hatch="//", label="beta=0"),
                       Patch(facecolor="gray", alpha=0.95, label="beta=1")])
    save(fig, out, "32_cuatro_metodos_masa", fmt)


def fig_solucion_masa_qcpinn(R, out, fmt):
    """Campo de la solución predicha por QCPINN, β=0 vs β=1 (u exacta idéntica)."""
    f0, f1 = load_field(R, "qcpinn"), load_field(R, "beta1_qcpinn")
    if not (f0 and f1):
        return
    m = max(np.nanmax(np.abs(f0["up"])), np.nanmax(np.abs(f1["up"])))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, et in [(axs[0], f0, "beta=0"), (axs[1], f1, "beta=1")]:
        im = _campo(ax, f["t"], f["x"], f["up"], "RdBu_r", vmin=-m, vmax=m)
        ax.set_title(f"u predicha, {et} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="u(t,x)", shrink=0.85)
    fig.suptitle("Solución predicha por QCPINN: sin masa vs con masa (u exacta idéntica)", y=1.02)
    save(fig, out, "33_solucion_masa_qcpinn", fmt)


def fig_residuo_masa_qcpinn(R, out, fmt):
    """Campo del residuo de QCPINN, β=0 vs β=1."""
    f0, f1 = load_field(R, "qcpinn"), load_field(R, "beta1_qcpinn")
    if not (f0 and f1):
        return
    vmax = max(np.nanmax(f0["ferr"]), np.nanmax(f1["ferr"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, et in [(axs[0], f0, "beta=0"), (axs[1], f1, "beta=1")]:
        im = _campo(ax, f["t"], f["x"], f["ferr"], "magma", vmin=0, vmax=vmax)
        ax.set_title(f"residuo, {et} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|residuo f|", shrink=0.85)
    fig.suptitle("Residuo de QCPINN: sin masa vs con masa (misma escala)", y=1.02)
    save(fig, out, "34_residuo_masa_qcpinn", fmt)




BETA_COL = {0: "#56B4E9", 1: "#D55E00"}
NOM_B1 = {"beta1_pinn": "PINN", "beta1_none": "sin centro",
          "beta1_cos": "cos(z)", "beta1_qcpinn": "QCPINN"}


def load_field_mean(R, grupo):
    """Promedia err y ferr sobre TODAS las semillas del grupo. dict o None."""
    pat = PATRON.get(grupo)
    if not pat:
        return None
    errs = ferrs = t = x = None
    ns = 0
    for run in glob.glob(os.path.join(R, pat, "*")):
        ep = os.path.join(run, "eval.npz")
        if not os.path.exists(ep):
            continue
        d = np.load(ep)
        X = d["X"]; n = int(round(math.sqrt(X.shape[0])))
        if n * n != X.shape[0]:
            continue
        us = d["u_star"].ravel().reshape(n, n); up = d["u_pred"].ravel().reshape(n, n)
        fs = d["f_star"].ravel().reshape(n, n); fp = d["f_pred"].ravel().reshape(n, n)
        e = np.abs(up - us); fe = np.abs(fp - fs)
        if errs is None:
            errs, ferrs = e.copy(), fe.copy()
            t = X[:, 0].reshape(n, n); x = X[:, 1].reshape(n, n)
        elif e.shape == errs.shape:
            errs += e; ferrs += fe
        else:
            continue
        ns += 1
    if ns == 0:
        return None
    return {"t": t, "x": x, "err": errs / ns, "ferr": ferrs / ns, "n": ns}


def _mapa_prom(R, g0, g1, nom, archivo, out, fmt):
    f0, f1 = load_field_mean(R, g0), load_field_mean(R, g1)
    if not (f0 and f1):
        return
    vmax = max(np.nanmax(f0["err"]), np.nanmax(f1["err"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, et in [(axs[0], f0, "beta=0"), (axs[1], f1, "beta=1")]:
        im = _campo(ax, f["t"], f["x"], f["err"], "viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{et}  (promedio de {f['n']} semillas)")
    fig.colorbar(im, ax=axs, label="|u_pred - u_exacta| media", shrink=0.85)
    fig.suptitle(f"Campo de error promedio de {nom}: sin masa vs con masa", y=1.02)
    save(fig, out, archivo, fmt)


def fig_mapa_prom_qcpinn(R, out, fmt):
    _mapa_prom(R, "qcpinn", "beta1_qcpinn", "QCPINN", "35_mapa_error_promedio_qcpinn", out, fmt)


def fig_mapa_prom_pinn(R, out, fmt):
    _mapa_prom(R, "pinn", "beta1_pinn", "PINN", "36_mapa_error_promedio_pinn", out, fmt)


def fig_espectro(R, out, fmt):
    f0, f1 = load_field_mean(R, "qcpinn"), load_field_mean(R, "beta1_qcpinn")
    got = [(f0, "beta=0", 0), (f1, "beta=1", 1)]
    got = [g for g in got if g[0]]
    if not got:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for f, nom, b in got:
        err = f["err"]                       # (nx, nt)
        nx = err.shape[0]; dx = 1.0 / (nx - 1)
        P = (np.abs(np.fft.rfft(err, axis=0)) ** 2).mean(axis=1)
        k = np.fft.rfftfreq(nx, d=dx)
        ax.plot(k[1:], P[1:], color=BETA_COL[b], label=nom)
    ax.set_yscale("log")
    ax.set_xlabel("frecuencia espacial en x  (ciclos/longitud)")
    ax.set_ylabel("potencia del error  [log]")
    ax.set_title("Espectro espacial del error de QCPINN (β=0 vs β=1)")
    ax.legend()
    save(fig, out, "37_espectro_error_qcpinn", fmt)


def fig_grad_var(R, out, fmt):
    got = []
    for g, nom, b in [("qcpinn", "beta=0", 0), ("beta1_qcpinn", "beta=1", 1)]:
        r = load_metrics_curva(R, g)
        if r and "grad_var_q" in r[0]:
            got.append((r[0], nom, b))
    if not got:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    algo = False
    for c, nom, b in got:
        ep, gv = c["epoch"], c["grad_var_q"]
        mask = ~np.isnan(gv)
        if mask.sum() > 1:
            ax.plot(ep[mask], gv[mask], color=BETA_COL[b], label=nom)
            algo = True
    if not algo:
        plt.close(fig); return
    ax.set_yscale("log")
    ax.set_xlabel("épocas"); ax.set_ylabel("varianza del gradiente cuántico  [log]")
    ax.set_title("Varianza del gradiente del circuito (meseta estéril)")
    ax.legend()
    save(fig, out, "38_varianza_gradiente", fmt)


def fig_causal(R, out, fmt):
    bars = []
    for g, nom, base, b in [("pinn", "PINN β=0", "pinn", 0),
                            ("beta1_pinn", "PINN β=1", "pinn", 1),
                            ("qcpinn", "QCPINN β=0", "qcpinn", 0),
                            ("beta1_qcpinn", "QCPINN β=1", "qcpinn", 1)]:
        f = load_field_mean(R, g)
        if not f:
            continue
        err, tv = f["err"], f["t"][0, :]
        A = err[:, tv < 0.5].mean(); B = err[:, tv >= 0.5].mean()
        if A > 0:
            bars.append((nom, B / A, COL[base], b))
    if len(bars) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (nom, r, c, b) in enumerate(bars):
        ax.bar(i, r, 0.6, color=c, alpha=0.5 if b == 0 else 0.95,
               hatch="//" if b == 0 else None, edgecolor="white")
        ax.annotate(f"{r:.2f}", (i, r), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.axhline(1, color="#999999", ls="--", lw=1)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([n for n, *_ in bars], rotation=12)
    ax.set_ylabel("Error de la 2.ª mitad / error de la 1.ª mitad")
    ax.set_title("Cociente del error entre mitades de la ventana temporal")
    save(fig, out, "39_prueba_causal", fmt)


def fig_curvas_perdida_beta1(R, out, fmt):
    series = []
    for g, nom, base in [("beta1_pinn", "PINN", "pinn"), ("beta1_qcpinn", "QCPINN", "qcpinn")]:
        r = load_metrics_curva(R, g)
        if r and "loss" in r[0]:
            series.append((r[0]["epoch"], r[0]["loss"], nom, COL[base]))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for ep, lo, nom, c in series:
        ax.plot(ep, np.maximum(lo, 1e-12), color=c, label=nom)
    ax.set_yscale("log")
    ax.set_xlabel("épocas"); ax.set_ylabel("pérdida media  [log]")
    ax.set_title("Curvas de pérdida con masa (β=1)")
    ax.legend()
    save(fig, out, "40_curvas_perdida_beta1", fmt)


def fig_regiones_beta1(M, out, fmt):
    orden = ["beta1_pinn", "beta1_none", "beta1_cos", "beta1_qcpinn"]
    orden = [k for k in orden if M.get(k)]
    if len(orden) < 2:
        return
    regs = [("L2_u_global_pct", "global"), ("L2_u_interior_pct", "interior"),
            ("L2_u_boundary_pct", "frontera"), ("L2_u_t0_pct", "t=0")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.2; x = np.arange(len(orden)); tono = [1.0, 0.75, 0.5, 0.3]
    from matplotlib.patches import Patch
    cbase = {"beta1_pinn": "pinn", "beta1_none": "ctrl_none",
             "beta1_cos": "ctrl_cos", "beta1_qcpinn": "qcpinn"}
    for j, (cc, nom) in enumerate(regs):
        vals = [stats(col(M[k], cc))[0] for k in orden]
        ax.bar(x + (j - 1.5) * w, vals, w, color=[COL[cbase[k]] for k in orden],
               alpha=tono[j], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([NOM_B1[k] for k in orden])
    ax.set_ylabel("error L2 (%)")
    ax.set_title("Error por región con masa (β=1)")
    ax.legend(handles=[Patch(facecolor="gray", alpha=tono[j], label=n)
                       for j, (_, n) in enumerate(regs)], ncol=2)
    save(fig, out, "41_regiones_beta1", fmt)




def fig_lbfgs_beta1(M, out, fmt):
    grupos = [("beta1_lbfgs_cls", "clásico (β=1)", "pinn"),
              ("beta1_lbfgs_dv", "híbrido (β=1)", "qcpinn")]
    grupos = [g for g in grupos if M.get(g[0])]
    if not grupos:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(grupos)); w = 0.38
    for j, (c, hatch, al) in enumerate([("L2_u_global_pct_adam", "//", 0.5),
                                        ("L2_u_global_pct", None, 0.95)]):
        mus = [stats(col(M[k], c))[0] for k, _, _ in grupos]
        sds = [stats(col(M[k], c))[1] for k, _, _ in grupos]
        ax.bar(x + (j - 0.5) * w, mus, w, yerr=sds, capsize=3,
               color=[COL[b] for _, _, b in grupos], alpha=al, hatch=hatch, edgecolor="white")
    for i, (k, _, _) in enumerate(grupos):
        v = col(M[k], "L2_u_global_pct")
        ax.scatter(np.full(len(v), i + 0.19) + np.random.uniform(-0.05, 0.05, len(v)),
                   v, s=16, color="black", alpha=0.5, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n, _ in grupos])
    ax.set_ylabel("error L2 global de u (%)")
    ax.set_title("Efecto de L-BFGS con masa (β=1)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.5, hatch="//", label="Adam"),
                       Patch(facecolor="gray", alpha=0.95, label="Adam + L-BFGS")])
    save(fig, out, "42_lbfgs_beta1", fmt)


def fig_campos_adam_vs_lbfgs_beta1(R, out, fmt):
    fa, fl = load_field(R, "beta1_qcpinn"), load_field(R, "beta1_lbfgs_dv")
    if not (fa and fl):
        return
    vmax = max(np.nanmax(fa["err"]), np.nanmax(fl["err"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f, nom in [(axs[0], fa, "QCPINN Adam (β=1)"), (axs[1], fl, "QCPINN L-BFGS (β=1)")]:
        im = _campo(ax, f["t"], f["x"], f["err"], "viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{nom} (semilla {f['sd']})")
    fig.colorbar(im, ax=axs, label="|u_pred - u_exacta|", shrink=0.85)
    fig.suptitle("Campo de error con masa: Adam vs L-BFGS (misma escala)", y=1.02)
    save(fig, out, "43_campos_adam_vs_lbfgs_beta1", fmt)


def fig_pesos_beta1(M, out, fmt):
    orden = [("beta1_pinn", 0.1), ("beta1_wres1", 1), ("beta1_wres10", 10),
             ("beta1_wres100", 100)]
    orden = [(k, w) for k, w in orden if M.get(k)]
    if len(orden) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for c, nom, mark in [("L2_u_global_pct", "global", "o"),
                         ("L2_u_interior_pct", "interior", "s"),
                         ("L2_u_boundary_pct", "frontera", "^")]:
        ws = [w for _, w in orden]
        ys = [stats(col(M[k], c))[0] for k, _ in orden]
        ax.plot(ws, ys, marker=mark, label=nom)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("peso del residuo  λ_f"); ax.set_ylabel("error L2 (%)  [log]")
    ax.set_title("Barrido del peso del residuo con masa (β=1)")
    ax.legend()
    save(fig, out, "44_barrido_pesos_beta1", fmt)




# --- figuras con los nombres EXACTOS que referencia la monografia (datos ya existentes) ---
def fig_masa_boxplot(M, out, fmt):
    orden = ["beta1_pinn", "beta1_none", "beta1_cos", "beta1_qcpinn"]
    cbase = {"beta1_pinn": "pinn", "beta1_none": "ctrl_none",
             "beta1_cos": "ctrl_cos", "beta1_qcpinn": "qcpinn"}
    datos = [(k, col(M.get(k, []), "L2_u_global_pct")) for k in orden]
    datos = [(k, v) for k, v in datos if v]
    if len(datos) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, v) in enumerate(datos):
        c = COL[cbase[k]]
        bp = ax.boxplot(v, positions=[i], widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor=c, alpha=0.6))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.55, edgecolor=c)
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.12, 0.12, len(v)),
                   v, s=16, color=c, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels([NOM_B1[k] for k, _ in datos])
    ax.set_ylabel("error L2 global de u (%)  [log]")
    ax.set_title("Distribución del error con masa (β=1), por modelo")
    save(fig, out, "17_masa_boxplot", fmt)


def fig_masa_regiones(M, out, fmt):
    orden = ["beta1_pinn", "beta1_none", "beta1_cos", "beta1_qcpinn"]
    orden = [k for k in orden if M.get(k)]
    if len(orden) < 2:
        return
    cbase = {"beta1_pinn": "pinn", "beta1_none": "ctrl_none",
             "beta1_cos": "ctrl_cos", "beta1_qcpinn": "qcpinn"}
    regs = [("L2_u_global_pct", "global"), ("L2_u_interior_pct", "interior"),
            ("L2_u_boundary_pct", "frontera"), ("L2_u_t0_pct", "t=0")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.2; x = np.arange(len(orden)); tono = [1.0, 0.75, 0.5, 0.3]
    from matplotlib.patches import Patch
    for j, (cc, nom) in enumerate(regs):
        vals = [stats(col(M[k], cc))[0] for k in orden]
        ax.bar(x + (j - 1.5) * w, vals, w, color=[COL[cbase[k]] for k in orden],
               alpha=tono[j], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([NOM_B1[k] for k in orden])
    ax.set_ylabel("error L2 (%)")
    ax.set_title("Error por región con masa (β=1)")
    ax.legend(handles=[Patch(facecolor="gray", alpha=tono[j], label=n)
                       for j, (_, n) in enumerate(regs)], ncol=2)
    save(fig, out, "18_masa_regiones", fmt)


def fig_masa_contrastes(M, out, fmt):
    contr = []
    def add(nom, a, b):
        if M.get(a) and M.get(b):
            d, lo, hi = welch_ci(col(M[a], "L2_u_global_pct"), col(M[b], "L2_u_global_pct"))
            if lo == lo:
                contr.append((nom, d, lo, hi))
    add("QCPINN - PINN", "beta1_qcpinn", "beta1_pinn")
    add("QCPINN - cos(z)", "beta1_qcpinn", "beta1_cos")
    add("cos(z) - PINN", "beta1_cos", "beta1_pinn")
    add("sin centro - PINN", "beta1_none", "beta1_pinn")
    if not contr:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(contr) + 1.5))
    for i, (nom, d, lo, hi) in enumerate(contr):
        y = len(contr) - 1 - i
        c = "#b00020" if (lo > 0 or hi < 0) else "#444444"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=2)
        ax.scatter([d], [y], s=55, color=c, zorder=3)
        ax.annotate(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", (d, y),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axvline(0, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.set_yticks(range(len(contr)))
    ax.set_yticklabels([n for n, *_ in reversed(contr)])
    ax.set_xlabel("diferencia de error L2 global con masa (pp)   IC 95%")
    ax.set_title("Contrastes con masa (β=1) (rojo = IC excluye 0)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "19_masa_contrastes", fmt)


def fig_energia_semillas(M, out, fmt):
    grupos = [("energia_cls_lbfgs", "PINN", "pinn"),
              ("energia_dv_lbfgs", "QCPINN", "qcpinn")]
    datos = []
    for k, nom, base in grupos:
        da = col(M.get(k, []), "drift_rel_adam")
        dl = col(M.get(k, []), "drift_rel")
        if da or dl:
            datos.append((nom, base, da, dl))
    if not datos:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    pos = 0; ticks = []; labels = []
    for nom, base, da, dl in datos:
        for vals, et, al in [(da, "Adam", 0.45), (dl, "L-BFGS", 1.0)]:
            if vals:
                ax.scatter(np.full(len(vals), pos) + np.random.uniform(-0.1, 0.1, len(vals)),
                           vals, s=28, color=COL[base], alpha=al, edgecolor="white",
                           linewidth=0.5, zorder=3)
                med, _, _ = medrange(vals)
                ax.plot([pos - 0.22, pos + 0.22], [med, med], color="black", lw=1.6, zorder=4)
                ticks.append(pos); labels.append(f"{nom}\n{et}")
                pos += 1
        pos += 0.5
    ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_ylabel("deriva relativa de E(t)  [log]")
    ax.set_title("Deriva de energía por semilla (barra negra = mediana)")
    save(fig, out, "20_energia_semillas", fmt)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis_revision")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figuras")
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--seed", type=int, default=0, help="jitter reproducible")
    a = ap.parse_args()
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    M = load_master(os.path.join(a.indir, "revision_por_semilla.csv"))
    if not M:
        print(f"No encontre {a.indir}/revision_por_semilla.csv. Corre antes "
              f"comparar_revision.py.")
    print(f"Figuras ({a.fmt}) en {a.out}/ :")
    fig_box_controles(M, a.out, a.fmt)
    fig_strip_semillas(M, a.out, a.fmt)
    fig_regiones(M, a.out, a.fmt)
    fig_params(M, a.out, a.fmt)
    fig_masa(M, a.out, a.fmt)
    fig_lbfgs(M, a.out, a.fmt)
    fig_energia(M, a.results, a.out, a.fmt)
    fig_pesos(M, a.out, a.fmt)
    fig_forest(M, a.out, a.fmt)
    fig_trayectoria(a.indir, M, a.out, a.fmt)
    fig_curvas_perdida(a.indir, a.out, a.fmt)
    fig_box_lbfgs(M, a.out, a.fmt)
    fig_cuatro_metodos(M, a.results, a.out, a.fmt)
    fig_slope_adam_lbfgs(M, a.out, a.fmt)
    fig_factor_lbfgs(M, a.out, a.fmt)
    fig_qcpinn_regimenes(M, a.out, a.fmt)
    fig_campo_error_qcpinn_masa(a.results, a.out, a.fmt)
    fig_campos_pinn_vs_qcpinn(a.results, a.out, a.fmt)
    fig_campos_adam_vs_lbfgs(a.results, a.out, a.fmt)
    fig_error_vs_t(a.results, a.out, a.fmt)
    fig_perfiles(a.results, a.out, a.fmt)
    fig_residuo_pinn_vs_qcpinn(a.results, a.out, a.fmt)
    fig_l2_residuo(M, a.out, a.fmt)
    fig_residuo_adam_vs_lbfgs(a.results, a.out, a.fmt)
    fig_componentes_perdida(a.results, a.out, a.fmt)
    fig_error_restricciones(M, a.out, a.fmt)
    fig_campo_masa_qcpinn(a.results, a.out, a.fmt)
    fig_campo_masa_pinn(a.results, a.out, a.fmt)
    fig_hist_error(a.results, a.out, a.fmt)
    fig_energia_curvas(a.results, a.out, a.fmt)
    fig_energia_deriva_t(a.results, a.out, a.fmt)
    fig_cuatro_metodos_masa(M, a.results, a.out, a.fmt)
    fig_solucion_masa_qcpinn(a.results, a.out, a.fmt)
    fig_residuo_masa_qcpinn(a.results, a.out, a.fmt)
    fig_mapa_prom_qcpinn(a.results, a.out, a.fmt)
    fig_mapa_prom_pinn(a.results, a.out, a.fmt)
    fig_espectro(a.results, a.out, a.fmt)
    fig_grad_var(a.results, a.out, a.fmt)
    fig_causal(a.results, a.out, a.fmt)
    fig_curvas_perdida_beta1(a.results, a.out, a.fmt)
    fig_regiones_beta1(M, a.out, a.fmt)
    fig_lbfgs_beta1(M, a.out, a.fmt)
    fig_campos_adam_vs_lbfgs_beta1(a.results, a.out, a.fmt)
    fig_pesos_beta1(M, a.out, a.fmt)
    fig_masa_boxplot(M, a.out, a.fmt)
    fig_masa_regiones(M, a.out, a.fmt)
    fig_masa_contrastes(M, a.out, a.fmt)
    fig_energia_semillas(M, a.out, a.fmt)
    print("Listo.")


if __name__ == "__main__":
    main()
