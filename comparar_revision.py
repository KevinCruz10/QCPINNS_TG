#!/usr/bin/env python3
"""
comparar_revision.py - Agrega y compara los experimentos de la revision del
profesor. Lee los metrics.json que dejaron run_kg2.py y la cola, en results/.
No entrena nada; corre en segundos.

    python comparar_revision.py
    python comparar_revision.py --results results --out analysis_revision

Produce en --out:
    revision_resumen.csv     una fila por grupo (media +- sd de cada metrica)
    revision_lbfgs.csv       antes/despues de L-BFGS por semilla
    revision_report.txt      el mismo texto que imprime, para pegar

Cubre los puntos del profesor:
    p1  control del cuello de botella   (bn_* vs qcpinn vs pinn)
    p3  peso del residuo                (wres* : interior vs frontera)
    p4  L-BFGS tras Adam                (antes/despues)
    p6  intervalo de confianza          (QCPINN - PINN, no "equivalencia")
    p8  termino de masa                 (beta1_* vs referencia numerica)
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def find_metrics(results_dir, tag):
    """metrics.json bajo results/<tag>/<timestamp>/ o results/<tag>/ directamente."""
    paths = glob.glob(os.path.join(results_dir, tag, "*", "metrics.json"))
    direct = os.path.join(results_dir, tag, "metrics.json")
    if os.path.exists(direct):
        paths.append(direct)
    return sorted(set(paths))


def load_group(results_dir, pattern):
    """pattern puede llevar comodin de semilla (kg2_cls_wres1_seed*)."""
    out = []
    bases = glob.glob(os.path.join(results_dir, pattern))
    for base in bases:
        tag = os.path.basename(base)
        for p in find_metrics(results_dir, tag):
            try:
                out.append(json.load(open(p)))
            except Exception as e:
                print(f"  aviso: no se pudo leer {p}: {e}")
    return out


def stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return (float("nan"), 0.0, 0)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return (m, sd, n)


def medrange(vals):
    """Mediana y rango [min, max]: robustos a outliers, donde la media enganha."""
    vals = sorted(v for v in vals if isinstance(v, (int, float)))
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return (med, vals[0], vals[-1])


def welch(a, b):
    """Prueba t de Welch y dof. Devuelve (t, dof, p_aprox_dosolado)."""
    ma, sa, na = stats(a)
    mb, sb, nb = stats(b)
    if na < 2 or nb < 2:
        return (float("nan"), float("nan"), float("nan"))
    va, vb = sa ** 2 / na, sb ** 2 / nb
    se = math.sqrt(va + vb)
    if se == 0:
        return (float("nan"), float("nan"), float("nan"))
    t = (ma - mb) / se
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    # p de dos colas via aproximacion de la t de Student (sin scipy)
    p = _t_sf(abs(t), dof) * 2
    return (t, dof, p)


def _t_sf(t, dof):
    """Cola superior de la t de Student: P(T > t) para t >= 0."""
    x = dof / (dof + t * t)
    return 0.5 * _betai(dof / 2.0, 0.5, x)


def _betacf(a, b, x):
    """Fraccion continua de la beta incompleta (Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def _betai(a, b, x):
    """Beta incompleta regularizada I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_crit_95(dof):
    """t critica de dos colas al 95% por biseccion sobre _t_sf."""
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _t_sf(mid, dof) > 0.025:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def key(ms, k):
    return [m.get(k) for m in ms if m.get(k) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis_revision")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = a.results
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # grupos: etiqueta -> patron de carpeta
    GRUPOS = {
        "qcpinn":        "kg_dv_angle_cascade_seed*",
        "pinn":          "kg_classical_seed*",
        "ctrl_none":     "kg2_bn_none_seed*",
        "ctrl_cos":      "kg2_bn_cos_seed*",
        "ctrl_tanh":     "kg2_bn_tanh_seed*",
        "ctrl_linear":   "kg2_bn_linear_seed*",
        "wres0.1":       "kg_classical_seed*",
        "wres1":         "kg2_cls_wres1_seed*",
        "wres10":        "kg2_cls_wres10_seed*",
        "wres100":       "kg2_cls_wres100_seed*",
        "adamext":       "kg2_cls_adamext_seed*",
        "lbfgs":         "kg2_cls_lbfgs_seed*",
        "lbfgs_cls800":  "kg2_cls_lbfgs800_seed*",
        "lbfgs_dv":      "kg2_dv_lbfgs_seed*",
        "beta1_pinn":    "kg2_cls_beta1_seed*",
        "beta1_ctrl":    "kg2_bn_linear_beta1_seed*",
        "beta1_qcpinn":  "kg2_dv_beta1_seed*",
        "beta1_cos":     "kg2_bn_cos_beta1_seed*",
        "beta1_none":    "kg2_bn_none_beta1_seed*",
    }
    data = {k: load_group(R, pat) for k, pat in GRUPOS.items()}

    rows = []
    for k, ms in data.items():
        if not ms:
            continue
        mu_u, sd_u, n = stats(key(ms, "L2_u_global_pct"))
        mu_i, sd_i, _ = stats(key(ms, "L2_u_interior_pct"))
        mu_b, sd_b, _ = stats(key(ms, "L2_u_boundary_pct"))
        mu_f, sd_f, _ = stats(key(ms, "L2_f_global_pct"))
        np_, _, _ = stats(key(ms, "n_params"))
        rows.append({"grupo": k, "n": n, "n_params": int(np_) if np_ == np_ else "",
                     "L2u": mu_u, "L2u_sd": sd_u, "L2int": mu_i, "L2bnd": mu_b,
                     "L2f": mu_f})
    with open(os.path.join(a.out, "revision_resumen.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # volcado COMPLETO por semilla: todos los grupos y todas las metricas crudas
    campos = ["L2_u_global_pct", "L2_u_global_pct_adam", "L2_u_interior_pct",
              "L2_u_boundary_pct", "L2_u_t0_pct", "L2_f_global_pct", "n_params",
              "epochs", "beta", "drift_rel", "drift_rel_adam", "lbfgs_iters"]
    energia_grupos = {
        "energia_cls": "energia_cls_seed*", "energia_dv": "energia_dv_seed*",
        "energia_cls_lbfgs": "energia_cls_lbfgs_seed*",
        "energia_dv_lbfgs": "energia_dv_lbfgs_seed*",
    }
    with open(os.path.join(a.out, "revision_por_semilla.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["grupo", "seed"] + campos)
        for k in GRUPOS:
            for m in data.get(k, []):
                w.writerow([k, m.get("seed")] + [m.get(c, "") for c in campos])
        for k, pat in energia_grupos.items():
            for m in load_group(R, pat):
                w.writerow([k, m.get("seed")] + [m.get(c, "") for c in campos])

    def show(k):
        ms = data.get(k, [])
        if not ms:
            return f"  {k:14s} (sin datos)"
        mu, sd, n = stats(key(ms, "L2_u_global_pct"))
        med, lo, hi = medrange(key(ms, "L2_u_global_pct"))
        npv, _, _ = stats(key(ms, "n_params"))
        return (f"  {k:14s} n={n:2d}  N_p={int(npv):5d}  L2_u = {mu:6.3f} +- {sd:5.3f} %"
                f"  | mediana {med:6.3f}  rango [{lo:.2f}, {hi:.2f}]")

    say("=" * 66)
    say("PUNTO 1 - CONTROL DEL CUELLO DE BOTELLA")
    say("=" * 66)
    say("Misma forma 2->50->5->(centro)->50->1. El centro es lo unico que cambia.")
    for k in ["pinn", "ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear", "qcpinn"]:
        say(show(k))
    if data.get("qcpinn") and data.get("ctrl_linear"):
        for ctrl in ["ctrl_none", "ctrl_cos", "ctrl_tanh", "ctrl_linear"]:
            if data.get(ctrl):
                t, dof, p = welch(key(data["qcpinn"], "L2_u_global_pct"),
                                  key(data[ctrl], "L2_u_global_pct"))
                say(f"  QCPINN vs {ctrl:12s}: Welch p = {p:.3f}")
        say("  Lectura: si un control iguala al QCPINN (p alto), el circuito no")
        say("  aporta nada que una capa clasica de 5 dim no haga.")

    say("")
    say("=" * 66)
    say("PUNTO 3 - PESO DEL RESIDUO (interior vs frontera)")
    say("=" * 66)
    say(f"  {'grupo':10s} {'L2 global':>10s} {'L2 interior':>12s} {'L2 frontera':>12s}")
    for k in ["wres0.1", "wres1", "wres10", "wres100"]:
        ms = data.get(k, [])
        if not ms:
            say(f"  {k:10s} (sin datos)")
            continue
        mu, _, n = stats(key(ms, "L2_u_global_pct"))
        mi, _, _ = stats(key(ms, "L2_u_interior_pct"))
        mb, _, _ = stats(key(ms, "L2_u_boundary_pct"))
        say(f"  {k:10s} {mu:10.3f} {mi:12.3f} {mb:12.3f}   (n={n})")
    say("  Lectura: si al subir w_res el interior baja y la frontera sube, la")
    say("  asimetria era de la ponderacion, no de la imposicion blanda.")

    say("")
    say("=" * 66)
    say("PUNTO 4 - L-BFGS TRAS ADAM (mismo protocolo en ambas arquitecturas)")
    say("=" * 66)
    say(f"  {'grupo':22s} {'Adam':>14s} {'Adam+L-BFGS':>14s}  n")
    def lbfgs_row(etq, k):
        ms = data.get(k, [])
        if not ms:
            return f"  {etq:22s} (sin datos)"
        before = key(ms, "L2_u_global_pct_adam")
        after = key(ms, "L2_u_global_pct")
        mb_, sb_, nb_ = stats(before)
        ma_, sa_, _ = stats(after)
        meda, loa, hia = medrange(after)
        return (f"  {etq:22s} {mb_:6.3f}+-{sb_:5.3f}  {ma_:6.3f}+-{sa_:5.3f}  (n={nb_})"
                f"  | med {meda:5.3f} rango [{loa:.2f},{hia:.2f}]")
    # Experimento 1: dar a Adam muchas mas epocas no lo lleva al error de L-BFGS
    a20 = key(data.get("pinn", []), "L2_u_global_pct")
    aex = key(data.get("adamext", []), "L2_u_global_pct")
    if aex:
        ep_ext = [m.get("epochs") for m in data.get("adamext", []) if m.get("epochs")]
        ep_txt = f"{int(max(ep_ext))}" if ep_ext else "extendidas"
        m20, s20, n20 = stats(a20)
        mex, sex, nex = stats(aex)
        say("  Control de iteraciones (PINN clasica, solo Adam):")
        say(f"    Adam 20k epocas     : {m20:6.3f}% +-{s20:.3f}  (n={n20})")
        say(f"    Adam {ep_txt:>5s} epocas   : {mex:6.3f}% +-{sex:.3f}  (n={nex})")
        say("    -> aun con muchas mas epocas, Adam se queda lejos del error de")
        say("       L-BFGS (abajo). La mejora es de ser 2do orden, no de mas pasos.")
        say("       (la trayectoria L2 por epoca esta en l2_trajectory.csv de cada")
        say("        corrida; usa analizar_perdida.py --l2traj para la curva media.)")
        say("")
    say(lbfgs_row("clasico nres=2000", "lbfgs"))
    say(lbfgs_row("clasico nres=800", "lbfgs_cls800"))
    say(lbfgs_row("hibrido nres=800", "lbfgs_dv"))
    mp, _, _ = stats(key(data.get("pinn", []), "L2_u_global_pct"))
    mq, _, _ = stats(key(data.get("qcpinn", []), "L2_u_global_pct"))
    say(f"  referencia solo-Adam: clasico {mp:.3f} %, hibrido {mq:.3f} %  (cap. 5)")
    # comparacion de las medias post-L-BFGS entre arquitecturas, mismo protocolo
    cl = key(data.get("lbfgs_cls800", []), "L2_u_global_pct")
    dv = key(data.get("lbfgs_dv", []), "L2_u_global_pct")
    if len(cl) > 1 and len(dv) > 1:
        t, dof, p = welch(dv, cl)
        say(f"  hibrido vs clasico tras L-BFGS (nres=800): Welch p = {p:.3f}")
    # guardar csv por semilla de los grupos L-BFGS
    with open(os.path.join(a.out, "revision_lbfgs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grupo", "seed", "L2u_adam", "L2u_lbfgs"])
        for k in ["lbfgs", "lbfgs_cls800", "lbfgs_dv"]:
            for m in data.get(k, []):
                w.writerow([k, m.get("seed"), m.get("L2_u_global_pct_adam"),
                            m.get("L2_u_global_pct")])
    say("  Lectura: si ambas bajan ~6x, el techo de ~4,5 % era de Adam (no de la")
    say("  formulacion PINN). Eso reescribe la conclusion general del capitulo 7.")

    say("")
    say("=" * 66)
    say("PUNTO 6 - INTERVALO DE CONFIANZA  (QCPINN - PINN, SIN masa)")
    say("=" * 66)
    q = key(data.get("qcpinn", []), "L2_u_global_pct")
    pn = key(data.get("pinn", []), "L2_u_global_pct")
    if len(q) > 1 and len(pn) > 1:
        mq, sq, nq = stats(q)
        mp, sp, npn = stats(pn)
        diff = mq - mp
        se = math.sqrt(sq ** 2 / nq + sp ** 2 / npn)
        va, vb = sq ** 2 / nq, sp ** 2 / npn
        dof = (va + vb) ** 2 / (va ** 2 / (nq - 1) + vb ** 2 / (npn - 1))
        tc = t_crit_95(dof)
        lo, hi = diff - tc * se, diff + tc * se
        _, _, p = welch(q, pn)
        say(f"  QCPINN = {mq:.3f} %   PINN = {mp:.3f} %")
        say(f"  diferencia = {diff:+.3f} pp   IC 95% = [{lo:+.3f}, {hi:+.3f}] pp   (p={p:.3f})")
        incluye0 = lo <= 0 <= hi
        if incluye0:
            say("  El IC incluye 0: no hay diferencia detectable, pero el intervalo se")
            say("  extiende varios pp -> los datos son compatibles con que el hibrido")
            say("  sea algo mejor O varios pp peor. NO afirmar 'equivalente'; describir")
            say("  el IC, que es ancho por el n y la dispersion.")
        else:
            lado = "peor" if diff > 0 else "mejor"
            say(f"  El IC NO incluye 0: el hibrido es {lado} de forma detectable en estos")
            say("  datos. Reportar la diferencia con su IC, no como 'equivalencia'.")
    else:
        say("  (faltan datos de qcpinn o pinn)")

    say("")
    say("=" * 66)
    say("PUNTO 8 - TERMINO DE MASA  (sin masa vs beta=1)")
    say("=" * 66)
    say(f"  {'modelo':20s} {'sin masa (n)':>18s} {'beta=1 (n)':>18s} {'delta':>14s}")
    def fila_masa(etq, k0, k1):
        m0 = key(data.get(k0, []), "L2_u_global_pct")
        m1 = key(data.get(k1, []), "L2_u_global_pct")
        a0, sd0, n0 = stats(m0)
        a1, sd1, n1 = stats(m1)
        if n0 == 0 and n1 == 0:
            return f"  {etq:20s} (sin datos)"
        s0 = f"{a0:6.3f}+-{sd0:.3f} ({n0})" if n0 else "--"
        s1 = f"{a1:6.3f}+-{sd1:.3f} ({n1})" if n1 else "--"
        if n0 and n1:
            sdd = math.sqrt(sd0 ** 2 / max(n0, 1) + sd1 ** 2 / max(n1, 1))
            d = f"{a1-a0:+6.2f}+-{sdd:.2f}"
        else:
            d = "--"
        return f"  {etq:20s} {s0:>18s} {s1:>18s} {d:>14s}"
    say(fila_masa("PINN Model-2",      "pinn",       "beta1_pinn"))
    say(fila_masa("hibrido (circuito)", "qcpinn",    "beta1_qcpinn"))
    say(fila_masa("ctrl_cos (0 qubits)", "ctrl_cos", "beta1_cos"))
    say(fila_masa("ctrl_none (lineal)", "ctrl_none", "beta1_none"))
    for ref, nombre in [("fd_reference_beta1", "FD  masa"),
                        ("fem_reference_beta1", "FEM masa")]:
        p = os.path.join(R, ref, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            say(f"  {nombre:20s} {'--':>18s} {m.get('L2_u_global_pct'):16.4f}   (ref.)")
    say("  Lectura: si el hibrido con masa (7,4%) y ctrl_cos con masa suben igual,")
    say("  la perdida con masa es del cuello de botella / readout, no del circuito.")
    say("  Si ctrl_cos se queda en ~4%, es algo especifico del circuito con masa.")

    say("")
    say("=" * 66)
    say("ENERGIA - CASO SIN FUENTE (f=0), conservacion en los 4 metodos")
    say("=" * 66)
    say(f"  {'metodo':24s} {'deriva relativa de E(t)':>24s}")
    # numericos: un metrics.json directo
    for ref, nombre in [("energia_fd_beta1", "Diferencias finitas"),
                        ("energia_fem_beta1", "Elementos finitos P1")]:
        p = os.path.join(R, ref, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            say(f"  {nombre:24s} {m.get('drift_rel_ref'):>24.3e}")
    # redes entrenadas con Adam: promedio sobre semillas
    for pat, nombre in [("energia_cls_seed*", "PINN clasica (Adam)"),
                        ("energia_dv_seed*", "QCPINN hibrida (Adam)")]:
        ms = load_group(R, pat)
        if ms:
            mu, sd, n = stats(key(ms, "drift_rel"))
            say(f"  {nombre:24s} {mu:>18.3e} +-{sd:.1e}  (n={n})")
    say("  Lectura: los metodos numericos conservan la energia a ~1e-5; las redes")
    say("  la derivan ordenes de magnitud mas. Es una debilidad fisica de las PINN")
    say("  que el esquema numerico no tiene, y refuerza la discusion fisica.")

    # redes entrenadas con Adam + L-BFGS: antes/despues en la misma corrida
    say("")
    say("  Efecto de L-BFGS sobre la conservacion de energia (misma corrida):")
    say(f"  {'red':16s} {'deriva Adam':>22s} {'deriva L-BFGS':>22s}  {'n':>3s}")
    hubo_lbfgs = False
    for pat, nombre in [("energia_cls_lbfgs_seed*", "PINN clasica"),
                        ("energia_dv_lbfgs_seed*", "QCPINN hibrida")]:
        ms = load_group(R, pat)
        if ms:
            hubo_lbfgs = True
            da = [m.get("drift_rel_adam") for m in ms if m.get("drift_rel_adam") is not None]
            mua, sda, _ = stats(da)
            mul, sdl, nl = stats(key(ms, "drift_rel"))
            say(f"  {nombre:16s} {mua:>11.3e}+-{sda:.1e} {mul:>11.3e}+-{sdl:.1e}  {nl:>3d}")
    if hubo_lbfgs:
        say("  Si la deriva baja con L-BFGS, la mala conservacion era de la")
        say("  optimizacion (Adam), no de la formulacion PINN; si no baja, la")
        say("  debilidad es estructural de la red y no la arregla el optimizador.")
    else:
        say("  (sin corridas de energia con L-BFGS todavia)")

    say("")
    say("Archivos guardados en " + a.out + "/ :")
    say("  revision_report.txt      este reporte completo")
    say("  revision_resumen.csv     media+-sd por grupo")
    say("  revision_por_semilla.csv TODOS los valores crudos, una fila por semilla")
    say("  revision_lbfgs.csv       L2 antes/despues de L-BFGS por semilla")
    with open(os.path.join(a.out, "revision_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nGuardado en {a.out}/revision_report.txt (+ 3 CSV)")


if __name__ == "__main__":
    main()
