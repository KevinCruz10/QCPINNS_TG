#!/usr/bin/env python3
"""
analizar_perdida.py - Experimento 1 (version minima, SIN correr nada):
muestra que la perdida de Adam ya estaba plana a las 20.000 epocas, de modo que
la mejora de L-BFGS no viene de "mas iteraciones" sino de ser de segundo orden.

Lee los metrics.csv que ya existen (epoch, loss) de un grupo de corridas,
promedia la curva sobre las semillas y mide la pendiente en el ultimo tramo.

    python analizar_perdida.py --runs "pinn=results/kg_classical_seed*" \\
                                       "qcpinn=results/kg_dv_angle_cascade_seed*" \\
                               --out analysis_revision

Reporta, por grupo:
    - perdida media en [0.5, 0.75] y en [0.75, 1.0] del entrenamiento
    - caida relativa entre esos dos tramos (si es ~0, ya estaba plana)
    - pendiente media por 1000 epocas en el ultimo cuarto
y guarda curva_perdida_<grupo>.csv (epoch, loss_media) para graficar.
"""
import argparse, csv, glob, os
import numpy as np


def leer_curva(run_dir):
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return None
    ep, lo = [], []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                ep.append(float(r["epoch"])); lo.append(float(r["loss"]))
            except (KeyError, ValueError):
                pass
    if len(ep) < 4:
        return None
    return np.array(ep), np.array(lo)


def cargar_grupo(pattern):
    curvas = []
    for base in glob.glob(pattern):
        for sub in glob.glob(os.path.join(base, "*")):
            c = leer_curva(sub)
            if c is not None:
                curvas.append(c)
        c = leer_curva(base)
        if c is not None:
            curvas.append(c)
    return curvas


def leer_traj(run_dir):
    p = os.path.join(run_dir, "l2_trajectory.csv")
    if not os.path.exists(p):
        return None
    ep, l2 = [], []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                ep.append(float(r["epoch"])); l2.append(float(r["L2_u_global_pct"]))
            except (KeyError, ValueError):
                pass
    return (np.array(ep), np.array(l2)) if len(ep) >= 2 else None


def cargar_traj(pattern):
    out = []
    for base in glob.glob(pattern):
        for sub in glob.glob(os.path.join(base, "*")):
            t = leer_traj(sub)
            if t is not None:
                out.append(t)
        t = leer_traj(base)
        if t is not None:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[], help='pares etiqueta=patron (curva de perdida)')
    ap.add_argument("--l2traj", nargs="+", default=[], help='pares etiqueta=patron (trayectoria L2_u)')
    ap.add_argument("--out", default="analysis_revision")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import builtins
    _log = []
    _p = builtins.print
    def print(*args, **kw):  # noqa: A001 - captura ademas de imprimir
        _p(*args, **kw)
        _log.append(" ".join(str(x) for x in args))

    print(f"{'grupo':14s} {'loss 50-75%':>12s} {'loss 75-100%':>13s} "
          f"{'caida rel.':>11s} {'pend./1000ep':>13s}")
    for spec in a.runs:
        etq, pat = spec.split("=", 1)
        curvas = cargar_grupo(pat)
        if not curvas:
            print(f"{etq:14s} (sin datos)")
            continue
        n = min(len(c[0]) for c in curvas)
        ep = curvas[0][0][:n]
        M = np.vstack([c[1][:n] for c in curvas])
        mu = M.mean(0)

        i50, i75 = int(0.5 * n), int(0.75 * n)
        l_mid = float(mu[i50:i75].mean())
        l_fin = float(mu[i75:].mean())
        caida = (l_mid - l_fin) / l_mid if l_mid else float("nan")
        # pendiente lineal en el ultimo cuarto, por cada 1000 epocas
        ep_q, lo_q = ep[i75:], mu[i75:]
        if len(ep_q) > 2:
            pend = np.polyfit(ep_q, lo_q, 1)[0] * 1000.0
        else:
            pend = float("nan")
        print(f"{etq:14s} {l_mid:12.4e} {l_fin:13.4e} {caida:11.3f} {pend:13.3e}")

        with open(os.path.join(a.out, f"curva_perdida_{etq}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["epoch", "loss_media"])
            for e, l in zip(ep, mu):
                w.writerow([e, l])

    if a.runs:
        print("\n  caida rel. ~ 0 y pendiente ~ 0  =>  Adam ya estaba plano a 20k epocas.")
        print(f"  curvas medias guardadas en {a.out}/curva_perdida_*.csv")

    # --- trayectoria L2_u por epoca (error real, no la perdida) ---
    for spec in a.l2traj:
        etq, pat = spec.split("=", 1)
        trajs = cargar_traj(pat)
        if not trajs:
            print(f"\n[L2 {etq}] (sin l2_trajectory.csv)")
            continue
        n = min(len(t[0]) for t in trajs)
        ep = trajs[0][0][:n]
        M = np.vstack([t[1][:n] for t in trajs])
        mu, sd = M.mean(0), M.std(0)
        print(f"\n[L2 {etq}] error L2_u (%) por epoca, media sobre {len(trajs)} semillas:")
        for i in range(n):
            if i == 0 or i == n - 1 or ep[i] % 10000 == 0:
                print(f"    epoca {int(ep[i]):6d}: {mu[i]:7.3f}% +-{sd[i]:.3f}")
        print(f"    minimo alcanzado: {mu.min():.3f}% en epoca {int(ep[int(mu.argmin())])}")
        with open(os.path.join(a.out, f"traj_l2_{etq}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["epoch", "L2_u_pct_media", "L2_u_pct_sd"])
            for e, m, s in zip(ep, mu, sd):
                w.writerow([e, m, s])
        print(f"    trayectoria media -> {a.out}/traj_l2_{etq}.csv")

    with open(os.path.join(a.out, "analisis_perdida.txt"), "w") as fh:
        fh.write("\n".join(_log) + "\n")
    _p(f"\n  salida guardada en {a.out}/analisis_perdida.txt")


if __name__ == "__main__":
    main()
