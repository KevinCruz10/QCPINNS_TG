#!/usr/bin/env python3
"""
fd_klein_gordon.py - Metodo numerico clasico de referencia para la ecuacion
de Klein-Gordon no lineal 1D. Cierra el objetivo especifico 4 (comparacion
con metodos numericos clasicos).

Colocar en la raiz del repo QCPINN y ejecutar:

    python fd_klein_gordon.py                       # estudio de convergencia completo
    python fd_klein_gordon.py --nx 200              # una sola malla

Problema (identico al de src/data/klein_gordon_dataset.py):

    u_tt - u_xx + u^3 = f(t,x)        (t,x) en [0,1] x [0,1]
    u(t,0) = 0
    u(t,1) = cos(5*pi*t) + t^3
    u(0,x) = x
    u_t(0,x) = 0

Solucion exacta:  u(t,x) = x*cos(5*pi*t) + (t*x)^3

Esquema: diferencias finitas centradas de segundo orden en espacio y tiempo,
explicito, con el termino no lineal evaluado en el nivel n. El arranque usa
la expansion de Taylor de segundo orden con u_t(0,x) = 0. Condicion de
estabilidad tipo CFL: r = dt/dx <= 1; se usa r = 0.5 por el termino no lineal.

Salidas en results/fd_reference/:
    metrics.json      metricas de la malla de referencia (formato compatible con run_kg.py)
    convergence.csv   error y costo por resolucion
    eval_fd.npz       solucion en la malla 200x200 de evaluacion
"""

import argparse
import csv
import json
import os
import resource
import time

import numpy as np


# ----------------------------------------------------------------- problema

def u_exact(T, X):
    return X * np.cos(5.0 * np.pi * T) + (T * X) ** 3


def source(T, X):
    """f = u_tt - u_xx + u^3, derivado analiticamente de la solucion exacta."""
    u_tt = -25.0 * np.pi ** 2 * X * np.cos(5.0 * np.pi * T) + 6.0 * T * X ** 3
    u_xx = 6.0 * T ** 3 * X
    u3 = (X * np.cos(5.0 * np.pi * T) + (T * X) ** 3) ** 3
    return u_tt - u_xx + u3


# ------------------------------------------------------------------ esquema

def solve_fd(nx, r=0.5):
    """Integra el problema en una malla de nx puntos espaciales.
    Devuelve (t, x, U) con U de forma (nt, nx), mas el tiempo de integracion."""
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = r * dx
    nt = int(round(1.0 / dt)) + 1
    dt = 1.0 / (nt - 1)          # ajuste para que t=1 caiga en un nivel exacto
    r_eff = dt / dx
    t = np.linspace(0.0, 1.0, nt)

    U = np.zeros((nt, nx))
    c2 = r_eff ** 2

    t0 = time.perf_counter()

    # nivel 0: condicion inicial
    U[0] = x.copy()

    # nivel 1: Taylor de 2do orden con u_t(0,x) = 0
    #   u^1 = u^0 + dt*v^0 + dt^2/2 * (u_xx - u^3 + f)|_{t=0}
    f0 = source(np.zeros_like(x), x)
    lap0 = np.zeros_like(x)
    lap0[1:-1] = (U[0, 2:] - 2.0 * U[0, 1:-1] + U[0, :-2]) / dx ** 2
    U[1] = U[0] + 0.5 * dt ** 2 * (lap0 - U[0] ** 3 + f0)
    U[1, 0] = 0.0
    U[1, -1] = np.cos(5.0 * np.pi * t[1]) + t[1] ** 3

    # niveles siguientes: salto de rana
    for n in range(1, nt - 1):
        fn = source(np.full_like(x, t[n]), x)
        U[n + 1, 1:-1] = (
            2.0 * U[n, 1:-1] - U[n - 1, 1:-1]
            + c2 * (U[n, 2:] - 2.0 * U[n, 1:-1] + U[n, :-2])
            + dt ** 2 * (fn[1:-1] - U[n, 1:-1] ** 3)
        )
        U[n + 1, 0] = 0.0
        U[n + 1, -1] = np.cos(5.0 * np.pi * t[n + 1]) + t[n + 1] ** 3

    elapsed = time.perf_counter() - t0
    return t, x, U, elapsed, dt, dx


def to_eval_grid(t, x, U, n_eval=200):
    """Interpola la solucion a la malla n_eval x n_eval usada por run_kg.py.
    Convencion de salida: fila = x fijo, columna = t (igual que eval.npz)."""
    from scipy.interpolate import RegularGridInterpolator

    te = np.linspace(0.0, 1.0, n_eval)
    xe = np.linspace(0.0, 1.0, n_eval)
    interp = RegularGridInterpolator((t, x), U, method="linear",
                                     bounds_error=False, fill_value=None)
    Xg, Tg = np.meshgrid(xe, te, indexing="ij")     # (nx_eval, nt_eval)
    pts = np.stack([Tg.ravel(), Xg.ravel()], axis=1)
    return te, xe, interp(pts).reshape(n_eval, n_eval)


def rel_l2(pred, exact, mask=None):
    if mask is not None:
        pred, exact = pred[mask], exact[mask]
    den = np.linalg.norm(exact)
    if den == 0.0:
        return float(np.linalg.norm(pred - exact) * 100.0)
    return float(np.linalg.norm(pred - exact) / den * 100.0)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, nargs="+", default=[51, 101, 201, 401],
                    help="resoluciones espaciales del estudio de convergencia")
    ap.add_argument("--ref-nx", type=int, default=None,
                    help="malla usada como referencia para metrics.json (por defecto, la mas fina)")
    ap.add_argument("--cfl", type=float, default=0.5, help="r = dt/dx")
    ap.add_argument("--eval-points", type=int, default=200)
    ap.add_argument("--out", default="results/fd_reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref_nx = args.ref_nx or max(args.nx)
    n_eval = args.eval_points

    te = np.linspace(0.0, 1.0, n_eval)
    xe = np.linspace(0.0, 1.0, n_eval)
    Xg, Tg = np.meshgrid(xe, te, indexing="ij")     # fila = x, columna = t
    U_exact_grid = u_exact(Tg, Xg)

    rows = []
    ref_payload = None

    print(f"{'nx':>6} {'nt':>7} {'dx':>10} {'dt':>10} {'L2_u (%)':>10} "
          f"{'orden':>7} {'tiempo (s)':>11}")
    prev_err = prev_dx = None

    for nx in sorted(args.nx):
        t, x, U, elapsed, dt, dx = solve_fd(nx, r=args.cfl)
        if not np.isfinite(U).all():
            print(f"{nx:>6} {'--':>7}  divergio (revisa la condicion CFL)")
            continue
        _, _, U_grid = to_eval_grid(t, x, U, n_eval)
        err = rel_l2(U_grid, U_exact_grid)

        order = ""
        if prev_err is not None and err > 0:
            order = f"{np.log(prev_err / err) / np.log(prev_dx / dx):.2f}"
        prev_err, prev_dx = err, dx

        print(f"{nx:>6} {len(t):>7} {dx:>10.2e} {dt:>10.2e} {err:>10.4f} "
              f"{order:>7} {elapsed:>11.4f}")
        rows.append({"nx": nx, "nt": len(t), "dx": dx, "dt": dt,
                     "L2_u_global_pct": err, "orden_observado": order,
                     "tiempo_s": elapsed})

        if nx == ref_nx:
            ref_payload = (t, x, U, U_grid, elapsed, dt, dx)

    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if ref_payload is None:
        print("\nNo se genero malla de referencia; revisa --ref-nx.")
        return

    t, x, U, U_grid, elapsed, dt, dx = ref_payload
    np.savez_compressed(os.path.join(args.out, "eval_fd.npz"),
                        t=te, x=xe, u_fd=U_grid, u_exact=U_exact_grid)

    m_t0 = np.zeros_like(U_grid, dtype=bool); m_t0[:, 0] = True
    m_x0 = np.zeros_like(U_grid, dtype=bool); m_x0[0, :] = True
    m_x1 = np.zeros_like(U_grid, dtype=bool); m_x1[-1, :] = True
    m_bnd = m_t0 | m_x0 | m_x1
    m_int = ~m_bnd

    metrics = {
        "metodo": "diferencias finitas explicitas (2do orden espacio y tiempo)",
        "nx": int(ref_nx), "nt": int(len(t)), "dx": float(dx), "dt": float(dt),
        "cfl_r": float(dt / dx),
        "n_incognitas": int(ref_nx * len(t)),
        "total_time_s": float(elapsed),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "L2_u_global_pct": rel_l2(U_grid, U_exact_grid),
        "L2_u_t0_pct": rel_l2(U_grid, U_exact_grid, m_t0),
        "L2_u_x0_pct": rel_l2(U_grid, U_exact_grid, m_x0),
        "L2_u_x1_pct": rel_l2(U_grid, U_exact_grid, m_x1),
        "L2_u_boundary_pct": rel_l2(U_grid, U_exact_grid, m_bnd),
        "L2_u_interior_pct": rel_l2(U_grid, U_exact_grid, m_int),
        "err_t_ratio_last_first_decile": float(
            np.abs(U_grid - U_exact_grid).mean(axis=0)[-n_eval // 10:].mean()
            / max(np.abs(U_grid - U_exact_grid).mean(axis=0)[:n_eval // 10].mean(), 1e-15)),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n--- malla de referencia (nx = {ref_nx}) ---")
    for k in ["L2_u_global_pct", "L2_u_t0_pct", "L2_u_boundary_pct",
              "L2_u_interior_pct", "err_t_ratio_last_first_decile",
              "total_time_s", "peak_rss_mb", "n_incognitas"]:
        print(f"  {k:32s} {metrics[k]}")
    print(f"\nSalidas en: {args.out}")
    print("Nota: la frontera se impone de forma DURA, por lo que su error es "
          "de redondeo; no es comparable con la imposicion blanda de las PINN.")


if __name__ == "__main__":
    main()
