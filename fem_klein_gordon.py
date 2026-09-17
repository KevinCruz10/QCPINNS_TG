#!/usr/bin/env python3
"""
fem_klein_gordon.py - Metodo de elementos finitos (Galerkin P1) para la
ecuacion de Klein-Gordon no lineal 1D. Segundo metodo numerico clasico de
referencia, complementario a fd_klein_gordon.py.

    python fem_klein_gordon.py                 # estudio de convergencia
    python fem_klein_gordon.py --nx 201        # una sola malla
    python fem_klein_gordon.py --lumped        # matriz de masa concentrada

Problema (identico al de src/data/klein_gordon_dataset.py):

    u_tt - u_xx + u^3 = f(t,x)      (t,x) en [0,1] x [0,1]
    u(t,0) = 0,   u(t,1) = cos(5*pi*t) + t^3
    u(0,x) = x,   u_t(0,x) = 0

Formulacion debil: hallar u en H^1 tal que para toda funcion de prueba v que
se anula en la frontera,

    (u_tt, v) + (u_x, v_x) + (u^3, v) = (f, v)

Discretizacion:
  - Espacio: elementos lineales P1 sobre malla uniforme. Matriz de masa M
    consistente (tridiagonal) o concentrada; matriz de rigidez K tridiagonal.
  - No linealidad y termino fuente: cuadratura de Gauss de 2 puntos por
    elemento, exacta hasta grado 3.
  - Tiempo: diferencias centradas (salto de rana),
        M (u^{n+1} - 2u^n + u^{n-1}) / dt^2 = -K u^n - N(u^n) + F^n
    Con masa consistente cada paso resuelve un sistema tridiagonal
    (factorizacion LU una sola vez, reutilizada en todos los pasos).

Diferencia conceptual con las PINN: FEM minimiza el residuo en forma DEBIL,
promediado contra funciones de prueba; las PINN lo minimizan en forma FUERTE
y puntual sobre puntos de colocacion muestreados.

Salidas en results/fem_reference/:
    metrics.json, convergence.csv, eval_fem.npz
"""

import argparse
import csv
import json
import os
import resource
import time

import numpy as np
from scipy.linalg import solve_banded


# ----------------------------------------------------------------- problema

def u_exact(T, X):
    return X * np.cos(5.0 * np.pi * T) + (T * X) ** 3


def source(t, x):
    """f = u_tt - u_xx + u^3 evaluado analiticamente."""
    x = np.asarray(x, dtype=float)
    u_tt = -25.0 * np.pi ** 2 * x * np.cos(5.0 * np.pi * t) + 6.0 * t * x ** 3
    u_xx = 6.0 * t ** 3 * x
    u3 = (x * np.cos(5.0 * np.pi * t) + (t * x) ** 3) ** 3
    return u_tt - u_xx + u3


def g_right(t):
    return np.cos(5.0 * np.pi * t) + t ** 3


# ------------------------------------------------------- ensamblado P1

def assemble_matrices(nx, dx, lumped=False):
    """Matrices de masa y rigidez P1 en formato banda (3, nx) para solve_banded."""
    M = np.zeros((3, nx))   # filas: superdiagonal, diagonal, subdiagonal
    K = np.zeros((3, nx))

    # diagonal
    M[1, 0] = dx / 3.0
    M[1, -1] = dx / 3.0
    M[1, 1:-1] = 2.0 * dx / 3.0
    K[1, 0] = 1.0 / dx
    K[1, -1] = 1.0 / dx
    K[1, 1:-1] = 2.0 / dx
    # fuera de la diagonal
    M[0, 1:] = dx / 6.0
    M[2, :-1] = dx / 6.0
    K[0, 1:] = -1.0 / dx
    K[2, :-1] = -1.0 / dx

    if lumped:
        M[0, :] = 0.0
        M[2, :] = 0.0
        M[1, 0] = dx / 2.0
        M[1, -1] = dx / 2.0
        M[1, 1:-1] = dx
    return M, K


def banded_matvec(A, v):
    """Producto matriz-vector para matriz tridiagonal en formato banda."""
    out = A[1] * v
    out[:-1] += A[0, 1:] * v[1:]
    out[1:] += A[2, :-1] * v[:-1]
    return out


# Cuadratura de Gauss de 2 puntos en [0,1]
_GP = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
_GW = np.array([0.5, 0.5])


def assemble_nonlinear_and_source(u, x, dx, t):
    """Vectores N_i = integral(u^3 * phi_i) y F_i = integral(f * phi_i),
    ensamblados elemento por elemento con Gauss de 2 puntos."""
    n = len(x)
    N = np.zeros(n)
    F = np.zeros(n)
    uL, uR = u[:-1], u[1:]
    xL = x[:-1]
    for gp, gw in zip(_GP, _GW):
        phiL, phiR = 1.0 - gp, gp
        u_q = uL * phiL + uR * phiR
        x_q = xL + gp * dx
        f_q = source(t, x_q)
        w = gw * dx
        cu = w * u_q ** 3
        cf = w * f_q
        np.add.at(N, np.arange(n - 1), cu * phiL)
        np.add.at(N, np.arange(1, n), cu * phiR)
        np.add.at(F, np.arange(n - 1), cf * phiL)
        np.add.at(F, np.arange(1, n), cf * phiR)
    return N, F


# ------------------------------------------------------------------ solver

def solve_fem(nx, r=0.5, lumped=False):
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = r * dx
    nt = int(round(1.0 / dt)) + 1
    dt = 1.0 / (nt - 1)
    t = np.linspace(0.0, 1.0, nt)

    M, K = assemble_matrices(nx, dx, lumped=lumped)

    # Sistema reducido a nodos interiores (Dirichlet fuerte en 0 y nx-1)
    Mi = M[:, 1:-1].copy()
    Mi[0, 1] = M[0, 2]          # ajuste de bandas al recortar
    Mi[2, -2] = M[2, -3]

    t0 = time.perf_counter()

    U = np.zeros((nt, nx))
    U[0] = x.copy()

    # Arranque de segundo orden con u_t(0,x) = 0:
    #   u^1 = u^0 + dt^2/2 * a^0,  con M a^0 = -K u^0 - N(u^0) + F^0
    N0, F0 = assemble_nonlinear_and_source(U[0], x, dx, t[0])
    rhs0 = (-banded_matvec(K, U[0]) - N0 + F0)[1:-1]
    a0 = np.zeros(nx)
    a0[1:-1] = solve_banded((1, 1), Mi, rhs0)
    U[1] = U[0] + 0.5 * dt ** 2 * a0
    U[1, 0] = 0.0
    U[1, -1] = g_right(t[1])

    for n in range(1, nt - 1):
        Nn, Fn = assemble_nonlinear_and_source(U[n], x, dx, t[n])
        rhs = (-banded_matvec(K, U[n]) - Nn + Fn) * dt ** 2
        rhs += banded_matvec(M, 2.0 * U[n] - U[n - 1])

        # traslada al lado derecho la contribucion de los nodos de frontera
        bl = 0.0
        br = g_right(t[n + 1])
        rhs_i = rhs[1:-1].copy()
        rhs_i[0] -= M[0, 1] * bl
        rhs_i[-1] -= M[2, -2] * br

        U[n + 1, 1:-1] = solve_banded((1, 1), Mi, rhs_i)
        U[n + 1, 0] = bl
        U[n + 1, -1] = br

    elapsed = time.perf_counter() - t0
    return t, x, U, elapsed, dt, dx


def to_eval_grid(t, x, U, n_eval=200):
    from scipy.interpolate import RegularGridInterpolator
    te = np.linspace(0.0, 1.0, n_eval)
    xe = np.linspace(0.0, 1.0, n_eval)
    interp = RegularGridInterpolator((t, x), U, method="linear",
                                     bounds_error=False, fill_value=None)
    Xg, Tg = np.meshgrid(xe, te, indexing="ij")
    return te, xe, interp(np.stack([Tg.ravel(), Xg.ravel()], 1)).reshape(n_eval, n_eval)


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
    ap.add_argument("--nx", type=int, nargs="+", default=[51, 101, 201, 401])
    ap.add_argument("--ref-nx", type=int, default=None)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--lumped", action="store_true", help="matriz de masa concentrada")
    ap.add_argument("--eval-points", type=int, default=200)
    ap.add_argument("--out", default="results/fem_reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref_nx = args.ref_nx or max(args.nx)
    n_eval = args.eval_points
    te = np.linspace(0.0, 1.0, n_eval)
    xe = np.linspace(0.0, 1.0, n_eval)
    Xg, Tg = np.meshgrid(xe, te, indexing="ij")
    U_exact_grid = u_exact(Tg, Xg)

    rows, ref_payload = [], None
    print(f"{'nx':>6} {'nt':>7} {'dx':>10} {'dt':>10} {'L2_u (%)':>10} "
          f"{'orden':>7} {'tiempo (s)':>11}")
    prev_err = prev_dx = None

    for nx in sorted(args.nx):
        t, x, U, elapsed, dt, dx = solve_fem(nx, r=args.cfl, lumped=args.lumped)
        if not np.isfinite(U).all():
            print(f"{nx:>6} {'--':>7}  divergio")
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
            ref_payload = (t, x, U_grid, elapsed, dt, dx)

    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if ref_payload is None:
        print("\nNo se genero malla de referencia; revisa --ref-nx.")
        return

    t, x, U_grid, elapsed, dt, dx = ref_payload
    np.savez_compressed(os.path.join(args.out, "eval_fem.npz"),
                        t=te, x=xe, u_fem=U_grid, u_exact=U_exact_grid)

    m_t0 = np.zeros_like(U_grid, dtype=bool); m_t0[:, 0] = True
    m_bnd = m_t0.copy(); m_bnd[0, :] = True; m_bnd[-1, :] = True
    m_int = ~m_bnd
    err_t = np.abs(U_grid - U_exact_grid).mean(axis=0)

    metrics = {
        "metodo": "elementos finitos Galerkin P1"
                  + (" (masa concentrada)" if args.lumped else " (masa consistente)"),
        "nx": int(ref_nx), "nt": int(len(t)), "dx": float(dx), "dt": float(dt),
        "n_incognitas": int(ref_nx * len(t)),
        "total_time_s": float(elapsed),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "L2_u_global_pct": rel_l2(U_grid, U_exact_grid),
        "L2_u_t0_pct": rel_l2(U_grid, U_exact_grid, m_t0),
        "L2_u_boundary_pct": rel_l2(U_grid, U_exact_grid, m_bnd),
        "L2_u_interior_pct": rel_l2(U_grid, U_exact_grid, m_int),
        "err_t_ratio_last_first_decile": float(
            err_t[-n_eval // 10:].mean() / max(err_t[:n_eval // 10].mean(), 1e-15)),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n--- malla de referencia (nx = {ref_nx}) ---")
    for k in ["L2_u_global_pct", "L2_u_interior_pct",
              "err_t_ratio_last_first_decile", "total_time_s",
              "peak_rss_mb", "n_incognitas"]:
        print(f"  {k:32s} {metrics[k]}")
    print(f"\nSalidas en: {args.out}")
    print("Nota: como en diferencias finitas, la frontera se impone de forma "
          "DURA; su error no es comparable con la imposicion blanda de las PINN.")


if __name__ == "__main__":
    main()
