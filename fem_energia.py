#!/usr/bin/env python3
"""
fem_energia.py - Caso SIN termino fuente (f=0), conservacion de energia por
elementos finitos P1. Companero de fd_energia.py (mismo problema, misma energia).

    u_tt - u_xx + beta u + u^3 = 0,  u(0,x)=A sin(pi x),  u_t(0,x)=0,  u(t,0)=u(t,1)=0

Energia:  E(t) = integral ( 1/2 u_t^2 + 1/2 u_x^2 + beta/2 u^2 + 1/4 u^4 ) dx

    python fem_energia.py
    python fem_energia.py --beta 0
"""
import argparse, csv, json, os
import numpy as np
from scipy.linalg import solve_banded


def assemble_matrices(nx, dx):
    M = np.zeros((3, nx)); K = np.zeros((3, nx))
    M[1, 0] = M[1, -1] = dx / 3.0; M[1, 1:-1] = 2.0 * dx / 3.0
    K[1, 0] = K[1, -1] = 1.0 / dx;  K[1, 1:-1] = 2.0 / dx
    M[0, 1:] = dx / 6.0; M[2, :-1] = dx / 6.0
    K[0, 1:] = -1.0 / dx; K[2, :-1] = -1.0 / dx
    return M, K


def matvec(A, v):
    out = A[1] * v
    out[:-1] += A[0, 1:] * v[1:]
    out[1:] += A[2, :-1] * v[:-1]
    return out


def nonlinear(u, x, dx):
    """vector N_i = integral(u^3 phi_i) por Gauss de 2 puntos."""
    n = len(x); N = np.zeros(n)
    gp = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]); gw = 0.5
    uL, uR = u[:-1], u[1:]
    for g in gp:
        uq = uL * (1 - g) + uR * g
        c = gw * dx * uq ** 3
        np.add.at(N, np.arange(n - 1), c * (1 - g))
        np.add.at(N, np.arange(1, n), c * g)
    return N


def solve_free(nx, T=1.0, beta=1.0, A=1.0, r=0.5):
    x = np.linspace(0.0, 1.0, nx); dx = x[1] - x[0]
    dt = r * dx; nt = int(round(T / dt)) + 1; dt = T / (nt - 1)
    t = np.linspace(0.0, T, nt)
    M, K = assemble_matrices(nx, dx)
    Mi = M[:, 1:-1].copy(); Mi[0, 1] = M[0, 2]; Mi[2, -2] = M[2, -3]

    U = np.zeros((nt, nx))
    U[0] = A * np.sin(np.pi * x); U[0, 0] = U[0, -1] = 0.0
    # arranque: M a0 = -K u0 - beta M u0 - N(u0),  u1 = u0 + dt^2/2 a0
    rhs0 = (-matvec(K, U[0]) - beta * matvec(M, U[0]) - nonlinear(U[0], x, dx))[1:-1]
    a0 = np.zeros(nx); a0[1:-1] = solve_banded((1, 1), Mi, rhs0)
    U[1] = U[0] + 0.5 * dt ** 2 * a0; U[1, 0] = U[1, -1] = 0.0

    for n in range(1, nt - 1):
        rhs = (-matvec(K, U[n]) - beta * matvec(M, U[n]) - nonlinear(U[n], x, dx)) * dt ** 2
        rhs += matvec(M, 2 * U[n] - U[n - 1])
        U[n + 1, 1:-1] = solve_banded((1, 1), Mi, rhs[1:-1])
        U[n + 1, 0] = U[n + 1, -1] = 0.0

    E = np.full(nt, np.nan)
    for n in range(1, nt - 1):
        u_t = (U[n + 1] - U[n - 1]) / (2 * dt)
        u_x = np.gradient(U[n], dx)
        dens = 0.5 * u_t ** 2 + 0.5 * u_x ** 2 + 0.5 * beta * U[n] ** 2 + 0.25 * U[n] ** 4
        E[n] = dx * (dens.sum() - 0.5 * (dens[0] + dens[-1]))
    return t, x, U, E


def drift(E):
    E = E[np.isfinite(E)]
    return float(np.max(np.abs(E - E[0])) / abs(E[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, nargs="+", default=[101, 201, 401, 801])
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"results/energia_fem_beta{a.beta:g}"
    os.makedirs(a.out, exist_ok=True)

    print(f"{'nx':>6} {'E(0)':>12} {'deriva rel.':>14} {'orden':>7}")
    rows = []; prev_d = prev_dx = None; ref = None
    for nx in sorted(a.nx):
        t, x, U, E = solve_free(nx, T=a.T, beta=a.beta, A=a.amp, r=a.cfl)
        if not np.isfinite(U).all():
            print(f"{nx:>6}  divergio"); continue
        d = drift(E); dx = 1.0 / (nx - 1)
        order = f"{np.log(prev_d / d) / np.log(prev_dx / dx):.2f}" if prev_d and d > 0 else ""
        prev_d, prev_dx = d, dx
        E0 = E[np.isfinite(E)][0]
        print(f"{nx:>6} {E0:>12.5f} {d:>14.3e} {order:>7}")
        rows.append({"nx": nx, "E0": E0, "drift_rel": d, "orden": order})
        if nx == max(a.nx):
            ref = (t, E)
    with open(os.path.join(a.out, "convergencia.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if ref:
        t, E = ref; E0 = E[np.isfinite(E)][0]
        with open(os.path.join(a.out, "energia.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "E"])
            [w.writerow([ti, ei]) for ti, ei in zip(t, E) if np.isfinite(ei)]
        json.dump({"caso": "sin fuente (f=0) FEM", "beta": a.beta, "T": a.T,
                   "E0": float(E0), "drift_rel_ref": drift(E), "nx_ref": max(a.nx)},
                  open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
        print(f"\n  E(0) = {E0:.6f}   deriva relativa (malla fina) = {drift(E):.3e}")
    print(f"\nSalidas en: {a.out}")


if __name__ == "__main__":
    main()
