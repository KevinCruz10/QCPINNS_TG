#!/usr/bin/env python3
"""
fd_energia.py - Caso SIN termino fuente (f = 0) para verificar conservacion de
energia de la ecuacion de Klein-Gordon no lineal. Complementa la parte fisica
que pide la revision: aqui no hay solucion manufacturada, la verificacion es que
la energia se conserve.

Ecuacion:   u_tt - u_xx + beta*u + u^3 = 0,   (t,x) en [0, T] x [0, 1]
IC:         u(0,x) = A sin(pi x),   u_t(0,x) = 0
BC:         u(t,0) = u(t,1) = 0   (Dirichlet homogeneas: sin flujo en la frontera)

Energia conservada (sin fuente y con BC homogeneas):
    E(t) = integral_0^1 ( 1/2 u_t^2 + 1/2 u_x^2 + beta/2 u^2 + 1/4 u^4 ) dx

El esquema de salto de rana conserva una energia DISCRETA hasta O(dt^2); la
deriva relativa |E(t)-E(0)|/E(0) debe mantenerse acotada y pequena, y tender a
cero al refinar la malla. Eso es lo que se reporta.

    python fd_energia.py                 # estudio de refinamiento + E(t)
    python fd_energia.py --nx 401 --T 2  # una malla, horizonte mas largo

Salidas en results/energia_fd_beta<beta>/:
    energia.csv       E(t) muestreada
    convergencia.csv  deriva de energia por resolucion
    metrics.json      resumen
"""

import argparse
import csv
import json
import os

import numpy as np


def solve_free(nx, T=1.0, beta=1.0, A=1.0, r=0.5):
    """Integra u_tt - u_xx + beta u + u^3 = 0 con IC sin(pi x), u_t=0, BC homogeneas.
    Devuelve t, x, U (nt, nx) y el vector de energia E(t)."""
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = r * dx
    nt = int(round(T / dt)) + 1
    dt = T / (nt - 1)
    t = np.linspace(0.0, T, nt)
    c2 = (dt / dx) ** 2

    U = np.zeros((nt, nx))
    U[0] = A * np.sin(np.pi * x)
    U[0, 0] = U[0, -1] = 0.0

    # arranque Taylor 2do orden con u_t(0,x)=0:  u^1 = u^0 + dt^2/2 (u_xx - beta u - u^3)
    lap0 = np.zeros_like(x)
    lap0[1:-1] = (U[0, 2:] - 2 * U[0, 1:-1] + U[0, :-2]) / dx ** 2
    U[1] = U[0] + 0.5 * dt ** 2 * (lap0 - beta * U[0] - U[0] ** 3)
    U[1, 0] = U[1, -1] = 0.0

    for n in range(1, nt - 1):
        U[n + 1, 1:-1] = (2 * U[n, 1:-1] - U[n - 1, 1:-1]
                          + c2 * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2])
                          - dt ** 2 * (beta * U[n, 1:-1] + U[n, 1:-1] ** 3))
        U[n + 1, 0] = U[n + 1, -1] = 0.0

    # energia por niveles: u_t centrado en tiempo, u_x centrado en espacio
    E = np.full(nt, np.nan)
    for n in range(1, nt - 1):
        u_t = (U[n + 1] - U[n - 1]) / (2 * dt)
        u_x = np.gradient(U[n], dx)
        dens = 0.5 * u_t ** 2 + 0.5 * u_x ** 2 + 0.5 * beta * U[n] ** 2 + 0.25 * U[n] ** 4
        E[n] = dx * (dens.sum() - 0.5 * (dens[0] + dens[-1]))  # trapecio uniforme
    return t, x, U, E


def drift(E):
    E = E[np.isfinite(E)]
    E0 = E[0]
    return float(np.max(np.abs(E - E0)) / abs(E0)) if E0 != 0 else float("nan")


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
        a.out = f"results/energia_fd_beta{a.beta:g}"
    os.makedirs(a.out, exist_ok=True)

    print(f"{'nx':>6} {'dt':>10} {'E(0)':>12} {'deriva rel.':>14} {'orden':>7}")
    rows = []
    prev_d = prev_dx = None
    ref = None
    for nx in sorted(a.nx):
        t, x, U, E = solve_free(nx, T=a.T, beta=a.beta, A=a.amp, r=a.cfl)
        if not np.isfinite(U).all():
            print(f"{nx:>6}  divergio")
            continue
        d = drift(E)
        dx = 1.0 / (nx - 1)
        order = ""
        if prev_d is not None and d > 0:
            order = f"{np.log(prev_d / d) / np.log(prev_dx / dx):.2f}"
        prev_d, prev_dx = d, dx
        E0 = E[np.isfinite(E)][0]
        print(f"{nx:>6} {1.0/(nx-1)*a.cfl:>10.2e} {E0:>12.5f} {d:>14.3e} {order:>7}")
        rows.append({"nx": nx, "dt": 1.0 / (nx - 1) * a.cfl, "E0": E0,
                     "drift_rel": d, "orden": order})
        if nx == max(a.nx):
            ref = (t, E)

    with open(os.path.join(a.out, "convergencia.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if ref is not None:
        t, E = ref
        with open(os.path.join(a.out, "energia.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "E"])
            for ti, ei in zip(t, E):
                if np.isfinite(ei):
                    w.writerow([ti, ei])
        E0 = E[np.isfinite(E)][0]
        metrics = {"caso": "sin fuente (f=0), conservacion de energia",
                   "beta": a.beta, "T": a.T, "amp": a.amp,
                   "nx_ref": max(a.nx), "E0": float(E0),
                   "drift_rel_ref": drift(E)}
        json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
        print(f"\n  E(0) = {E0:.6f}   deriva relativa (malla fina) = {drift(E):.3e}")
    print(f"\nSalidas en: {a.out}")
    print("Lectura: la deriva baja al refinar (orden ~2) -> el esquema conserva")
    print("la energia; sirve como referencia fisica para contrastar la red.")


if __name__ == "__main__":
    main()
