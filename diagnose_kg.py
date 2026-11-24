#!/usr/bin/env python3
"""
diagnose_kg.py - Diagnostico fino sobre los eval.npz ya generados por run_kg.py.
No entrena nada: es post-hoc y corre en segundos.

    python diagnose_kg.py \
        --runs "qcpinn_cascade=results/kg_dv_angle_cascade_seed*" \
               "pinn_model2=results/kg_classical_seed*" \
        --out analysis/

Que agrega respecto de analyze_kg.py:
  1. Mapas de error PROMEDIADOS sobre las semillas (analyze_kg usaba dirs[0],
     una sola corrida, no representativa).
  2. Error contra t: verifica acumulacion temporal en la hiperbolica.
  3. Espectro DEL ERROR, no cociente de espectros. El cociente de amplitudes
     medias es ciego al error de fase: un modelo puede reproducir la amplitud
     del modo 5*pi y estar desfasado, dando ratio ~1 con error grande.
  4. Corte en x=1, donde u(t,1) = cos(5*pi*t) + t^3. Separa visualmente
     fallo de AMPLITUD (envolvente achatada) de fallo de FASE (desplazamiento).
  5. Descomposicion amplitud/fase del modo dominante por minimos cuadrados.
"""

import argparse
import glob
import json
import os

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
    """Devuelve (t, x, U_pred[n_seeds,nx,nt], U_exact[nx,nt]).
    Convencion de la malla: fila = x fijo, columna = t (verificado con run_kg.py)."""
    preds, exact, n = [], None, None
    for p in paths:
        z = np.load(p)
        n = int(round(np.sqrt(z["X"].shape[0])))
        preds.append(z["u_pred"].reshape(n, n))
        if exact is None:
            exact = z["u_star"].reshape(n, n)
    t = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, n)
    return t, x, np.stack(preds), exact


def fit_mode(signal, t, freq):
    """Ajusta signal ~ A*cos(2*pi*f*t + phi) + tendencia polinomica de grado 3.
    Devuelve (A, phi). Separa amplitud de fase para el modo dominante."""
    w = 2.0 * np.pi * freq
    M = np.column_stack([np.cos(w * t), np.sin(w * t),
                         np.ones_like(t), t, t ** 2, t ** 3])
    c, *_ = np.linalg.lstsq(M, signal, rcond=None)
    a, b = c[0], c[1]
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-x", type=float, default=0.5,
                help="posicion x del corte; 1.0 es frontera supervisada")
    ap.add_argument("--runs", nargs="+", required=True, help='pares "etiqueta=glob"')
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups, report = {}, {}
    for spec in args.runs:
        label, pattern = spec.split("=", 1)
        paths = find_evals(pattern)
        print(f"{label}: {len(paths)} eval.npz")
        if paths:
            groups[label] = load_group(paths)

    if not groups:
        print("Sin datos. Revisa los patrones.")
        return

    # ---------- 1. mapas de error promediados sobre semillas ----------
    vmax = 0.0
    for label, (t, x, P, E) in groups.items():
        vmax = max(vmax, float(np.abs(P - E[None]).mean(0).max()))
    for label, (t, x, P, E) in groups.items():
        Emap = np.abs(P - E[None]).mean(0)
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        im = ax.imshow(Emap, origin="lower", extent=[0, 1, 0, 1],
                       aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_xlabel("t"); ax.set_ylabel("x")
        ax.set_title(f"|error| medio sobre {P.shape[0]} semillas: {label}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"error_map_mean_{label}.png"), dpi=150)
        plt.close(fig)

    # ---------- 2. error contra t ----------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, (t, x, P, E) in groups.items():
        per_seed = np.abs(P - E[None]).mean(axis=1)      # (n_seeds, nt)
        mu, sd = per_seed.mean(0), per_seed.std(0)
        ax.plot(t, mu, label=label)
        ax.fill_between(t, np.maximum(mu - sd, 0), mu + sd, alpha=0.2)
        report.setdefault(label, {})["err_t_ratio_last_first_decile"] = float(
            mu[-len(t) // 10:].mean() / max(mu[:len(t) // 10].mean(), 1e-12))
    ax.set_xlabel("t"); ax.set_ylabel(r"$\langle|u_{pred}-u_{exact}|\rangle_x$")
    ax.set_title("Acumulacion temporal del error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_vs_t.png"), dpi=150)
    plt.close(fig)

    # ---------- 3. espectro DEL ERROR ----------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ref_done = False
    for label, (t, x, P, E) in groups.items():
        n = len(t)
        dt = 1.0 / (n - 1)
        freqs = np.fft.rfftfreq(n, d=dt)
        err = P - E[None]
        A_err = np.abs(np.fft.rfft(err, axis=2)).mean(axis=(0, 1))
        ax.semilogy(freqs, A_err, label=f"error: {label}")
        if not ref_done:
            A_ex = np.abs(np.fft.rfft(E, axis=1)).mean(0)
            ax.semilogy(freqs, A_ex, "k--", lw=0.9, label="solucion exacta")
            ref_done = True
        k = int(np.argmax(A_err[1:])) + 1
        report.setdefault(label, {})["error_peak_freq"] = float(freqs[k])
    ax.axvline(2.5, color="gray", lw=0.8, ls=":")
    ax.set_xlim(0, 12)
    ax.set_xlabel("frecuencia en t [ciclos por unidad]")
    ax.set_ylabel("amplitud media")
    ax.set_title(r"Espectro del error (linea punteada: modo $5\pi$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_spectrum.png"), dpi=150)
    plt.close(fig)

    # ---------- 4. corte en x = 1 : amplitud vs fase ----------
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    first = True
    for label, (t, x, P, E) in groups.items():
        i = int(round(args.slice_x * (len(x) - 1)))
        exact_slice = E[i, :]
        pred_slices = P[:, i, :]
        mu = pred_slices.mean(0)
        if first:
            axes[0].plot(t, exact_slice, "k--", lw=1.4, label="exacta")
            first = False
        axes[0].plot(t, mu, lw=1.0, label=label)
        axes[1].plot(t, np.abs(mu - exact_slice), lw=1.0, label=label)

        A_e, ph_e = fit_mode(exact_slice, t, 2.5)
        A_p, ph_p = fit_mode(mu, t, 2.5)
        dphi = float(np.arctan2(np.sin(ph_p - ph_e), np.cos(ph_p - ph_e)))
        report.setdefault(label, {}).update({
            "x1_amplitude_ratio": A_p / A_e if A_e else np.nan,
            "x1_phase_shift_rad": dphi,
            "x1_phase_shift_deg": np.degrees(dphi),
        })

    axes[0].set_ylabel("u(t, x=1)")
    axes[0].set_title(r"Corte en la frontera dura: $u(t,1)=\cos(5\pi t)+t^3$")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("|error|"); axes[1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "slice_x1.png"), dpi=150)
    plt.close(fig)

    print("\n--- diagnostico ---")
    for label, r in report.items():
        print(f"\n[{label}]")
        print(f"  razon de amplitud del modo 5pi en x=1 : {r['x1_amplitude_ratio']:.4f}"
              "   (<1 = envolvente achatada)")
        print(f"  desfase del modo 5pi en x=1           : {r['x1_phase_shift_deg']:+.2f} deg")
        print(f"  frecuencia dominante DEL ERROR        : {r['error_peak_freq']:.2f} ciclos/unidad")
        print(f"  error ultimo decil / primer decil en t: {r['err_t_ratio_last_first_decile']:.2f}"
              "   (>1 = acumulacion temporal)")

    with open(os.path.join(args.out, "diagnose_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nFiguras y reporte en: {args.out}")


if __name__ == "__main__":
    main()
