#!/usr/bin/env python3
"""
run_kg_causal.py - Prueba causal de la acumulacion temporal del error.

Colocar en la raiz del repo QCPINN.

PREGUNTA
El error crece con t (razon ultimo/primer decil de 3.30 en QCPINN y 2.94 en
PINN). Dos hipotesis distintas producen ese mismo dato:

  H1 ACUMULACION      el error de tiempos tempranos se propaga y contamina los
                      tardios; la ventana final es mala PORQUE viene despues.
  H2 DIFICULTAD       la solucion es intrinsecamente mas dificil cerca de t=1;
                      seria igual de mala aunque se empezara ahi.

DISEÑO
Dos brazos, cada uno entrenado sobre media ventana temporal:

  Brazo A   t en [0.0, 0.5]   condiciones iniciales originales en t=0
  Brazo B   t en [0.5, 1.0]   condiciones iniciales EXACTAS en t=0.5

En el brazo B se le entrega al modelo un arranque perfecto, sin error
heredado. Entonces:

  error(B) ~ error(A)  ->  la ventana tardia no es mas dificil; parecia mala
                           solo por el error que le llegaba.  H1 CONFIRMADA.
  error(B) >> error(A) ->  la ventana tardia si es intrinsecamente dificil.
                           H1 DESCARTADA.

Condiciones iniciales exactas en t0, derivadas de u(t,x)=x*cos(5*pi*t)+(t*x)^3:
    u(t0,x)   = x*cos(5*pi*t0) + t0^3 * x^3
    u_t(t0,x) = -5*pi*x*sin(5*pi*t0) + 3*t0^2 * x^3
Para t0=0.5:  u = (0.5x)^3  y  u_t = -5*pi*x + 0.75*x^3  (velocidad no nula).

SOBRE EL SESGO DE DENSIDAD
Al reducir el dominio a la mitad manteniendo el mismo lote, la densidad de
puntos de colocacion se duplica. Eso favorece a AMBOS brazos por igual, de modo
que la comparacion A frente a B -que es la decisiva- no se ve afectada. Solo la
comparacion contra el modelo de dominio completo queda sesgada, y se reporta
como referencia, no como conclusion. Con --match-density se reduce el lote a la
mitad para eliminar tambien ese sesgo.

USO
    # brazos (3 semillas cada uno)
    python run_kg_causal.py --solver DV --arm A --seed 1
    python run_kg_causal.py --solver DV --arm B --seed 1

    # referencia gratuita: ventanas del modelo de dominio completo ya entrenado
    python run_kg_causal.py --compare-full "results/kg_dv_angle_cascade_seed*"
"""

import argparse
import csv
import glob
import json
import os
import random
import resource
import time

import numpy as np
import torch

from src.utils.logger import Logging
from src.nn.DVPDESolver import DVPDESolver
from src.nn.ClassicalSolver import ClassicalSolver
from src.nn.ClassicalSolver2 import ClassicalSolver2
from src.nn.pde import klein_gordon_operator

ARMS = {"A": (0.0, 0.5), "B": (0.5, 1.0)}


# ------------------------------------------------------ solucion de referencia

def u_exact(t, x):
    return x * torch.cos(5.0 * np.pi * t) + (t * x) ** 3


def ut_exact(t, x):
    return -5.0 * np.pi * x * torch.sin(5.0 * np.pi * t) + 3.0 * t ** 2 * x ** 3


def source(t, x):
    """f = u_tt - u_xx + u^3, identico al de src/data/klein_gordon_dataset.py."""
    u_tt = -25.0 * np.pi ** 2 * x * torch.cos(5.0 * np.pi * t) + 6.0 * t * x ** 3
    u_xx = 6.0 * t ** 3 * x
    u3 = (x * torch.cos(5.0 * np.pi * t) + (t * x) ** 3) ** 3
    return u_tt - u_xx + u3


def sample_batch(n, t0, t1, device, kind):
    """kind: 'res' | 'bc0' | 'bc1' | 'ic'. Columna 0 = t, columna 1 = x."""
    if kind == "res":
        t = torch.rand(n, 1, device=device) * (t1 - t0) + t0
        x = torch.rand(n, 1, device=device)
        return torch.cat([t, x], 1), source(t, x)
    if kind == "bc0":
        t = torch.rand(n, 1, device=device) * (t1 - t0) + t0
        x = torch.zeros(n, 1, device=device)
        return torch.cat([t, x], 1), u_exact(t, x)
    if kind == "bc1":
        t = torch.rand(n, 1, device=device) * (t1 - t0) + t0
        x = torch.ones(n, 1, device=device)
        return torch.cat([t, x], 1), u_exact(t, x)
    if kind == "ic":
        t = torch.full((n, 1), float(t0), device=device)
        x = torch.rand(n, 1, device=device)
        return torch.cat([t, x], 1), u_exact(t, x), ut_exact(t, x)
    raise ValueError(kind)


def rel_l2(pred, exact):
    den = np.linalg.norm(exact)
    if den == 0.0:
        return float(np.linalg.norm(pred - exact) * 100.0)
    return float(np.linalg.norm(pred - exact) / den * 100.0)


# ------------------------------------------- modo referencia (sin entrenar)

def compare_full(pattern, split=0.5):
    """Errores por ventana del modelo de dominio completo, desde eval.npz."""
    paths = []
    for base in glob.glob(pattern):
        paths += glob.glob(os.path.join(base, "**", "eval.npz"), recursive=True)
    paths = sorted(set(paths))
    if not paths:
        print(f"Sin eval.npz bajo '{pattern}'.")
        return
    early, late = [], []
    for p in paths:
        z = np.load(p)
        n = int(round(np.sqrt(z["X"].shape[0])))
        U = z["u_pred"].reshape(n, n)      # fila = x, columna = t
        E = z["u_star"].reshape(n, n)
        t = np.linspace(0.0, 1.0, n)
        me, ml = t <= split, t >= split
        early.append(rel_l2(U[:, me], E[:, me]))
        late.append(rel_l2(U[:, ml], E[:, ml]))
    early, late = np.array(early), np.array(late)
    print(f"\n--- modelo de dominio completo ({len(paths)} corridas) ---")
    print(f"  L2_u en t <= {split}  : {early.mean():.3f} +- {early.std(ddof=1):.3f} %")
    print(f"  L2_u en t >= {split}  : {late.mean():.3f} +- {late.std(ddof=1):.3f} %")
    print(f"  razon tardia/temprana : {late.mean() / early.mean():.2f}")
    return {"early_mean": float(early.mean()), "early_std": float(early.std(ddof=1)),
            "late_mean": float(late.mean()), "late_std": float(late.std(ddof=1))}


# --------------------------------------------------------------- entrenamiento

def build_model(a, logger, device):
    args = {
        "batch_size": a.batch, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
        "print_every": a.log_every, "log_path": a.outdir, "input_dim": 2,
        "output_dim": 1, "num_qubits": a.num_qubits, "hidden_dim": 50,
        "num_quantum_layers": 1, "classic_network": [2, 50, 1],
        "q_ansatz": a.ansatz, "mode": "hybrid" if a.solver == "DV" else "classical",
        "activation": "tanh", "shots": None, "problem": "klein_gordon",
        "solver": a.solver, "device": str(device), "method": "None",
        "cutoff_dim": 20, "class": "CVNeuralNetwork1", "encoding": a.encoding,
    }
    cls = {"DV": DVPDESolver, "Classical": ClassicalSolver,
           "Classical2": ClassicalSolver2}[a.solver]
    return cls(args, logger, data=None, device=device)


@torch.enable_grad()
def evaluate_window(model, t0, t1, device, n_points, chunk):
    t = np.linspace(t0, t1, n_points, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")     # fila = x, columna = t
    X_star = torch.from_numpy(
        np.stack([Tg.ravel(), Xg.ravel()], 1)).to(torch.float32).to(device)
    u_star = u_exact(X_star[:, 0:1], X_star[:, 1:2]).detach().cpu().numpy()

    preds = []
    for i in range(0, X_star.shape[0], chunk):
        xb = X_star[i:i + chunk].clone()
        up, _ = klein_gordon_operator(model, xb[:, 0:1], xb[:, 1:2])
        preds.append(up.detach().cpu().numpy())
        del up, xb
    return X_star.detach().cpu().numpy(), u_star, np.vstack(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-full", default=None,
                    help="patron glob de corridas de dominio completo; solo reporta y sale")
    ap.add_argument("--split", type=float, default=0.5)
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--solver", choices=["DV", "Classical", "Classical2"], default="DV")
    ap.add_argument("--ansatz", default="cascade")
    ap.add_argument("--encoding", default="angle")
    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--w-res", type=float, default=0.1)
    ap.add_argument("--w-ut", type=float, default=0.1)
    ap.add_argument("--w-bc", type=float, default=10.0)
    ap.add_argument("--w-ics", type=float, default=10.0)
    ap.add_argument("--match-density", action="store_true",
                    help="reduce el lote a la mitad para conservar densidad de colocacion")
    ap.add_argument("--outdir", default="./results_causal")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-points", type=int, default=200)
    ap.add_argument("--eval-chunk", type=int, default=500)
    a = ap.parse_args()

    if a.compare_full:
        compare_full(a.compare_full, a.split)
        return

    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cpu")

    t0, t1 = ARMS[a.arm]
    if a.split != 0.5:
        t0, t1 = (0.0, a.split) if a.arm == "A" else (a.split, 1.0)
    batch = a.batch // 2 if a.match_density else a.batch
    nb = max(1, batch // 3)

    tag = (f"causal{a.arm}_{a.solver.lower()}"
           + (f"_{a.ansatz}" if a.solver == "DV" else "") + f"_seed{a.seed}")
    root = os.path.join(a.outdir, tag)
    os.makedirs(root, exist_ok=True)
    logger = Logging(root)
    run_dir = logger.get_output_dir()
    print(f"[causal] brazo {a.arm}: t en [{t0}, {t1}] | lote {batch} | {run_dir}", flush=True)

    model = build_model(a, logger, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[causal] parametros = {n_params}", flush=True)

    fcsv = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["epoch", "loss", "loss_res", "loss_bc0", "loss_bc1",
                "loss_ics", "loss_ut", "lr", "iter_time_s"])

    t_start = time.time()
    times = []
    for it in range(a.epochs + 1):
        tic = time.perf_counter()
        model.optimizer.zero_grad()

        X_res, f_res = sample_batch(batch, t0, t1, device, "res")
        X_b0, u_b0 = sample_batch(nb, t0, t1, device, "bc0")
        X_b1, u_b1 = sample_batch(nb, t0, t1, device, "bc1")
        X_ic, u_ic, ut_ic = sample_batch(nb, t0, t1, device, "ic")

        X_ic.requires_grad_(True)
        u_ic_pred = model.forward(X_ic)
        grad = torch.autograd.grad(u_ic_pred, X_ic,
                                   grad_outputs=torch.ones_like(u_ic_pred),
                                   create_graph=True)[0]
        _, residual = klein_gordon_operator(model, X_res[:, 0:1], X_res[:, 1:2])

        loss_res = model.loss_fn(residual, f_res)
        loss_bc0 = model.loss_fn(model.forward(X_b0), u_b0)
        loss_bc1 = model.loss_fn(model.forward(X_b1), u_b1)
        loss_ics = model.loss_fn(u_ic_pred, u_ic)
        loss_ut = model.loss_fn(grad[:, 0:1], ut_ic)

        loss = (a.w_res * loss_res + a.w_ut * loss_ut
                + a.w_bc * (loss_bc0 + loss_bc1) + a.w_ics * loss_ics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()
        model.scheduler.step(loss)
        model.loss_history.append(float(loss.item()))

        dt = time.perf_counter() - tic
        times.append(dt)
        if it % a.log_every == 0:
            w.writerow([it, float(loss.item()), float(loss_res.item()),
                        float(loss_bc0.item()), float(loss_bc1.item()),
                        float(loss_ics.item()), float(loss_ut.item()),
                        model.optimizer.param_groups[0]["lr"], dt])
            if it % (a.log_every * 50) == 0:
                fcsv.flush()
                print(f"[{it:6d}] loss={loss.item():.3e} res={loss_res.item():.3e} "
                      f"ic={loss_ics.item():.3e} ut={loss_ut.item():.3e} "
                      f"{dt*1000:.0f} ms/it", flush=True)
    fcsv.close()
    total = time.time() - t_start
    model.save_state()

    X, u_star, u_pred = evaluate_window(model, t0, t1, device,
                                        a.eval_points, a.eval_chunk)
    np.savez_compressed(os.path.join(run_dir, "eval.npz"),
                        X=X, u_star=u_star, u_pred=u_pred, t0=t0, t1=t1)

    n = a.eval_points
    err_t = np.abs(u_pred - u_star).reshape(n, n).mean(axis=0)
    metrics = {
        "arm": a.arm, "t0": t0, "t1": t1, "solver": a.solver,
        "ansatz": a.ansatz, "seed": a.seed, "epochs": a.epochs,
        "batch": batch, "match_density": bool(a.match_density),
        "n_params": int(n_params),
        "total_time_s": total,
        "mean_iter_time_s": float(np.mean(times)),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "loss_final": float(model.loss_history[-1]),
        "L2_u_window_pct": rel_l2(u_pred, u_star),
        "err_t_ratio_last_first_decile": float(
            err_t[-n // 10:].mean() / max(err_t[:n // 10].mean(), 1e-15)),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n  brazo {a.arm}  t en [{t0}, {t1}]")
    print(f"  L2_u en la ventana              : {metrics['L2_u_window_pct']:.4f} %")
    print(f"  acumulacion dentro de la ventana: {metrics['err_t_ratio_last_first_decile']:.2f}")
    print(f"  tiempo total                    : {total/60:.1f} min")
    print(f"[causal] listo -> {run_dir}")


if __name__ == "__main__":
    main()
