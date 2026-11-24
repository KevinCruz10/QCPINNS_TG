#!/usr/bin/env python3
"""
run_kg.py - Runner instrumentado para Klein-Gordon (QCPINN DV vs PINN clasica).

Colocar en la RAIZ del repo QCPINN (al lado de src/) y ejecutar:

    python run_kg.py --solver DV        --ansatz cascade --seed 1
    python run_kg.py --solver Classical --seed 1                    # PINN Model-2 (2751 par)
    python run_kg.py --solver Classical2 --seed 1                   # PINN Model-1 (7851 par)

Que arregla respecto de src/trainer/klein_gordon_trainer.py:
  1. device: el trainer original pasa DEVICE como argumento posicional `data`,
     dejando self.device=None. Aqui se pasa por keyword y se fuerza CPU.
  2. seed: args["seed"] nunca se usa en el repo. Aqui se siembra torch/numpy/random.
  3. timing: el repo mide el tiempo ANTES de loss.backward(), excluyendo la
     retropropagacion por el circuito (la parte cara). Aqui se mide la iteracion completa.
  4. batch_size: train() del repo ignora args["batch_size"] (default 128 hardcodeado).
     Aqui es un flag explicito.
  5. perdidas: el repo suma loss_ics dentro de loss_bc y no permite separar L_f de L_u.
     Aqui cada componente se registra cruda en metrics.csv.
  6. evaluacion: el repo evalua 200x200=40000 puntos en un solo forward con grafo de
     segundo orden. Aqui se evalua por chunks.
  7. diagnostico: varianza de gradientes de los parametros cuanticos (mesetas esteriles)
     y normas de gradiente por bloque (desbalance clasico/cuantico).
"""

import argparse
import csv
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
from src.data.klein_gordon_dataset import generate_training_dataset, u as u_exact_fn, f as f_exact_fn

# Parametros de la ecuacion, identicos a src/nn/pde.py::klein_gordon_operator
ALPHA, BETA, GAMMA, KPOW = -1.0, 0.0, 1.0, 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["DV", "Classical", "Classical2"], default="DV")
    p.add_argument("--ansatz", choices=["cascade", "cross_mesh", "alternate", "layered"],
                   default="cascade")
    p.add_argument("--encoding", choices=["angle", "amplitude"], default="angle")
    p.add_argument("--num-qubits", type=int, default=5)
    p.add_argument("--num-quantum-layers", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=50)

    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128, help="batch del residuo; BC/IC usan batch//3")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=1)

    # Pesos. Los defaults reproducen EXACTAMENTE el codigo del repo:
    #   loss = 0.1*(loss_r + loss_ut) + 10*(loss_bc1 + loss_bc2 + loss_ics)
    # Nota: difieren del apendice A.4 del paper (lambda1=1.0, lambda2=10.0, lambda3=1.0).
    p.add_argument("--w-res", type=float, default=0.1)
    p.add_argument("--w-ut", type=float, default=0.1)
    p.add_argument("--w-bc", type=float, default=10.0)
    p.add_argument("--w-ics", type=float, default=10.0)

    p.add_argument("--outdir", default="./results")
    p.add_argument("--tag", default=None, help="nombre de la corrida; por defecto se autogenera")
    p.add_argument("--threads", type=int, default=2, help="hilos de torch por proceso")
    p.add_argument("--log-every", type=int, default=10, help="cada cuantas epocas escribir fila CSV")
    p.add_argument("--grad-every", type=int, default=100, help="cada cuantas epocas medir gradientes")
    p.add_argument("--ckpt-every", type=int, default=2000, help="cada cuantas epocas guardar model.pth")
    p.add_argument("--eval-points", type=int, default=200, help="malla de evaluacion N x N")
    p.add_argument("--eval-chunk", type=int, default=2000, help="puntos por chunk en evaluacion")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(a, logger, device):
    classic_network = [2, a.hidden_dim, 1]
    args = {
        "batch_size": a.batch,
        "epochs": a.epochs,
        "lr": a.lr,
        "seed": a.seed,
        "print_every": a.log_every,
        "log_path": a.outdir,
        "input_dim": 2,
        "output_dim": 1,
        "num_qubits": a.num_qubits,
        "hidden_dim": a.hidden_dim,
        "num_quantum_layers": a.num_quantum_layers,
        "classic_network": classic_network,
        "q_ansatz": a.ansatz,
        "mode": "hybrid" if a.solver == "DV" else "classical",
        "activation": "tanh",
        "shots": None,                 # gradientes analiticos por backprop
        "problem": "klein_gordon",
        "solver": a.solver,
        "device": str(device),         # str, no torch.device: evita problemas con torch.load
        "method": "None",
        "cutoff_dim": 20,
        "class": "CVNeuralNetwork1",
        "encoding": a.encoding,
    }
    if a.solver == "DV":
        model = DVPDESolver(args, logger, data=None, device=device)
    elif a.solver == "Classical":
        model = ClassicalSolver(args, logger, data=None, device=device)
    else:
        model = ClassicalSolver2(args, logger, data=None, device=device)
    return model, args


def grad_diagnostics(model, solver):
    """Varianza de gradientes cuanticos (meseta esteril) y normas por bloque."""
    out = {"grad_var_q": "", "grad_norm_q": "", "grad_norm_pre": "", "grad_norm_post": ""}

    def block_norm(module):
        tot = 0.0
        for p in module.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        return tot ** 0.5

    out["grad_norm_pre"] = block_norm(model.preprocessor)
    out["grad_norm_post"] = block_norm(model.postprocessor)
    if solver == "DV":
        g = model.quantum_layer.params.grad
        if g is not None:
            g = g.detach().flatten()
            out["grad_var_q"] = float(g.var(unbiased=False))
            out["grad_norm_q"] = float(g.norm())
    return out


@torch.enable_grad()
def evaluate(model, device, n_points, chunk):
    """Evalua u y el residuo f en malla n x n por chunks (grafo de 2do orden = caro en RAM)."""
    t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    T, X = np.meshgrid(t, x)
    X_star = torch.from_numpy(
        np.hstack([T.flatten()[:, None], X.flatten()[:, None]])
    ).to(torch.float32).to(device)

    alpha = torch.tensor(ALPHA, device=device)
    beta = torch.tensor(BETA, device=device)
    gamma = torch.tensor(GAMMA, device=device)

    u_star = u_exact_fn(X_star).detach().cpu().numpy()
    f_star = f_exact_fn(X_star, alpha, beta, gamma, KPOW).detach().cpu().numpy()

    u_pred, f_pred = [], []
    for i in range(0, X_star.shape[0], chunk):
        xb = X_star[i:i + chunk].clone()
        up, fp = klein_gordon_operator(model, xb[:, 0:1], xb[:, 1:2])
        u_pred.append(up.detach().cpu().numpy())
        f_pred.append(fp.detach().cpu().numpy())
        del up, fp, xb
    u_pred = np.vstack(u_pred)
    f_pred = np.vstack(f_pred)
    return X_star.detach().cpu().numpy(), u_star, u_pred, f_star, f_pred


def rel_l2(pred, exact, mask=None):
    if mask is not None:
        pred, exact = pred[mask], exact[mask]
    den = np.linalg.norm(exact)
    if den == 0.0:
        return float(np.linalg.norm(pred - exact) * 100.0)
    return float(np.linalg.norm(pred - exact) / den * 100.0)


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    device = torch.device("cpu")   # ver seccion 4.2 de la guia: CPU a proposito

    tag = a.tag or (
        f"kg_{a.solver.lower()}"
        + (f"_{a.encoding}_{a.ansatz}" if a.solver == "DV" else "")
        + f"_seed{a.seed}"
    )
    root = os.path.join(a.outdir, tag)
    os.makedirs(root, exist_ok=True)

    logger = Logging(root)                       # crea root/<timestamp>/output.log
    run_dir = logger.get_output_dir()
    print(f"[run_kg] run_dir = {run_dir}", flush=True)

    model, args = build_model(a, logger, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.print(f"solver={a.solver} ansatz={a.ansatz} encoding={a.encoding} seed={a.seed}")
    logger.print(f"Total number of parameters: {n_params}")
    print(f"[run_kg] parametros entrenables = {n_params}", flush=True)

    ics_sampler, bcs_sampler, res_sampler = generate_training_dataset(device)

    csv_path = os.path.join(run_dir, "metrics.csv")
    fcsv = open(csv_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow([
        "epoch", "loss", "loss_res", "loss_bc1", "loss_bc2", "loss_ics", "loss_ut",
        "L_f_raw", "L_u_raw", "lr", "iter_time_s",
        "grad_var_q", "grad_norm_q", "grad_norm_pre", "grad_norm_post",
    ])

    t_start = time.time()
    iter_times = []

    for it in range(a.epochs + 1):
        t0 = time.perf_counter()
        model.optimizer.zero_grad()

        nb = max(1, a.batch // 3)
        X_ics, u_ics = ics_sampler.sample(nb)
        X_bc1, u_bc1 = bcs_sampler[0].sample(nb)
        X_bc2, u_bc2 = bcs_sampler[1].sample(nb)
        X_res, f_res = res_sampler.sample(a.batch)

        X_ics.requires_grad_(True)
        u_bc1_pred = model.forward(X_bc1)
        u_bc2_pred = model.forward(X_bc2)
        u_ics_pred = model.forward(X_ics)

        u_grad = torch.autograd.grad(
            u_ics_pred, X_ics,
            grad_outputs=torch.ones_like(u_ics_pred),
            create_graph=True,
        )[0]

        _, residual = klein_gordon_operator(model, X_res[:, 0:1], X_res[:, 1:2])

        loss_res = model.loss_fn(residual, f_res)
        loss_bc1 = model.loss_fn(u_bc1_pred, u_bc1)
        loss_bc2 = model.loss_fn(u_bc2_pred, u_bc2)
        loss_ics = model.loss_fn(u_ics_pred, u_ics)
        loss_ut = model.loss_fn(u_grad[:, 0], torch.zeros_like(u_grad[:, 0]))

        loss = (a.w_res * loss_res + a.w_ut * loss_ut
                + a.w_bc * (loss_bc1 + loss_bc2) + a.w_ics * loss_ics)

        loss.backward()

        gd = {"grad_var_q": "", "grad_norm_q": "", "grad_norm_pre": "", "grad_norm_post": ""}
        if it % a.grad_every == 0:
            gd = grad_diagnostics(model, a.solver)

        clip = 0.1 if a.solver == "CV" else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        model.optimizer.step()
        model.scheduler.step(loss)
        model.loss_history.append(float(loss.item()))

        dt = time.perf_counter() - t0
        iter_times.append(dt)

        if it % a.log_every == 0:
            L_f_raw = float(loss_res.item())
            L_u_raw = float(loss_bc1.item() + loss_bc2.item() + loss_ics.item())
            writer.writerow([
                it, float(loss.item()), L_f_raw, float(loss_bc1.item()),
                float(loss_bc2.item()), float(loss_ics.item()), float(loss_ut.item()),
                L_f_raw, L_u_raw,
                model.optimizer.param_groups[0]["lr"], dt,
                gd["grad_var_q"], gd["grad_norm_q"], gd["grad_norm_pre"], gd["grad_norm_post"],
            ])
            if it % (a.log_every * 20) == 0:
                fcsv.flush()
                print(f"[{it:6d}] loss={loss.item():.3e} L_f={L_f_raw:.3e} "
                      f"L_u={L_u_raw:.3e} L_ut={loss_ut.item():.3e} "
                      f"{dt*1000:.0f} ms/it", flush=True)

        if a.ckpt_every and it % a.ckpt_every == 0 and it > 0:
            model.save_state()

    fcsv.close()
    total_time = time.time() - t_start
    model.save_state()

    X, u_star, u_pred, f_star, f_pred = evaluate(model, device, a.eval_points, a.eval_chunk)
    np.savez_compressed(os.path.join(run_dir, "eval.npz"),
                        X=X, u_star=u_star, u_pred=u_pred, f_star=f_star, f_pred=f_pred)

    m_t0 = X[:, 0] == 0.0                       # condicion inicial
    m_x0 = X[:, 1] == 0.0                       # frontera izquierda
    m_x1 = X[:, 1] == 1.0                       # frontera derecha
    m_bnd = m_t0 | m_x0 | m_x1
    m_int = ~m_bnd

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    metrics = {
        "tag": tag,
        "run_dir": run_dir,
        "solver": a.solver, "ansatz": a.ansatz, "encoding": a.encoding,
        "seed": a.seed, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
        "weights": {"res": a.w_res, "ut": a.w_ut, "bc": a.w_bc, "ics": a.w_ics},
        "n_params": int(n_params),
        "total_time_s": total_time,
        "mean_iter_time_s": float(np.mean(iter_times)),
        "median_iter_time_s": float(np.median(iter_times)),
        "peak_rss_mb": peak_mb,
        "loss_final": float(model.loss_history[-1]),
        "L2_u_global_pct": rel_l2(u_pred, u_star),
        "L2_f_global_pct": rel_l2(f_pred, f_star),
        "L2_u_t0_pct": rel_l2(u_pred, u_star, m_t0),
        "L2_u_x0_pct": rel_l2(u_pred, u_star, m_x0),
        "L2_u_x1_pct": rel_l2(u_pred, u_star, m_x1),
        "L2_u_boundary_pct": rel_l2(u_pred, u_star, m_bnd),
        "L2_u_interior_pct": rel_l2(u_pred, u_star, m_int),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    for k in ["n_params", "total_time_s", "mean_iter_time_s", "peak_rss_mb",
              "loss_final", "L2_u_global_pct", "L2_f_global_pct",
              "L2_u_t0_pct", "L2_u_boundary_pct", "L2_u_interior_pct"]:
        print(f"  {k:24s} {metrics[k]}")
        logger.print(f"{k}: {metrics[k]}")
    print(f"[run_kg] listo -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
