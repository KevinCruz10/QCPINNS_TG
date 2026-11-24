#!/usr/bin/env python3
"""
run_energia.py - Caso SIN termino fuente (f=0) para las REDES (PINN clasica y
QCPINN hibrida). Entrena la vibracion libre y mide cuanto se le deriva la energia
al modelo. Companero de fd_energia.py / fem_energia.py (mismo problema).

    u_tt - u_xx + beta u + u^3 = 0,  u(0,x)=A sin(pi x),  u_t(0,x)=0,  u(t,0)=u(t,1)=0

No hay solucion exacta (termino u^3). La metrica es la deriva relativa de
    E(t) = integral ( 1/2 u_t^2 + 1/2 u_x^2 + beta/2 u^2 + 1/4 u^4 ) dx
que la ecuacion conserva. Los metodos numericos la conservan a ~1e-5; aqui se
mide cuanto la conserva la red, para contrastar.

    python run_energia.py --solver Classical --seed 1 --tag energia_cls_seed1
    python run_energia.py --solver DV --seed 1 --tag energia_dv_seed1
"""
import argparse, csv, json, os, random, time
import numpy as np
import torch

from src.utils.logger import Logging
from src.nn.ClassicalSolver import ClassicalSolver
from pde_mass import klein_gordon_operator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["DV", "Classical"], default="Classical")
    p.add_argument("--ansatz", default="cascade")
    p.add_argument("--encoding", default="angle")
    p.add_argument("--num-qubits", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--w-res", type=float, default=0.1)
    p.add_argument("--w-ut", type=float, default=0.1)
    p.add_argument("--w-bc", type=float, default=10.0)
    p.add_argument("--w-ics", type=float, default=10.0)
    p.add_argument("--outdir", default="./results")
    p.add_argument("--tag", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lbfgs-iters", type=int, default=0,
                   help="iteraciones de L-BFGS tras Adam (0 = solo Adam)")
    p.add_argument("--lbfgs-nres", type=int, default=800)
    p.add_argument("--lbfgs-nb", type=int, default=200)
    p.add_argument("--energy-nt", type=int, default=100)
    p.add_argument("--energy-nx", type=int, default=200)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(a, logger, device):
    args = {"batch_size": a.batch, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
            "print_every": a.log_every, "log_path": a.outdir, "input_dim": 2,
            "output_dim": 1, "num_qubits": a.num_qubits, "hidden_dim": a.hidden_dim,
            "num_quantum_layers": 1, "classic_network": [2, a.hidden_dim, 1],
            "q_ansatz": a.ansatz, "mode": "hybrid" if a.solver == "DV" else "classical",
            "activation": "tanh", "shots": None, "problem": "klein_gordon",
            "solver": a.solver, "device": str(device), "method": "None",
            "cutoff_dim": 20, "class": "CVNeuralNetwork1", "encoding": a.encoding}
    if a.solver == "DV":
        from src.nn.DVPDESolver import DVPDESolver     # pennylane solo si hace falta
        return DVPDESolver(args, logger, data=None, device=device)
    return ClassicalSolver(args, logger, data=None, device=device)


def sample(n, t0, t1, device, kind, A):
    """kind: res | bc | ic. Columna 0 = t, columna 1 = x."""
    if kind == "res":
        t = torch.rand(n, 1, device=device) * (t1 - t0) + t0
        x = torch.rand(n, 1, device=device)
        return torch.cat([t, x], 1)
    if kind == "bc":            # x in {0,1}
        t = torch.rand(n, 1, device=device) * (t1 - t0) + t0
        x = (torch.rand(n, 1, device=device) > 0.5).float()
        return torch.cat([t, x], 1)
    if kind == "ic":            # t=0
        x = torch.rand(n, 1, device=device)
        t = torch.zeros(n, 1, device=device)
        u0 = A * torch.sin(np.pi * x)
        return torch.cat([t, x], 1), u0


@torch.enable_grad()
def energy_curve(model, device, beta, A, nt, nx, T):
    """E(t) sobre una malla; u_t y u_x por autodiff. Devuelve t, E (nt,)."""
    ts = torch.linspace(0.0, T, nt, device=device)
    xs = torch.linspace(0.0, 1.0, nx, device=device)
    E = np.zeros(nt)
    for i, tv in enumerate(ts):
        t = torch.full((nx, 1), float(tv), device=device, requires_grad=True)
        x = xs.view(-1, 1).clone().requires_grad_(True)
        u = model.forward(torch.cat([t, x], 1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=False, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=False)[0]
        dens = (0.5 * u_t ** 2 + 0.5 * u_x ** 2
                + 0.5 * beta * u ** 2 + 0.25 * u ** 4).detach().cpu().numpy().ravel()
        xg = xs.cpu().numpy(); dx = xg[1] - xg[0]
        E[i] = dx * (dens.sum() - 0.5 * (dens[0] + dens[-1]))
    return ts.cpu().numpy(), E


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    device = torch.device("cpu")
    tag = a.tag or (f"energia_{a.solver.lower()}"
                    + ("_lbfgs" if a.lbfgs_iters > 0 else "") + f"_seed{a.seed}")
    root = os.path.join(a.outdir, tag); os.makedirs(root, exist_ok=True)
    logger = Logging(root); run_dir = logger.get_output_dir()
    print(f"[energia] {run_dir}", flush=True)

    model = build_model(a, logger, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[energia] solver={a.solver} beta={a.beta} params={n_params}", flush=True)

    t_start = time.time()
    for it in range(a.epochs + 1):
        model.optimizer.zero_grad()
        nb = max(1, a.batch // 3)
        X_res = sample(a.batch, 0.0, a.T, device, "res", a.amp)
        X_bc = sample(nb, 0.0, a.T, device, "bc", a.amp)
        X_ic, u_ic = sample(nb, 0.0, a.T, device, "ic", a.amp)

        X_ic.requires_grad_(True)
        u_ic_pred = model.forward(X_ic)
        u_grad = torch.autograd.grad(u_ic_pred, X_ic, torch.ones_like(u_ic_pred),
                                     create_graph=True)[0]
        _, residual = klein_gordon_operator(model, X_res[:, 0:1], X_res[:, 1:2], beta=a.beta)

        loss_res = model.loss_fn(residual, torch.zeros_like(residual))     # f = 0
        loss_bc = model.loss_fn(model.forward(X_bc), torch.zeros_like(X_bc[:, :1]))
        loss_ics = model.loss_fn(u_ic_pred, u_ic)
        loss_ut = model.loss_fn(u_grad[:, 0:1], torch.zeros_like(u_grad[:, 0:1]))
        loss = (a.w_res * loss_res + a.w_ut * loss_ut
                + a.w_bc * loss_bc + a.w_ics * loss_ics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()
        model.scheduler.step(loss)
        model.loss_history.append(float(loss.item()))
        if it % (a.log_every * 50) == 0:
            print(f"[{it:6d}] loss={loss.item():.3e} res={loss_res.item():.3e} "
                  f"ic={loss_ics.item():.3e}", flush=True)

    drift_adam = None
    if a.lbfgs_iters > 0:
        # energia tras Adam, para comparar antes/despues en la misma corrida
        t_a, E_a = energy_curve(model, device, a.beta, a.amp, a.energy_nt, a.energy_nx, a.T)
        drift_adam = float(np.max(np.abs(E_a - E_a[0])) / abs(E_a[0]))
        print(f"[energia] fin de Adam: deriva = {drift_adam:.3e} -> L-BFGS "
              f"({a.lbfgs_iters} it, lote fijo)", flush=True)
        # lote FIJO (L-BFGS necesita perdida determinista)
        X_res = sample(a.lbfgs_nres, 0.0, a.T, device, "res", a.amp)
        X_bc = sample(a.lbfgs_nb, 0.0, a.T, device, "bc", a.amp)
        X_ic, u_ic = sample(a.lbfgs_nb, 0.0, a.T, device, "ic", a.amp)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=a.lbfgs_iters,
                                max_eval=int(a.lbfgs_iters * 1.25), history_size=50,
                                tolerance_grad=1e-9, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Xi = X_ic.clone().requires_grad_(True)
            u_ic_p = model.forward(Xi)
            g = torch.autograd.grad(u_ic_p, Xi, torch.ones_like(u_ic_p),
                                    create_graph=True)[0]
            _, res = klein_gordon_operator(model, X_res[:, 0:1].clone(),
                                           X_res[:, 1:2].clone(), beta=a.beta)
            loss = (a.w_res * model.loss_fn(res, torch.zeros_like(res))
                    + a.w_ut * model.loss_fn(g[:, 0:1], torch.zeros_like(g[:, 0:1]))
                    + a.w_bc * model.loss_fn(model.forward(X_bc), torch.zeros_like(X_bc[:, :1]))
                    + a.w_ics * model.loss_fn(u_ic_p, u_ic))
            loss.backward()
            return loss
        opt.step(closure)

    total = time.time() - t_start
    t, E = energy_curve(model, device, a.beta, a.amp, a.energy_nt, a.energy_nx, a.T)
    E0 = E[0]
    drift = float(np.max(np.abs(E - E0)) / abs(E0)) if E0 != 0 else float("nan")

    with open(os.path.join(run_dir, "energia.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "E"])
        [w.writerow([ti, ei]) for ti, ei in zip(t, E)]
    metrics = {"caso": "sin fuente (f=0), red", "solver": a.solver, "beta": a.beta,
               "seed": a.seed, "n_params": int(n_params), "total_time_s": total,
               "E0": float(E0), "E_final": float(E[-1]), "drift_rel": drift,
               "loss_final": float(model.loss_history[-1]),
               "lbfgs_iters": a.lbfgs_iters,
               "drift_rel_adam": drift_adam}
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print(f"\n  E(0) = {E0:.5f}   E(T) = {E[-1]:.5f}   deriva relativa = {drift:.3e}")
    print(f"  tiempo {total/60:.1f} min   loss_final {metrics['loss_final']:.3e}")
    print(f"[energia] listo -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
