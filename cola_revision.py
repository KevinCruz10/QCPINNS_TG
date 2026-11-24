#!/usr/bin/env python3
"""
cola_revision.py - COPIA de cola_experimentos.py para los experimentos de la
revision. Lanza run_kg2.py (no run_kg.py) y salta las semillas ya terminadas,
de modo que una cola interrumpida se retoma con el mismo comando.

    python cola_revision.py --list
    python cola_revision.py --plan clasico --dry-run
    python cola_revision.py --plan clasico --parallel 4 --threads 1 --deadline-hours 3
    python cola_revision.py --plan noche_beta --parallel 2 --threads 2 --deadline-hours 10

GRUPOS
  Punto 1 del profesor (control del cuello de botella, todo clasico):
     bn_linear, bn_cos, bn_tanh, bn_none
  Punto 3 (peso del residuo; el caso 0.1 ya existe en kg_classical_seed*):
     wres1, wres10, wres100            clasico
     dv_wres1, dv_wres10, dv_wres100   hibrido: lanzar SOLO el valor que elija el barrido clasico
  Punto 4 (L-BFGS tras Adam, Model-2):
     lbfgs
  Punto 8 (termino de masa, beta = 1):
     beta1_cls, beta1_bn               clasico
     beta1_dv                          hibrido (una noche)

Este script NO apaga la maquina ni evita que se suspenda. Antes de una noche:
    systemd-inhibit --what=sleep:idle --who="cola" --why="experimentos" sleep infinity &
"""

import argparse
import glob
import os
import subprocess
import time

# nombre -> (tag, args_extra, horas_por_semilla, semillas_por_defecto, punto del profesor)
_DV = ["--solver", "DV", "--ansatz", "cascade", "--encoding", "angle"]
_H_CLS, _H_DV = 0.06, 1.8      # horas por corrida en el Ryzen 3 5300U (piloto: 306 ms/it)
GRUPOS = {
    "wres1":   ("kg2_cls_wres1",   ["--solver", "Classical", "--w-res", "1.0"],   _H_CLS, 10, "p3 peso del residuo"),
    "wres10":  ("kg2_cls_wres10",  ["--solver", "Classical", "--w-res", "10.0"],  _H_CLS, 10, "p3 peso del residuo"),
    "wres100": ("kg2_cls_wres100", ["--solver", "Classical", "--w-res", "100.0"], _H_CLS, 10, "p3 peso del residuo"),
    "bn_linear": ("kg2_bn_linear", ["--solver", "Bottleneck", "--mid", "linear"], _H_CLS, 10, "p1 control 786 par."),
    "bn_cos":    ("kg2_bn_cos",    ["--solver", "Bottleneck", "--mid", "cos"],    _H_CLS, 10, "p1 control 756 par."),
    "bn_tanh":   ("kg2_bn_tanh",   ["--solver", "Bottleneck", "--mid", "tanh"],   _H_CLS, 10, "p1 control 756 par."),
    "bn_none":   ("kg2_bn_none",   ["--solver", "Bottleneck", "--mid", "none"],   _H_CLS, 10, "p1 control 756 par."),
    "lbfgs":     ("kg2_cls_lbfgs", ["--solver", "Classical", "--lbfgs-iters", "5000"], 0.12, 10, "p4 Adam + L-BFGS"),
    "beta1_cls": ("kg2_cls_beta1", ["--solver", "Classical", "--beta", "1.0"],    _H_CLS, 10, "p8 masa, clasico"),
    "beta1_bn":  ("kg2_bn_linear_beta1", ["--solver", "Bottleneck", "--mid", "linear", "--beta", "1.0"],
                  _H_CLS, 10, "p8 masa, control"),
    "beta1_dv":  ("kg2_dv_beta1",  _DV + ["--beta", "1.0"],                       _H_DV, 10, "p8 masa, hibrido"),
    "dv_wres1":   ("kg2_dv_wres1",   _DV + ["--w-res", "1.0"],   _H_DV, 10, "p3 hibrido"),
    "dv_wres10":  ("kg2_dv_wres10",  _DV + ["--w-res", "10.0"],  _H_DV, 10, "p3 hibrido"),
    "dv_wres100": ("kg2_dv_wres100", _DV + ["--w-res", "100.0"], _H_DV, 10, "p3 hibrido"),
}

PLANES = {
    # todo lo clasico: ~1 h con --parallel 4 --threads 1
    "clasico": ["wres1", "wres10", "wres100", "bn_linear", "bn_cos", "bn_tanh", "bn_none",
                "lbfgs", "beta1_cls", "beta1_bn"],
    "barrido": ["wres1", "wres10", "wres100"],
    "control": ["bn_linear", "bn_cos", "bn_tanh", "bn_none"],
    "noche_beta": ["beta1_dv"],
}


def hay_procesos(patron):
    r = subprocess.run(["pgrep", "-f", patron], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANES), default="clasico")
    ap.add_argument("--grupos", nargs="+", default=None,
                    help="lista explicita de grupos; anula --plan")
    ap.add_argument("--list", action="store_true", help="lista los grupos y sale")
    ap.add_argument("--seeds", type=int, default=None,
                    help="fuerza el mismo numero de semillas en todos los grupos")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--eval-chunk", type=int, default=500)
    ap.add_argument("--deadline-hours", type=float, default=11.0)
    ap.add_argument("--wait-for", default=None)
    ap.add_argument("--logdir", default="results/cola_logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.list:
        print(f"{'grupo':14s} {'h/semilla':>10s} {'semillas':>9s}  objetivo")
        for k, (_, _, h, n, obj) in GRUPOS.items():
            print(f"{k:14s} {h:>10.1f} {n:>9d}  {obj}")
        print("\nPlanes:")
        for k, v in PLANES.items():
            print(f"  {k:14s} {' -> '.join(v)}")
        return

    nombres = a.grupos or PLANES[a.plan]
    for n in nombres:
        if n not in GRUPOS:
            raise SystemExit(f"grupo desconocido: {n}. Usa --list.")

    os.makedirs(a.logdir, exist_ok=True)
    env = dict(os.environ,
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    etiqueta_plan = "personalizado" if a.grupos else a.plan
    print(f"=== plan '{etiqueta_plan}' ===")
    total = 0.0
    for nombre in nombres:
        _, _, h, n_def, obj = GRUPOS[nombre]
        n = a.seeds or n_def
        oleadas = -(-n // a.parallel)
        est = oleadas * h
        total += est
        print(f"  {nombre:14s} {n:2d} semillas x {h:.1f} h -> "
              f"{oleadas} oleada(s) ~{est:.1f} h   [{obj}]")
    print(f"  {'TOTAL':14s} ~{total:.1f} h    (limite {a.deadline_hours} h)")
    if total > a.deadline_hours:
        print("  AVISO: excede el limite; los ultimos grupos quedaran incompletos.")
    if a.dry_run:
        return

    if a.wait_for:
        print(f"\nEsperando a que terminen los procesos '{a.wait_for}'...", flush=True)
        while hay_procesos(a.wait_for):
            time.sleep(60)
        print("Listo. Iniciando cola.\n", flush=True)

    inicio, limite = time.time(), a.deadline_hours * 3600
    hechas = detenidas = 0

    for nombre in nombres:
        tag, extra, horas, n_def, obj = GRUPOS[nombre]
        todas = list(range(1, (a.seeds or n_def) + 1))
        # reanudacion: una semilla con metrics.json ya esta terminada y no se repite
        pendientes = [sd for sd in todas
                      if not glob.glob(os.path.join("results", f"{tag}_seed{sd}", "*", "metrics.json"))]
        print(f"\n=== grupo: {nombre}  [{obj}] ===", flush=True)
        if len(pendientes) < len(todas):
            print(f"  {len(todas) - len(pendientes)} semilla(s) ya terminadas, se omiten", flush=True)
        cortado = False
        while pendientes:
            restante = limite - (time.time() - inicio)
            if restante < horas * 3600:
                print(f"  quedan {restante/3600:.1f} h y una oleada necesita "
                      f"{horas:.1f} h. Se detiene la cola.", flush=True)
                detenidas += len(pendientes)
                cortado = True
                break
            lote, pendientes = pendientes[:a.parallel], pendientes[a.parallel:]
            procs = []
            for s in lote:
                log = os.path.join(a.logdir, f"{nombre}_seed{s}.log")
                cmd = ["python", "run_kg2.py", *extra,
                       "--seed", str(s), "--epochs", str(a.epochs),
                       "--threads", str(a.threads),
                       "--eval-chunk", str(a.eval_chunk),
                       "--tag", f"{tag}_seed{s}"]
                fh = open(log, "w")
                procs.append((s, subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env), fh))
                print(f"  lanzada semilla {s} -> {log}", flush=True)
            for s, p, fh in procs:
                p.wait()
                fh.close()
                hechas += 1
                estado = "ok" if p.returncode == 0 else f"FALLO ({p.returncode})"
                print(f"  semilla {s}: {estado}  "
                      f"[{(time.time()-inicio)/3600:.2f} h]", flush=True)
        if cortado:
            break

    print(f"\n=== fin: {hechas} corridas en "
          f"{(time.time()-inicio)/3600:.2f} h, {detenidas} no lanzadas ===")


if __name__ == "__main__":
    main()
