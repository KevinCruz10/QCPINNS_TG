#!/usr/bin/env python3
"""
cola_experimentos.py - Cola de experimentos desatendida con fecha limite.

Ejecuta grupos en orden de prioridad; dentro de cada grupo lanza las semillas en
paralelo. Antes de cada oleada comprueba el tiempo restante y se detiene si no
cabe completa, en vez de dejar corridas a medias.

    python cola_experimentos.py --list
    python cola_experimentos.py --dry-run
    python cola_experimentos.py --wait-for run_kg_causal.py --deadline-hours 11

Cada corrida escribe su metrics.json al terminar: una cola interrumpida deja
resultados parciales utilizables, solo con menos semillas en el ultimo grupo.

GRUPOS
  Replicacion de la Tabla 7 del articulo, objetivo 2 (espacio de diseño):
     alternate, layered, cross_mesh
  Optimizacion, objetivo 3 (coherencia de las condiciones de Cauchy):
     wut1, wut10       -- el caso w_ut = 0.1 ya existe: son las corridas cascade
  Extension propia, objetivo 2 (codificacion):
     amp_cascade

APAGADO
Este script NO apaga la maquina. Programa el apagado aparte, con margen:
    sudo shutdown -h +720     # 12 h
    sudo shutdown -c          # cancelar
y deja --deadline-hours una o dos horas por debajo.
"""

import argparse
import os
import subprocess
import time

# nombre -> (tag, args_extra, horas_por_semilla, semillas_por_defecto, objetivo)
GRUPOS = {
    "alternate": ("kg_dv_angle_alternate",
                  ["--solver", "DV", "--ansatz", "alternate", "--encoding", "angle"],
                  0.9, 10, "obj2 replicacion Tabla 7"),
    "layered": ("kg_dv_angle_layered",
                ["--solver", "DV", "--ansatz", "layered", "--encoding", "angle"],
                1.0, 10, "obj2 replicacion Tabla 7"),
    "wut1": ("kg_dv_cascade_wut1",
             ["--solver", "DV", "--ansatz", "cascade", "--encoding", "angle",
              "--w-ut", "1.0"],
             1.4, 5, "obj3 optimizacion: peso de u_t(0,x)"),
    "wut10": ("kg_dv_cascade_wut10",
              ["--solver", "DV", "--ansatz", "cascade", "--encoding", "angle",
               "--w-ut", "10.0"],
              1.4, 5, "obj3 optimizacion: peso de u_t(0,x)"),
    "cross_mesh": ("kg_dv_angle_cross_mesh",
                   ["--solver", "DV", "--ansatz", "cross_mesh", "--encoding", "angle"],
                   2.4, 10, "obj2 replicacion Tabla 7"),
    "amp_cascade": ("kg_dv_amplitude_cascade",
                    ["--solver", "DV", "--ansatz", "cascade", "--encoding", "amplitude"],
                    1.4, 10, "obj2 extension propia: codificacion"),
}

PLANES = {
    "revisado": ["alternate", "layered", "wut1", "wut10", "cross_mesh"],
    "replica": ["alternate", "layered", "cross_mesh"],
    "optimizacion": ["wut1", "wut10"],
    "completo": ["alternate", "layered", "wut1", "wut10", "cross_mesh", "amp_cascade"],
}


def hay_procesos(patron):
    r = subprocess.run(["pgrep", "-f", patron], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANES), default="revisado")
    ap.add_argument("--grupos", nargs="+", default=None,
                    help="lista explicita de grupos; anula --plan")
    ap.add_argument("--list", action="store_true", help="lista los grupos y sale")
    ap.add_argument("--seeds", type=int, default=None,
                    help="fuerza el mismo numero de semillas en todos los grupos")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
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
        pendientes = list(range(1, (a.seeds or n_def) + 1))
        print(f"\n=== grupo: {nombre}  [{obj}] ===", flush=True)
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
                cmd = ["python", "run_kg.py", *extra,
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
