#!/usr/bin/env python3
"""
dedupe_causal.py - Detecta corridas duplicadas en results_causal/.

Una corrida es duplicada si comparte (solver, brazo, semilla) con otra ya vista.
Se conserva la mas antigua y se marcan las posteriores.

    python dedupe_causal.py            # solo lista, no borra nada
    python dedupe_causal.py --delete   # borra las duplicadas
"""

import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_causal")
    ap.add_argument("--delete", action="store_true", help="borra las duplicadas")
    args = ap.parse_args()

    vistos, dups = {}, []
    for p in sorted(glob.glob(os.path.join(args.dir, "**", "metrics.json"),
                              recursive=True)):
        try:
            m = json.load(open(p))
        except Exception as e:
            print(f"  ilegible: {p} ({e})")
            continue
        key = (m.get("solver"), m.get("arm"), m.get("seed"))
        d = os.path.dirname(p)
        if key in vistos:
            dups.append((key, d, m.get("L2_u_window_pct")))
        else:
            vistos[key] = d

    print(f"{len(vistos)} corridas unicas, {len(dups)} duplicadas\n")
    for key, d, err in dups:
        print(f"  DUP {key}  L2={err}  ->  {d}")

    if not dups:
        print("Nada que borrar.")
    elif args.delete:
        for _, d, _ in dups:
            shutil.rmtree(d)
            padre = os.path.dirname(d)
            if os.path.isdir(padre) and not os.listdir(padre):
                os.rmdir(padre)
        print(f"\n{len(dups)} carpetas eliminadas.")
    else:
        print("\nRevisa la lista y vuelve a ejecutar con --delete si es correcta.")

    print("\nCorridas unicas por celda:")
    celdas = {}
    for (s, a, _) in vistos:
        celdas[(s, a)] = celdas.get((s, a), 0) + 1
    for k in sorted(celdas):
        print(f"  {k[0]:10s} brazo {k[1]}: {celdas[k]}")


if __name__ == "__main__":
    main()
