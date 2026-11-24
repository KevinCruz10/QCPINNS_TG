#!/usr/bin/env python3
"""
fig_topologias.py — Figura de las cuatro topologías evaluadas (cap. 4, sección 4.6.6).

Dibuja los circuitos EXACTAMENTE como están implementados en src/nn/DVQuantumLayer.py
(5 qubits, 1 capa, codificación angular RX), y verifica profundidad, número de
parámetros y compuertas de dos qubits contra la Tabla 4-1.

Uso (desde la raíz del repositorio QCPINNS_TG):
    python fig_topologias.py
Salida: figuras/fig_topologias.pdf  y  figuras/fig_topologias.png
"""
import sys, types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Solo se necesitan los métodos de construcción del circuito, que no usan torch.
_stub = False
try:
    import torch  # noqa: F401
except ImportError:  # permite dibujar sin tener torch instalado
    _stub = True
    stub = types.ModuleType("torch"); stub.nn = types.ModuleType("torch.nn")
    stub.nn.Module = object; stub.Tensor = object
    sys.modules["torch"] = stub; sys.modules["torch.nn"] = stub.nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.nn.DVQuantumLayer import DVQuantumLayer as DV
if _stub:  # matplotlib no debe ver el torch falso
    del sys.modules["torch"], sys.modules["torch.nn"]

N = 5  # qubits
ns = SimpleNamespace(num_qubits=N)

TOPOS = {  # nombre en la tesis: (método, n.º de parámetros por capa según el código)
    "Alternate":  (DV.alternate,         4 * N - 4),
    "Cascade":    (DV.cascade,           3 * N),
    "Cross-mesh": (DV.create_cross_mesh, 4 * N + N * (N - 1)),
    "Layered":    (DV.layered,           4 * N),
}

def build(fn, n_par):
    def circuit(x, params):
        qml.AngleEmbedding(x, wires=range(N), rotation="X")
        fn(ns, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    return qml.QNode(circuit, qml.device("default.qubit", wires=N))

x0 = np.linspace(0.1, 0.5, N)

print(f"{'Topología':<11} {'prof. ansatz':>12} {'parámetros':>10} {'2 qubits':>9}")
for name, (fn, n_par) in TOPOS.items():
    p = np.linspace(0.1, 1.0, n_par)
    tape = qml.workflow.construct_tape(build(fn, n_par))(x0, p)
    ops = [op for op in tape.operations if op.name != "AngleEmbedding"]
    depth = qml.tape.QuantumScript(ops).graph.get_depth()
    two_q = sum(len(op.wires) == 2 for op in ops)
    print(f"{name:<11} {depth:>12} {n_par:>10} {two_q:>9}")

# ---------- Figura ----------
out = Path("figuras"); out.mkdir(exist_ok=True)
for name, (fn, n_par) in TOPOS.items():
    p = np.linspace(0.1, 1.0, n_par)
    fig, ax = qml.draw_mpl(build(fn, n_par), decimals=None, style="black_white",
                           wire_order=range(N), level="top")(x0, p)
    slug = name.lower().replace("-", "_")
    fig.savefig(out / f"circ_{slug}.pdf", bbox_inches="tight")
    fig.savefig(out / f"circ_{slug}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
print("Guardado en figuras/circ_*.pdf")
