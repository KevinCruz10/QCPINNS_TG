#!/usr/bin/env python3
"""
gate_check.py - Verificacion previa del entorno para QCPINN (Klein-Gordon).
Colocar en la raiz del repo QCPINN y ejecutar:  python gate_check.py

Version en archivo (en vez de heredoc) porque fish no soporta `<<EOF`.

Comprueba, en orden:
  1. Versiones de torch / pennylane / numpy.
  2. Que el QNode devuelva un TENSOR PLANO reshapeable a (num_qubits, batch).
     PennyLane >= 0.33 devuelve una tupla y rompe DVPDESolver.forward().
  3. Que la doble derivada respecto de las ENTRADAS atraviese el circuito
     (es lo que necesita el residuo u_tt - u_xx + u^3).
  4. Que DVPDESolver se construya con device por keyword y devuelva 771
     parametros con el ansatz cascade.
"""

import sys
import traceback

FAILS = []


def check(name, fn):
    try:
        fn()
        print(f"  [OK]   {name}")
    except Exception as e:
        print(f"  [FALLA] {name}: {e}")
        traceback.print_exc()
        FAILS.append(name)


def main():
    print("== 1. Versiones ==")
    import numpy
    import torch
    import pennylane as qml
    print(f"  torch      {torch.__version__}")
    print(f"  pennylane  {qml.version()}")
    print(f"  numpy      {numpy.__version__}")
    if not qml.version().startswith("0.29"):
        print("  AVISO: se espera pennylane 0.29.x; otras versiones rompen el reshape.")

    print("\n== 2. Capa cuantica ==")
    from src.nn.DVQuantumLayer import DVQuantumLayer

    args = dict(num_qubits=5, num_quantum_layers=1, shots=None,
                q_ansatz="cascade", problem="klein_gordon", encoding="angle")
    ql = DVQuantumLayer(args)
    x = torch.randn(7, 5, requires_grad=True)
    y = ql(x)
    print(f"  tipo de salida: {type(y).__name__}")

    def _reshape():
        out = y.view(5, -1).T
        print(f"         reshape -> {out.shape}")
        assert tuple(out.shape) == (7, 5), f"se esperaba (7,5), se obtuvo {tuple(out.shape)}"

    check("reshape (num_qubits, batch).T", _reshape)

    def _double_grad():
        g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        g2 = torch.autograd.grad(g.sum(), x)[0]
        print(f"         doble derivada -> {g2.shape}")
        assert tuple(g2.shape) == (7, 5)

    check("autodiff de 2do orden por el circuito", _double_grad)

    print("\n== 3. Solver completo ==")

    def _solver():
        import os
        import tempfile
        from src.utils.logger import Logging
        from src.nn.DVPDESolver import DVPDESolver

        tmp = tempfile.mkdtemp(prefix="gate_")
        logger = Logging(tmp)
        solver_args = {
            "batch_size": 8, "epochs": 1, "lr": 0.005, "seed": 1, "print_every": 1,
            "log_path": tmp, "input_dim": 2, "output_dim": 1, "num_qubits": 5,
            "hidden_dim": 50, "num_quantum_layers": 1, "classic_network": [2, 50, 1],
            "q_ansatz": "cascade", "mode": "hybrid", "activation": "tanh",
            "shots": None, "problem": "klein_gordon", "solver": "DV",
            "device": "cpu", "method": "None", "cutoff_dim": 20,
            "class": "CVNeuralNetwork1", "encoding": "angle",
        }
        # device POR KEYWORD: la firma real es (args, logger, data=None, device=None).
        # Los trainers del repo lo pasan posicionalmente y acaba en `data`.
        model = DVPDESolver(solver_args, logger, data=None,
                            device=torch.device("cpu"))
        n = sum(p.numel() for p in model.parameters())
        print(f"         parametros = {n}")
        assert n == 771, f"se esperaban 771 parametros (angle-cascade), hay {n}"

        from src.nn.pde import klein_gordon_operator
        pts = torch.rand(16, 2)
        u, res = klein_gordon_operator(model, pts[:, 0:1], pts[:, 1:2])
        print(f"         u {tuple(u.shape)}  residuo {tuple(res.shape)}")
        assert u.shape == res.shape == (16, 1)
        print(f"         artefactos de prueba en {tmp}")

    check("DVPDESolver cascade -> 771 params + residuo KG", _solver)

    print()
    if FAILS:
        print("GATE FALLIDO en: " + ", ".join(FAILS))
        sys.exit(1)
    print("GATE OK - entorno listo para run_kg.py")


if __name__ == "__main__":
    main()
