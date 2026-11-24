# ---------------------------------------------------------------------------
# solver_bottleneck.py - COPIA de src/nn/DVPDESolver.py (Farea, Khan y Celebi,
# licencia MIT) con el circuito cuantico retirado. El archivo original no se
# modifica.
#
# Proposito: control clasico del cuello de botella. El QCPINN cambia dos cosas
# a la vez respecto de la PINN: introduce el circuito Y estrecha la red a
# dimension num_qubits. Esta clase conserva el preprocesador y el
# postprocesador IDENTICOS a los de DVPDESolver y sustituye el circuito por un
# bloque clasico elegido con `mid`:
#
#   mid = "none"    nada: Linear(50,5) seguido de Linear(5,50)         756 par.
#   mid = "tanh"    tanh sin parametros (no linealidad acotada)        756 par.
#   mid = "cos"     cos sin parametros: es lo que mide <Z> tras RX(x)
#                   sobre |0>, es decir, el circuito sin puertas
#                   entrenables ni entrelazamiento                     756 par.
#   mid = "linear"  Linear(5,5) + tanh ("capa clasica pequena")        786 par.
#
# Referencia: el QCPINN angle-cascade tiene 771 parametros (756 + 15 angulos).
# ---------------------------------------------------------------------------
import os

import torch
import torch.nn as nn

from src.utils.logger import Logging

MID_CHOICES = ("none", "tanh", "cos", "linear")


class _Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class BottleneckSolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=None, mid="linear"):
        super().__init__()
        if mid not in MID_CHOICES:
            raise ValueError(f"mid debe ser uno de {MID_CHOICES}, se recibio {mid!r}")
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.mid_kind = mid
        self.batch_size = self.args["batch_size"]
        self.num_qubits = self.args["num_qubits"]   # aqui: ancho del cuello de botella
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.classic_network = self.args["classic_network"]

        # --- identico a DVPDESolver (rama angle) ---
        self.preprocessor = nn.Sequential(
            nn.Linear(self.classic_network[0], self.classic_network[-2]).to(self.device),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
        ).to(self.device)

        # --- identico a DVPDESolver ---
        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.classic_network[-2]).to(self.device),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(self.device),
        ).to(self.device)

        # --- lo unico que cambia: el bloque que ocupa el lugar del circuito ---
        if mid == "none":
            self.mid = nn.Identity()
        elif mid == "tanh":
            self.mid = nn.Tanh()
        elif mid == "cos":
            self.mid = _Cos()
        else:  # "linear"
            self.mid = nn.Sequential(
                nn.Linear(self.num_qubits, self.num_qubits).to(self.device),
                nn.Tanh(),
            ).to(self.device)

        self.activation = nn.Tanh()

        # --- optimizador, planificador y perdida: identicos a DVPDESolver ---
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args["lr"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=1000
        )
        self.loss_fn = torch.nn.MSELoss()

        self._initialize_logging()
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier sobre el preprocesador, igual que en DVPDESolver."""
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            preprocessed = self.preprocessor(x)
            features = self.mid(preprocessed)        # en DVPDESolver: el circuito
            return self.postprocessor(features)
        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    def save_state(self, path=None):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "mid_kind": self.mid_kind,
            "preprocessor": self.preprocessor.state_dict(),
            "mid": self.mid.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }
        model_path = path if path is not None else os.path.join(self.log_path, "model.pth")
        with open(model_path, "wb") as f:
            torch.save(state, f)
        self.logger.print(f"Model state saved to {model_path}")

    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            return torch.load(f, map_location=map_location)
