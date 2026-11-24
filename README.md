# QCPINN: Quantum-Classical Physics-Informed Neural Networks

Este repositorio contiene el código base de **QCPINN** (redes neuronales informadas por la física, híbridas cuántico-clásicas) y, sobre él, los experimentos adicionales de la monografía: comparación de topologías cuánticas contra una PINN clásica en la ecuación de **Klein-Gordon no lineal 1D**, referencias numéricas clásicas (diferencias finitas y elementos finitos), pruebas causales, conservación de energía y el análisis estadístico multi-semilla.

El código base proviene del artículo [QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs](https://iopscience.iop.org/article/10.1088/2632-2153/ae1c91) (Farea, Khan, Çelebi; *Machine Learning: Science and Technology*, 2025) y de su repositorio original [afrah/QCPINN](https://github.com/afrah/QCPINN).

---

## Estructura del repositorio

```
.
├── qcpinn.yaml            # Entorno conda
├── src/                   # Código base de QCPINN (sin modificar)
│   ├── nn/                # Redes: DV (variable discreta), CV, clásicas
│   ├── trainer/           # Entrenadores por PDE
│   ├── contour_plots/     # Gráficas de contorno
│   ├── testing/           # Pruebas amplitud vs. ángulo (cavity, helmholtz)
│   ├── data/  utils/      # Generador de datos y utilidades
│   └── notebooks/         # Notebooks de entrenamiento y visualización
├── data/                  # Datos de la cavidad (cavity.mat)
├── models/                # Modelos guardados (cavity, helmholtz, klein-gordon)
├── doc/results/           # Resultados del artículo original
│
├── run_*.py               # Runners de experimentos (ver abajo)
├── fd_*.py  fem_*.py      # Métodos numéricos clásicos de referencia
├── comparar_revision.py   # Agregación y contrastes estadísticos
├── graficar*.py  fig_*.py # Generación de figuras
│
├── results/               # Corridas individuales (una carpeta por semilla)
├── results_causal/        # Corridas de la prueba causal
├── analysis/              # Resúmenes y mapas de error de los experimentos base
├── analysis_revision/     # CSV e informes de los experimentos de revisión
└── figuras/               # Figuras finales (PDF)
```

## Instalación

Requiere [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/KevinCruz10/QCPINNS_TG.git
cd QCPINNS_TG
conda env create -f qcpinn.yaml
conda activate qcpinn
```

Comprueba el entorno (versiones de torch/pennylane/numpy y que el QNode funcione):

```bash
python gate_check.py
```

## Experimentos de Klein-Gordon

Ecuación estudiada: `u_tt - u_xx + β·u + u³ = f(x,t)` en `[0,1]`, con solución manufacturada. Con `f = 0` se usa el caso de vibración libre para medir la conservación de energía.

### Runners

| Script | Qué hace |
|---|---|
| `run_kg.py` | Runner instrumentado original: QCPINN DV vs. PINN clásica, una semilla por corrida. |
| `run_kg2.py` | Copia de `run_kg.py` para la revisión. Con los valores por defecto reproduce exactamente `run_kg.py`; añade L-BFGS, `--beta`, `--mid` (cuello de botella), pesos de pérdida, etc. |
| `run_kg_causal.py` | Prueba causal de la acumulación temporal del error (hipótesis H1 vs. H2). |
| `run_energia.py` | Caso sin fuente: mide la deriva de energía de la PINN y la QCPINN. |

```bash
# QCPINN con topología cascade, semilla 1
python run_kg.py --solver DV --ansatz cascade --seed 1

# PINN clásica (Model-2)
python run_kg.py --solver Classical --seed 1

# Revisión: Adam + L-BFGS con β = 1
python run_kg2.py --solver DV --ansatz cascade --beta 1 --lbfgs-iters 800 --seed 1
```

Topologías disponibles (`--ansatz`): `cascade`, `cross_mesh`, `alternate`, `layered`. Codificaciones (`--encoding`): `angle`, `amplitude`. Cada corrida guarda `metrics.csv`, `metrics.json`, `eval.npz` y checkpoints en `results/<nombre>_seed<N>/`.

### Colas de experimentos

Ejecutan grupos de semillas en paralelo, con fecha límite, y retoman si se interrumpen:

```bash
python cola_experimentos.py          # experimentos base
python cola_revision.py --list       # ver grupos de la revisión
python cola_revision.py --plan clasico
```

### Referencias numéricas clásicas

```bash
python fd_klein_gordon.py            # diferencias finitas, estudio de convergencia
python fem_klein_gordon.py           # elementos finitos P1 (Galerkin)
python fd_energia.py                 # conservación de energía, FD
python fem_energia.py                # conservación de energía, FEM
```

Las variantes `*_mass.py` (`fd_kg_mass.py`, `fem_kg_mass.py`, `pde_mass.py`, `kg_dataset_mass.py`) incluyen el término de masa β.

### Análisis

```bash
python analyze_kg.py                 # agregación multi-semilla
python diagnose_kg.py                # diagnóstico post-hoc sobre eval.npz
python aggregate_causal.py           # agrega results_causal/ (Welch, Mann-Whitney)
python dedupe_causal.py              # detecta corridas duplicadas (no borra)
python analizar_perdida.py           # Adam ya estaba plano: la mejora de L-BFGS es de segundo orden
python comparar_revision.py          # contrastes de la revisión → analysis_revision/
```

### Figuras

```bash
python graficar.py --indir analysis_revision --results results --out figuras --fmt pdf
python graficar_beta1.py
python figuras_extra.py              # convergencia numérica y campos
python fig_prediccion.py             # predicción vs. exacta (media ± desviación)
python fig_topologias.py             # las cuatro topologías (5 qubits, 1 capa)
python plot_comparison.py            # precisión vs. costo y grados de libertad
```

## Código base (artículo original)

Entrenamiento de las PDE del artículo:

```bash
python -m src.trainer.helmholtz_hybrid_trainer
python -m src.trainer.cavity_hybrid_trainer
python -m src.trainer.klein_gordon_hybrid_trainer
python -m src.trainer.wave_hybrid_trainer
python -m src.trainer.diffusion_hybrid_trainer
```

Gráficas de inferencia:

```bash
python -m src.contour_plots.helmholtz_hybrid_plotting
python -m src.contour_plots.cavity_hybrid_plotting
python -m src.contour_plots.klein_gordon_hybrid_plotting
python -m src.contour_plots.wave_hybrid_plotting
python -m src.contour_plots.diffusion_hybrid_plotting
```

Amplitud vs. ángulo:

```bash
python -m src.testing.cavity_test
python -m src.testing.helmholtz_test
```

Configuraciones del artículo (todas con codificación angular):

| PDE | Topología | Resultados |
|---|---|---|
| Helmholtz | Cascade | `doc/results/helmholtz` |
| Cavity flow | Cascade | `doc/results/cavity` |
| Wave | Cross-mesh | `doc/results/Wave` |
| Klein-Gordon | Cascade | `doc/results/klein-Gordon` |
| Convección-difusión | Cascade | `doc/results/convection-diffusion` |

Comparación de embeddings en `doc/results/different-embeddings` y resultados del modelo CV-QCPINN en `doc/results/CV-QCPINN`.

## Notas

- Los scripts se ejecutan desde la **raíz del repositorio**, al lado de `src/`.
- Los logs (`*.log`) y el instalador de Miniforge no se versionan (ver `.gitignore`).
- `results/` contiene muchas carpetas, una por semilla; los resúmenes agregados están en `analysis/` y `analysis_revision/`.

## Licencia

MIT, ver [LICENSE](LICENSE).

## Cita

Si este trabajo te resulta útil, cita el artículo original:

```bibtex
@article{Farea:2025:MLST,
    author={Farea, Afrah and Khan, Saiful and ÇELEBİ, Mustafa Serdar},
    title={QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs},
    journal={Machine Learning: Science and Technology},
    url={http://iopscience.iop.org/article/10.1088/2632-2153/ae1c91},
    year={2025},
}
```
