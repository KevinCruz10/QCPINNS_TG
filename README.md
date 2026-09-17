# QCPINNS_TG — Réplica y diagnóstico de QCPINN sobre la ecuación de Klein-Gordon

> **Aviso de origen.** Este repositorio **contiene una copia del código de
> [afrah/QCPINN](https://github.com/afrah/QCPINN)** (licencia MIT), redistribuida
> aquí bajo los términos de dicha licencia, **más los scripts de réplica y
> diagnóstico desarrollados para este trabajo de grado**. La sección
> [Qué es propio y qué no](#qué-es-propio-y-qué-no) precisa la autoría de cada
> archivo. El código original replicado corresponde al commit
> `0dc467be8b16cccaa5101e63fe8cdf8d18bfdcb9`.

Código y resultados agregados de la monografía *Implementación y Análisis de
Eficiencia de Redes Neuronales Híbridas Cuántico-Clásicas Informadas por la
Física para la Ecuación de Klein-Gordon*, Universidad Distrital Francisco José
de Caldas, 2026.

---

## Trabajo replicado

A. Farea, S. Khan y M. S. Celebi, «QCPINN: quantum-classical physics-informed
neural networks for solving PDEs», *Machine Learning: Science and Technology*,
vol. 6, n.º 4, 045053, 2025. DOI: [10.1088/2632-2153/ae1c91](https://doi.org/10.1088/2632-2153/ae1c91)

La réplica abarca **únicamente la ecuación de Klein-Gordon**, una de las cinco
que el estudio original evalúa.

---

## Qué es propio y qué no

| Ruta | Autoría | Contenido |
|---|---|---|
| `src/`, `data/`, `doc/`, `models/`, `qcpinn.yaml` | Farea, Khan y Celebi | Código original, redistribuido bajo MIT sin modificación |
| `LICENSE` | Farea, Khan y Celebi | Licencia MIT del código original |
| `run_kg.py`, `run_kg_causal.py`, `analyze_kg.py`, `diagnose_kg.py`, `aggregate_causal.py`, `dedupe_causal.py`, `fd_klein_gordon.py`, `fem_klein_gordon.py`, `plot_comparison.py`, `figuras_extra.py`, `cola_experimentos.py`, `gate_check.py` | Kevin Cruz | Scripts propios de réplica, diagnóstico y análisis |
| `resultados/`, `figuras/` | Kevin Cruz | Resultados agregados y figuras de la monografía |
| `environment.yml`, `LICENSE-TG`, este `README.md` | Kevin Cruz | Entorno, licencia propia y documentación de la réplica |

Los scripts propios **importan** el código original (`src.nn`, `src.data`,
`src.utils`) sin modificarlo. Ninguna línea de `src/` fue alterada.

---

## Scripts propios

| Script | Función |
|---|---|
| `gate_check.py` | Verificación del entorno: versiones, forma del tensor del QNode, autodiff de segundo orden, conteo paramétrico |
| `run_kg.py` | Entrenamiento instrumentado: semilla efectiva, tiempo con retropropagación, pérdida desagregada, evaluación por bloques |
| `run_kg_causal.py` | Experimento causal de dos brazos sobre ventanas temporales complementarias |
| `analyze_kg.py` | Agregación multi-semilla, media y desviación por grupo |
| `diagnose_kg.py` | Mapas de error promediados, espectro del error, amplitud y fase del modo dominante |
| `aggregate_causal.py` | Agregación del experimento causal con pruebas de Welch y Mann-Whitney |
| `dedupe_causal.py` | Detección y eliminación de ejecuciones duplicadas |
| `fd_klein_gordon.py` | Solucionador de diferencias finitas de segundo orden |
| `fem_klein_gordon.py` | Solucionador de elementos finitos de Galerkin P1 |
| `plot_comparison.py` | Figura de precisión frente a costo y grados de libertad |
| `figuras_extra.py` | Figuras de convergencia, campos, espacio de diseño y prueba causal |
| `cola_experimentos.py` | Cola de ejecución desatendida con fecha límite |

---

## Requisitos

```bash
conda create -n qcpinn python=3.10 -y && conda activate qcpinn
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.23.5" "scipy==1.10.1" "autoray==0.7.1" "setuptools==69.5.1" \
            "pennylane==0.29.0" "pennylane-lightning==0.29.0" \
            "matplotlib==3.7.5" "pandas==2.0.3" "h5py==3.10.0"
```

**El pin de PennyLane es obligatorio.** `DVPDESolver.forward()` ejecuta
`quantum_out.view(num_qubits, -1).T` sobre la salida del QNode. Desde PennyLane
0.33 esa salida es una tupla y el código falla. Python 3.10 es consecuencia de
que numpy 1.23.5 no dispone de ruedas para versiones superiores.

La ejecución es en CPU de manera deliberada: con cinco qubits el vector de
estado tiene 32 amplitudes y el costo de transferencia a memoria de vídeo domina
sobre el cálculo. Los autores originales llegan a la misma conclusión.

---

## Réplica

```bash
python gate_check.py                                    # debe terminar en GATE OK

# conteos paramétricos: 771 / 796 / 2751 / 7851
python run_kg.py --solver DV --ansatz cascade --seed 1 --epochs 0 --eval-points 20

# baseline clásico, 10 semillas
seq 1 10 | xargs -P 8 -I{} python run_kg.py --solver Classical --seed {} \
    --epochs 20000 --threads 2 --eval-chunk 500

# barrido completo desatendido
python cola_experimentos.py --plan completo --deadline-hours 15

# agregación y figuras
python analyze_kg.py --runs "cascade=results/kg_dv_angle_cascade_seed*" \
    "pinn=results/kg_classical_seed*" --out analysis/
python fd_klein_gordon.py && python fem_klein_gordon.py
python figuras_extra.py
```

Tiempo aproximado en un Ryzen 7 3700X de ocho núcleos: 0,26 s por iteración para
el modelo híbrido y 0,0045 s para el clásico, de modo que una ejecución de
20 000 épocas tarda unos 85 minutos y 1,5 minutos respectivamente.

---

## Correcciones aplicadas al código original

Durante la réplica se detectaron cinco divergencias entre el código publicado y
su descripción en el artículo. Los scripts propios las corrigen; `src/` no fue
modificado.

1. **Dispositivo mal asignado.** `DVPDESolver.__init__(self, args, logger, data=None, device=None)`
   recibe el dispositivo en posición de `data` desde los entrenadores, por lo que
   `self.device` queda en `None`. El entrenamiento ocurre en CPU de manera
   silenciosa y la evaluación falla si hay GPU disponible.
2. **Semilla no aplicada.** El parámetro existe en la configuración pero no se
   invoca `torch.manual_seed` en ningún punto; las repeticiones no son
   deterministas.
3. **Tiempo mal medido.** El cronómetro se detiene antes de `loss.backward()`,
   excluyendo la fase computacionalmente dominante.
4. **Tamaño de lote ignorado.** `train()` fija 128 en el cuerpo del
   procedimiento y descarta el valor de la configuración.
5. **Ponderaciones discrepantes.** El código aplica `0.1*(L_res + L_ut) + 10*L_bc`,
   mientras el apéndice del artículo consigna λ₁=1.0, λ₂=10.0, λ₃=1.0.

Adicionalmente, la evaluación original procesa 40 000 puntos en un único paso con
grafo de segundo orden; `run_kg.py` la trocea mediante `--eval-chunk`.

---

## Resultados principales

- La réplica es exitosa: todas las métricas caen dentro de las desviaciones
  publicadas y el ordenamiento de las cuatro topologías se reproduce.
- El modelo híbrido alcanza precisión estadísticamente equivalente a la del
  clásico con el 28 % de los parámetros, sin superarlo y con mayor dispersión
  entre ejecuciones.
- La topología del circuito produce un efecto sobre la precisión de 3,46 veces,
  frente a 1,25 de la presencia misma del circuito.
- Cinco mecanismos candidatos a explicar el error residual resultaron
  descartados: fallo de anclaje, sesgo espectral, meseta estéril, acumulación
  temporal y limitación de capacidad.

Los valores agregados que sustentan estas afirmaciones están en `resultados/`.

---

## Licencia

El código original bajo `src/`, `data/`, `doc/`, `models/` y `qcpinn.yaml` se
distribuye bajo la licencia MIT de sus autores, cuyo texto se conserva en
`LICENSE`.

Los scripts propios enumerados arriba, junto con `resultados/`, `figuras/` y
esta documentación, se distribuyen bajo licencia MIT con copyright propio, cuyo
texto está en `LICENSE-TG`.

---

## Cómo citar

Si utiliza los scripts de réplica o los resultados agregados:

```bibtex
@mastersthesis{cruz2026qcpinn,
  author = {Cruz, Kevin},
  title  = {Implementación y Análisis de Eficiencia de Redes Neuronales
            Híbridas Cuántico-Clásicas Informadas por la Física para la
            Ecuación de Klein-Gordon},
  school = {Universidad Distrital Francisco José de Caldas},
  year   = {2026},
  note   = {Código: https://github.com/KevinCruz10/QCPINNS_TG}
}
```

Si utiliza el modelo QCPINN, cite el trabajo original de Farea, Khan y Celebi.
