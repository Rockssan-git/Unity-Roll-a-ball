# FunModelling

FunModelling models how an agent learns to act in simple motor tasks, and how its **sense of agency** changes while it learns.

Agents are built from *blocks*. Each block:
- learns a forward model (action → outcome, a Gaussian process);
- searches for the action whose predicted outcome is closest to its goal;
- reports agency signals: goal closure, prediction error, competence progress, learning progress, feeling of effort and action convergence.

Blocks can be chained. In the experiments, a **D block** chooses a target position and a **M block** chooses the motor command that reaches it.

## Repository layout

```
src/
  env/toy_envs.py          2D linear-damping environments (one target / many targets)
  model/base.py            Agent and goal perception base classes
  model/block_agency.py    BlockAgency (one block) and ChainBlockAgency (chain of blocks)
  model/forward_model.py   Gaussian process forward models (standard and recency-weighted)
  model/formal_solutions.py  hand-coded agents for the ArrowRoll Unity build
  utils/                   JAX Gaussian process utilities, Plotly figures, helpers
  experiments/             runnable experiment scripts
  run_common.py            runs an agent on the ArrowRoll Unity build (ML-Agents)
tests/                     tests of the Gaussian process code
docs/                      Sphinx documentation
data/                      experiment outputs (Plotly .html figures)
env/                       ArrowRoll Unity build (Linux)
```

## Installation

Python 3.10 or newer. The code is not packaged: run everything from the repository root.

```bash
pip install numpy scipy pandas scikit-learn jax plotly pytest
# only for src/run_common.py (ArrowRoll Unity build):
pip install mlagents_envs gym
```

## Usage

Run an experiment:

```bash
python -m src.experiments.n_actions_n_targets
```

| Script | Environment | Setting |
|---|---|---|
| `one_action_one_target.py` | `LinearDampingSimpleTargetEnv` | one action per episode, one target |
| `n_actions_one_target.py` | `LinearDampingSimpleTargetEnv` | several actions per episode, one target |
| `one_action_n_targets.py` | `LinearDampingManyTargetsEnv` | one action per episode, five targets |
| `n_actions_n_targets.py` | `LinearDampingManyTargetsEnv` | several actions per episode, five targets |

Figures are written to `data/<environment name>/`.

Run an agent on the ArrowRoll Unity build:

```bash
python -m src.run_common
```

Run the tests:

```bash
python -m pytest tests -v
```

## Documentation

```bash
pip install -r docs/requirements.txt
cd docs && make html
```

Then open `docs/_build/html/index.html`. The documentation covers the concepts (blocks, sense of agency, forward models), the environments, the experiments, and the API reference generated from the docstrings.
