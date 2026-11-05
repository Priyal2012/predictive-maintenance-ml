"""pdm — predictive maintenance on the AI4I 2020 machine-failure schema.

A compact, reproducible pipeline:

* ``data``      load the UCI AI4I 2020 dataset, or simulate data from the same
                documented failure physics when the download is unavailable
* ``features``  physically meaningful engineered features (power, ΔT, strain)
* ``train``     model comparison with stratified CV, class-imbalance handling,
                feature ablation, held-out evaluation, plots and a Markdown report
* ``cli``       ``pdm simulate`` / ``pdm train`` / ``pdm predict``
"""

__version__ = "0.1.0"
