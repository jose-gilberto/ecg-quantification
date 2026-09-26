
# ECG-Quantification

## Beyond Single-Beat Classification: Quantifying Arrhythmia in Long-Term ECG via Prevalence Estimation

### Abstract
Long-term electrocardiogram (ECG) monitoring is essential for determining the arrhythmic burden, a critical clinical metric for diagnosing cardiovascular conditions. Traditionally, this burden is estimated using a Classify-and-Count (CC) approach, which labels individual heartbeats and aggregates results by counting predictions for each label. However, even state-of-the-art Deep Learning classifiers exhibit systematic biases that accumulate over long-term recordings, leading to significant diagnostic inaccuracies. This paper investigates the application of quantification techniques to estimate arrhythmia prevalence in long-term ECG signals from the MIT-BIH Arrhythmia Database. We compare several base classifiers paired with quantification algorithms against a high-performance Deep Learning baseline, LITETime, using the standard CC method. Our results demonstrate a quantification paradox: while the LITETime achieves superior beat-by-beat accuracy, simpler classifiers equipped with quantification adjustment layers, particularly the Expectation-Maximization Quantifier (EMQ), significantly reduce the Mean Absolute Error (MAE) in prevalence estimation. By correcting the systematic bias caused by Prior Probability Shifts, our framework provides a more reliable diagnostic tool for long-term monitoring and wearable cardiac devices.

---

This repository contain all files needed to run the experiments for the CBMS ECG Quantification paper.

In order to make it more easier, all experiments are stored in the jupyter notebook format, inside of notebooks directory. All notebooks and files must be in order to execute (01_, 02_, ...). All the data is fetched from the PhysioNet python library, there is no need to manually download them.

For future organization, all files will be converted to python scripts, and become a lib format.

---

### Requirements & installation

Requires **Python 3.11+**. This is a hard floor, not a preference: the
`quack` dependency pins a recent `scikit-learn` (>=1.9.0), and
`scikit-learn>=1.8` itself requires Python 3.11+. Installing under an
older Python will fail during dependency resolution rather than at
runtime, so check your interpreter version first (`python3 --version`)
-- if it's below 3.11, create a virtual environment with a newer one
(e.g. via `pyenv`, `conda`, or your OS's `python3.11`/`python3.12`
package) before installing.

```bash
python3.11 -m venv .venv   # or 3.12 / 3.13
source .venv/bin/activate
pip install -e ".[test]"
```

#### Deep-learning classifier adapters (`.[deep]`, e.g. `LITETimeClassifier`)

The `deep` extra (`pip install -e ".[test,deep]"`) additionally installs
`aeon` + `tensorflow`/`tensorflow-cpu`, needed only for deep-learning
classifier adapters under `ecg_quantification.quantifiers`
(`LITETimeClassifier` today; likely `ECGFounderClassifier` later). This
has a Python **upper** bound the base package does not: as of this
writing, TensorFlow's newest *stable* release (2.21.0) only ships wheels
up to Python 3.13 -- Python 3.14+ (only a pre-release/`rc` build declares
support, not yet suitable for a reproducible experiment environment) will
fail to resolve with something like:

```
ERROR: Could not find a version that satisfies the requirement tensorflow-cpu>=2.16 ...
```

If `python3 --version` reports 3.14 or newer, create a *separate* venv
pinned to 3.11/3.12/3.13 specifically for `deep`-extra work, rather than
whatever your system's default/newest Python happens to be:

```bash
python3.12 -m venv .venv-deep   # any of 3.11 / 3.12 / 3.13; not 3.14+ yet
source .venv-deep/bin/activate
pip install -e ".[test,deep]"
```

This will move forward on its own as TensorFlow ships stable 3.14
support; no action needed here once that happens.

