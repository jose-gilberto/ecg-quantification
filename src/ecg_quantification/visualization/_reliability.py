"""Reliability diagrams and calibration summaries (Fase 5).

Static matplotlib figures for the paper. Every series pairs its color with
a marker shape (and every figure has a legend), so identity never rests
on color alone; the uncalibrated baseline is a neutral dark gray and the
calibration methods take the project's colorblind-safe palette in a fixed
order.
"""
import matplotlib.pyplot as plt
import numpy as np

from ecg_quantification.metrics._calibration import (
  classwise_calibration_error,
  expected_calibration_error,
  reliability_curve,
)
from ecg_quantification.visualization._colors import COLORBLIND_PALETTE

INK = "#262626"
MUTED_INK = "#6b6b6b"
GRID = "#e6e6e3"
DIAGONAL = "#b5b5b0"
UNCALIBRATED = "#404040"

# fixed styles: the same method looks the same in every figure
METHOD_STYLES = {
  "uncalibrated": {"color": UNCALIBRATED, "marker": "o"},
  "sigmoid": {"color": COLORBLIND_PALETTE[0], "marker": "s"},
  "isotonic": {"color": COLORBLIND_PALETTE[1], "marker": "^"},
  "temperature": {"color": COLORBLIND_PALETTE[2], "marker": "D"},
}
_EXTRA = [{"color": c, "marker": m} for c, m in zip(COLORBLIND_PALETTE[3:], "vP*X")]


def method_style(name: str, index: int = 0) -> dict:
  """Color/marker for a series, by method name (e.g. 'sigmoid'); the method
  is taken from the last word so 'post-hoc sigmoid' styles like 'sigmoid'."""
  key = name.split()[-1].lower()
  return METHOD_STYLES.get(key, _EXTRA[index % len(_EXTRA)])


def _style_axes(ax, xlabel=None, ylabel=None, square=True):
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  if square:
    ax.set_aspect("equal")
  ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED_INK)
    ax.spines[side].set_linewidth(0.8)
  ax.tick_params(colors=MUTED_INK, labelcolor=INK, labelsize=8)
  if xlabel:
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
  if ylabel:
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def _draw_curves(ax, y_true, probas: dict, classes, n_bins, strategy, class_label, with_ece=True):
  ax.plot([0, 1], [0, 1], color=DIAGONAL, linewidth=1.0, zorder=1, label="_perfect")
  for index, (name, proba) in enumerate(probas.items()):
    curve = reliability_curve(y_true, proba, classes, n_bins=n_bins, strategy=strategy, class_label=class_label)
    filled = curve["bin_count"] > 0
    style = method_style(name, index)
    if with_ece:
      ece = (expected_calibration_error(y_true, proba, classes, n_bins=n_bins, strategy=strategy)
             if class_label is None else
             classwise_calibration_error(y_true, proba, classes, n_bins=n_bins, strategy=strategy,
                                         average=False)[class_label])
      label = f"{name} (ECE {ece:.3f})"
    else:
      label = name
    ax.plot(
      curve["bin_confidence"][filled], curve["bin_accuracy"][filled],
      color=style["color"], marker=style["marker"], linewidth=1.8, markersize=5.5,
      markeredgecolor="white", markeredgewidth=1.0, solid_joinstyle="round",
      solid_capstyle="round", label=label, zorder=3 + index,
    )


def plot_reliability_diagram(y_true, probas: dict, classes, n_bins: int = 15, strategy: str = "uniform",
                             class_label=None, title: str = None, ax=None, show_counts: bool = True):
  """Reliability diagram comparing several probability matrices for the
  same samples (e.g. uncalibrated vs. sigmoid/isotonic/temperature).

  Parameters
  ----------
  y_true : array-like of shape (n_samples,)
  probas : dict {series name: ndarray (n_samples, n_classes)}
    Columns ordered as `classes`. Names ending in a method name
    ('uncalibrated', 'sigmoid', 'isotonic', 'temperature') get that
    method's fixed color/marker.
  classes : array-like of shape (n_classes,)
  n_bins, strategy : see `ecg_quantification.metrics.reliability_curve`.
  class_label : optional
    `None`: top-label (confidence of the predicted class vs. accuracy).
    A class: that class's probability vs. its observed frequency.
  title : str, optional
  ax : matplotlib Axes, optional
    Draw into this axes (no count panel). Otherwise a new figure with a
    small panel underneath showing the share of samples per bin.
  show_counts : bool, default = True

  Returns
  -------
  fig : matplotlib Figure
  """
  own_figure = ax is None
  if not own_figure:
    fig = ax.figure
    count_ax = None
  elif show_counts:
    # no forced equal aspect here (it would shrink the reliability axes and
    # misalign them with the count panel); the figure size keeps it ~square
    fig, (ax, count_ax) = plt.subplots(
      2, 1, figsize=(4.0, 5.6), gridspec_kw={"height_ratios": [4, 1.1], "hspace": 0.08}, sharex=True,
    )
  else:
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    count_ax = None

  what = "Confidence (top-label)" if class_label is None else f"Predicted P({class_label})"
  observed = "Accuracy" if class_label is None else f"Observed frequency of {class_label}"
  _draw_curves(ax, y_true, probas, classes, n_bins, strategy, class_label)
  _style_axes(ax, None if count_ax is not None else what, observed, square=count_ax is None)
  if title:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
  if not own_figure:
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK)

  if count_ax is not None:
    for index, (name, proba) in enumerate(probas.items()):
      curve = reliability_curve(y_true, proba, classes, n_bins=n_bins, strategy=strategy, class_label=class_label)
      share = curve["bin_count"] / curve["bin_count"].sum()
      count_ax.stairs(share, curve["bin_edges"], color=method_style(name, index)["color"], linewidth=1.4)
    count_ax.set_xlim(0, 1)
    count_ax.set_yscale("log")
    count_ax.set_ylim(1e-4, 1.2)
    count_ax.grid(True, color=GRID, linewidth=0.8)
    count_ax.set_axisbelow(True)
    for side in ("top", "right"):
      count_ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      count_ax.spines[side].set_color(MUTED_INK)
    count_ax.tick_params(colors=MUTED_INK, labelcolor=INK, labelsize=7)
    count_ax.set_ylabel("Share", color=INK, fontsize=8)
    count_ax.set_xlabel(what, color=INK, fontsize=9)

  if own_figure:
    # legend below everything, where it can never cover a curve
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.5, frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.02))
  return fig


def plot_classwise_reliability(y_true, probas: dict, classes, n_bins: int = 15, strategy: str = "uniform",
                               title: str = None):
  """Small multiples: one reliability diagram per class (one-vs-rest),
  same series in every panel, one shared legend."""
  classes = np.asarray(classes)
  n = len(classes)
  fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.3), sharey=True)
  axes = np.atleast_1d(axes)
  for index, (ax, label) in enumerate(zip(axes, classes)):
    _draw_curves(ax, y_true, probas, classes, n_bins, strategy, label, with_ece=False)
    _style_axes(ax, f"Predicted P({label})", "Observed frequency" if index == 0 else None)
    ax.set_title(f"Class {label}", color=INK, fontsize=9, loc="left")
  fig.tight_layout(rect=(0, 0, 1, 0.86))
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=8, frameon=False,
             labelcolor=INK, bbox_to_anchor=(0.995, 0.995))
  if title:
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, y=0.985, ha="left", va="top")
  return fig


PROTOCOL_LABELS = {
  "post-hoc": "Post-hoc\n(cross-fitted on\ntest patients)",
  "cv": "CalibratedClassifierCV\n(training split)",
}


def plot_calibration_summary(results, metrics=("ece", "classwise_ece", "brier"),
                             metric_labels=None, protocols=None):
  """Dot plot of calibration metrics: one column per metric, one row per
  protocol, classifiers on the y axis, one marker per method.

  Parameters
  ----------
  results : pandas.DataFrame
    Columns `classifier`, `protocol`, `method` and the metric columns
    (the `calibration_metrics.csv` written by `scripts/calibration_study.py`).
    The uncalibrated rows (`protocol == "uncalibrated"`) are shown in
    every protocol row as the reference.
  """
  metric_labels = metric_labels or {
    "ece": "Top-label ECE", "classwise_ece": "Classwise ECE", "brier": "Brier score",
  }
  uncalibrated = results[results["protocol"] == "uncalibrated"]
  protocols = protocols or [p for p in results["protocol"].unique() if p != "uncalibrated"]
  classifiers = list(dict.fromkeys(results["classifier"]))
  fig, axes = plt.subplots(
    len(protocols), len(metrics), figsize=(3.4 * len(metrics), 0.45 * len(classifiers) * len(protocols) + 1.4),
    squeeze=False, sharey=True, sharex="col",
  )
  for row, protocol in enumerate(protocols):
    subset = results[results["protocol"] == protocol]
    for col, metric in enumerate(metrics):
      ax = axes[row, col]
      positions = {name: i for i, name in enumerate(classifiers)}
      series = [("uncalibrated", uncalibrated)] + [
        (method, subset[subset["method"] == method]) for method in subset["method"].unique()
      ]
      offsets = np.linspace(-0.2, 0.2, len(series))
      for index, ((method, rows), offset) in enumerate(zip(series, offsets)):
        if rows.empty:
          continue
        rows = rows[rows["classifier"].isin(subset["classifier"]) | (method != "uncalibrated")]
        style = method_style(method, index)
        ax.scatter(
          rows[metric], [positions[c] + offset for c in rows["classifier"]],
          color=style["color"], marker=style["marker"], s=34, edgecolor="white", linewidth=0.8,
          label=method, zorder=3,
        )
      if col == 0:
        from matplotlib.transforms import blended_transform_factory
        for name in classifiers:
          if name not in set(subset["classifier"]):
            ax.text(0.02, positions[name], "not refit: post-hoc only", color=MUTED_INK, fontsize=7.5,
                    va="center", transform=blended_transform_factory(ax.transAxes, ax.transData))
      ax.set_yticks(range(len(classifiers)))
      ax.set_yticklabels(classifiers)
      ax.set_ylim(len(classifiers) - 0.5, -0.5)
      ax.grid(True, axis="x", color=GRID, linewidth=0.8)
      ax.set_axisbelow(True)
      for side in ("top", "right"):
        ax.spines[side].set_visible(False)
      for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
      ax.tick_params(colors=MUTED_INK, labelcolor=INK, labelsize=8)
      if row == 0:
        ax.set_title(metric_labels.get(metric, metric), color=INK, fontsize=9, loc="left")
      if col == 0:
        ax.set_ylabel(PROTOCOL_LABELS.get(protocol, protocol), color=INK, fontsize=8.5)
      if row == len(protocols) - 1:
        ax.set_xlabel("lower is better", color=MUTED_INK, fontsize=8)
  handles, labels = [], []
  for ax in axes.ravel():
    for handle, label in zip(*ax.get_legend_handles_labels()):
      if label not in labels:
        handles.append(handle)
        labels.append(label)
  fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
             labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
  fig.tight_layout(rect=(0, 0.06, 1, 1))
  return fig
