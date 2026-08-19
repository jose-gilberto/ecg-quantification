"""Continuous ECG signal visualization, colored by beat class.

Rather than looking at isolated beat segments, this plots a longer
contiguous stretch of the raw signal in the patient's own timeline, with
each beat's surrounding region colored by its annotated class — useful
for eyeballing transitions (e.g. N -> V runs) that per-beat plots hide.
"""
from typing import Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ecg_quantification.visualization._colors import get_color_palette, REFERENCE_COLOR


def find_variation_window(ann_samples: np.ndarray,
                          ann_symbols: np.ndarray,
                          window_beats: int = 30,
                          classes_of_interest: Sequence[str] = None,
                          stride: int = 1) -> tuple[int, int]:
  """Finds the `window_beats`-beat-wide stretch of annotations with the
  most class diversity, as a good default window to visualize.

  Slides a window of `window_beats` consecutive annotations across the
  whole record and scores each position by the number of distinct
  classes present (restricted to `classes_of_interest`, if given).
  Ties are broken by preferring the position with the most balanced
  class counts (highest entropy), so a window with one dominant class
  and a single outlier doesn't beat a genuinely mixed window.

  Parameters
  ----------
  ann_samples : ndarray of shape (n_annotations,)
    Sample index of every annotation in the record (e.g. `annotation.sample`).
  ann_symbols : ndarray of shape (n_annotations,)
    Beat symbol per annotation (e.g. `np.array(annotation.symbol)`).
  window_beats : int, default = 30
    Number of consecutive beats considered at each candidate position.
  classes_of_interest : Sequence[str], default = None
    If given, only these symbols count toward the diversity score;
    annotations with other symbols are ignored when scoring (but still
    fall inside the returned sample range if they happen to land there).
  stride : int, default = 1
    Step (in beats) between candidate window positions; increase for
    faster scanning on very long records.

  Returns
  -------
  start_sample, end_sample : int, int
    Sample-index bounds of the most class-diverse window found.

  Raises
  ------
  ValueError
    If there are fewer than `window_beats` annotations available.
  """
  n_ann = len(ann_samples)
  if n_ann < window_beats:
    raise ValueError(f"Only {n_ann} annotations available, fewer than window_beats={window_beats}.")

  if classes_of_interest is not None:
    relevant_mask = np.isin(ann_symbols, classes_of_interest)
  else:
    relevant_mask = np.ones(n_ann, dtype=bool)

  best_score = -1.0
  best_start_idx = 0

  for start_idx in range(0, n_ann - window_beats + 1, stride):
    end_idx = start_idx + window_beats
    window_symbols = ann_symbols[start_idx:end_idx][relevant_mask[start_idx:end_idx]]

    if window_symbols.size == 0:
      continue

    labels, counts = np.unique(window_symbols, return_counts=True)
    n_distinct = labels.size

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    # primary key: number of distinct classes; secondary: entropy (balance)
    score = n_distinct + entropy / np.log(window_beats)

    if score > best_score:
      best_score = score
      best_start_idx = start_idx

  end_idx = best_start_idx + window_beats
  start_sample = int(ann_samples[best_start_idx])
  end_sample = int(ann_samples[end_idx - 1])

  return start_sample, end_sample


def plot_signal_window(signal: np.ndarray,
                       ann_samples: np.ndarray,
                       ann_symbols: np.ndarray,
                       start_sample: int,
                       end_sample: int,
                       sampling_rate: float = 360.0,
                       margin_samples: int = 40,
                       colors: dict = None,
                       fig_size: tuple[float, float] = (14, 4),
                       font_size: int = 11,
                       line_width: float = 1.0,
                       title: str = "ECG Signal Window (colored by beat class)",
                       show_legend: bool = True,
                       ax: matplotlib.axes.Axes = None) -> matplotlib.figure.Figure:
  """Plots a contiguous ECG signal window, coloring each beat's local
  segment by its annotated class, preserving temporal continuity.

  Splits `[start_sample, end_sample]` into per-beat segments at the
  annotation midpoints (so each colored piece is centered on its own
  beat and the coloring partitions the whole window with no gaps), then
  draws every piece as its own colored line, connected end-to-end.

  Parameters
  ----------
  signal : ndarray of shape (n_samples,)
    The full raw (single-channel) signal for the record.
  ann_samples : ndarray of shape (n_annotations,)
    Sample index of every annotation in the record.
  ann_symbols : ndarray of shape (n_annotations,)
    Beat symbol per annotation, aligned with `ann_samples`.
  start_sample, end_sample : int
    Sample-index bounds of the window to plot (e.g. from `find_variation_window`).
  sampling_rate : float, default = 360.0
    Samples per second, used to render the x-axis in seconds.
  margin_samples : int, default = 40
    Extra samples included on each side of `[start_sample, end_sample]`
    so the first/last beat aren't cut off mid-segment.
  colors : dict[str, str], default = None
    Explicit `{class_symbol: color}` mapping. Classes without an entry
    fall back to the colorblind-safe palette. Defaults to None (fully
    auto-assigned).
  fig_size : tuple[float, float], default = (14, 4)
    Figure size in inches.
  font_size : int, default = 11
    Base font size for axis labels/title.
  line_width : float, default = 1.0
    Signal line width.
  title : str, default = "ECG Signal Window (colored by beat class)"
    Plot title.
  show_legend : bool, default = True
    Whether to draw a class-color legend.
  ax : matplotlib.axes.Axes, default = None
    Existing axes to draw on. A new figure/axes pair is created when None.

  Returns
  -------
  fig : matplotlib.figure.Figure
    The generated figure. Call `fig.savefig(path)` to persist it.

  Examples
  --------
  >>> import wfdb
  >>> from ecg_quantification.visualization import find_variation_window, plot_signal_window
  >>> record = wfdb.rdrecord('data/mitbih/records/100')
  >>> annotation = wfdb.rdann('data/mitbih/records/100', 'atr')
  >>> signal = record.p_signal[:, 0]
  >>> start, end = find_variation_window(annotation.sample, np.array(annotation.symbol))
  >>> fig = plot_signal_window(signal, annotation.sample, np.array(annotation.symbol), start, end)
  >>> fig.savefig("signal_window.png", dpi=300)
  """
  plot_start = max(start_sample - margin_samples, 0)
  plot_end = min(end_sample + margin_samples, len(signal))

  in_window = (ann_samples >= plot_start) & (ann_samples <= plot_end)
  window_samples = ann_samples[in_window]
  window_symbols = ann_symbols[in_window]

  if window_samples.size == 0:
    raise ValueError("No annotations fall within the requested [start_sample, end_sample] window.")

  # boundaries between consecutive beats' colored segments: midpoints
  # between annotations, so the coloring partitions the window with no
  # gaps and no overlap, and each piece is centered on its own beat
  midpoints = (window_samples[:-1] + window_samples[1:]) // 2
  segment_starts = np.concatenate(([plot_start], midpoints))
  segment_ends = np.concatenate((midpoints, [plot_end]))

  unique_classes = sorted(set(window_symbols.tolist()))
  palette = get_color_palette(len(unique_classes))
  color_map = {cls: (colors.get(cls, palette[i]) if colors else palette[i])
               for i, cls in enumerate(unique_classes)}

  own_axes = ax is None
  if own_axes:
    fig, ax = plt.subplots(figsize=fig_size)
  else:
    fig = ax.get_figure()

  time_axis = np.arange(plot_start, plot_end + 1) / sampling_rate

  plotted_classes = set()
  for seg_start, seg_end, symbol in zip(segment_starts, segment_ends, window_symbols):
    seg_start_idx = max(int(seg_start), plot_start)
    seg_end_idx = min(int(seg_end) + 1, plot_end + 1)
    if seg_end_idx <= seg_start_idx:
      continue

    x = np.arange(seg_start_idx, seg_end_idx) / sampling_rate
    y = signal[seg_start_idx:seg_end_idx]

    label = symbol if symbol not in plotted_classes else None
    ax.plot(x, y, color=color_map[symbol], linewidth=line_width, label=label, zorder=2)
    plotted_classes.add(symbol)

  # mark each beat's annotated sample with a small tick on the x-axis
  ax.scatter(window_samples / sampling_rate, [signal[int(s)] for s in window_samples],
             color=[color_map[s] for s in window_symbols], s=18, zorder=3,
             edgecolor=REFERENCE_COLOR, linewidth=0.5)

  ax.set_xlabel("Time (s)", fontsize=font_size)
  ax.set_ylabel("Amplitude", fontsize=font_size)
  if title:
    ax.set_title(title, fontsize=font_size + 2)
  ax.grid(alpha=0.3)

  if show_legend:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Beat class",
              fontsize=max(font_size - 1, 6), loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=min(6, len(by_label)), frameon=False)

  if own_axes:
    fig.tight_layout()

  return fig