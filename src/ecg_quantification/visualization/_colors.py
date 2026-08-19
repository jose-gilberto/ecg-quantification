"""Color utilities for ecg_quantification plots, mirroring the
colorblind-safe palette convention used in `quack.visualization`."""
import matplotlib.pyplot as plt
import numpy as np

COLORBLIND_PALETTE: list[str] = [
  '#0072B2',  # blue
  '#D55E00',  # vermillion
  '#009E73',  # bluish green
  '#CC79A7',  # reddish purple
  '#E69F00',  # orange
  '#56B4E9',  # sky blue
  '#F0E442',  # yellow
  '#000000',  # black
]

REFERENCE_COLOR: str = '#404040'


def get_color_palette(n_colors: int, palette=None) -> list:
  """Builds a list of `n_colors` visually distinct colors, extending the
  colorblind-safe base palette via a perceptually-uniform colormap when
  more colors are requested than the base palette provides."""
  base = list(palette) if palette is not None else list(COLORBLIND_PALETTE)

  if n_colors <= len(base):
    return base[:n_colors]

  cmap = plt.get_cmap('turbo')
  n_extra = n_colors - len(base)
  extra = [cmap(x) for x in np.linspace(0.05, 0.95, n_extra)]
  return base + extra