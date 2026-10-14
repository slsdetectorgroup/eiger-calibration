"""Plots made by the calibration steps. Every function returns its figure;
the caller saves and closes it."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .fit import gaussian
from .geometry import EigerGeometry


def random_pixels(shape: tuple[int, int], n: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng()
    return list(zip(rng.integers(0, shape[0], n), rng.integers(0, shape[1], n)))


def histogram(data, xmin: float, xmax: float, nbins: int) -> dict:
    """Histogram of the finite values in ``data`` with the mean and standard
    deviation of the values inside the range, like a ROOT TH1D reported."""
    values = np.asarray(data, dtype=np.double).ravel()
    values = values[np.isfinite(values)]
    y, edges = np.histogram(values, bins=nbins, range=(xmin, xmax))
    inside = values[(values >= xmin) & (values < xmax)]
    return {'x': edges[:-1], 'y': y,
            'mean': inside.mean() if inside.size else np.nan,
            'std': inside.std() if inside.size else np.nan}


def image(data, cmap: str = 'coolwarm', clim=None, log: bool = False, figsize=(16, 10)):
    """Image with a colour bar. Returns ``(fig, ax, im)``."""
    data = np.asarray(data)
    if data.ndim == 3:
        data = data[:, :, 0]
    fig, ax = plt.subplots(figsize=figsize)
    norm = mpl.colors.LogNorm(vmin=0.1) if log else None
    im = ax.imshow(data, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    if clim is not None:
        im.set_clim(*clim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax)
    ax.grid(False)
    return fig, ax, im


def vrf_scan(x, data, geometry: EigerGeometry, pars, hostnames, title: str):
    """Counts and their derivative per half module with the Gaussian fits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    xx = np.linspace(x.min(), x.max(), 300)
    legend = geometry.n_halfmodules <= 10
    for i, hm in enumerate(geometry.halfmodule):
        y = data[hm].sum(axis=(0, 1))
        line, = ax1.plot(x, y, 'o', label=hostnames[i] if legend else None)
        ax2.plot(x, np.gradient(y), 'o', color=line.get_color(),
                 label=f'$\\mu$: {pars[i][1]:.0f}' if legend else None)
        ax2.plot(xx, gaussian(xx, *pars[i]), color=line.get_color())
    mu = np.mean([p[1] for p in pars])
    ax2.set_xlim(mu - 300, mu + 300)
    ax1.set_xlabel('vrf [1]')
    ax1.set_ylabel('Counts [1]')
    ax2.set_xlabel('vrf [1]')
    ax2.set_ylabel('Counts (differential) [1]')
    if legend:
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def scurve_scan(x, data, geometry: EigerGeometry, title: str):
    """Sample pixels and the sum per chip of a threshold scan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for r, c in random_pixels(data.shape[:2], 50):
        ax1.plot(x, data[r, c, :])
    for module in geometry.module:
        for chip in geometry.chips(module):
            ax2.plot(x, data[chip].sum(axis=(0, 1)))
    ax1.set_xlabel('vcmp [1]')
    ax1.set_ylabel('counts [1]')
    ax1.set_title('Sample pixels')
    ax2.set_xlabel('vcmp [1]')
    ax2.set_ylabel('counts [1]')
    ax2.set_title('Sum per chip')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def chip_histograms(values, geometry: EigerGeometry, xmin: float = 0, xmax: float = 2000,
                    bins: int = 400, log: bool = False):
    """Histogram per chip of a per-pixel quantity such as the inflection point.

    Returns the lists of means and standard deviations, chip by chip in the
    order of ``geometry.vcmp_names``, and the figure.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    mean, std = [], []
    ymax = 0
    for k, module in enumerate(geometry.module):
        for i, chip in enumerate(geometry.chips(module)):
            h = histogram(values[chip], xmin, xmax, bins)
            mean.append(h['mean'])
            std.append(h['std'])
            ax.step(h['x'], h['y'], label=f'{k}-{i}: $\\mu$: {h["mean"]:.1f} $\\sigma$: {h["std"]:.1f}')
            ymax = max(ymax, h['y'][1:].max(initial=0))
    ax.legend(loc='best')
    ax.set_ylim(0, ymax * 1.1)
    ax.set_xlim(xmin, xmax)
    if log:
        ax.set_yscale('log')
    return mean, std, fig


def trimbit_scan(x, data):
    """Sum over all pixels and sample pixels of a trimbit scan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.plot(x, data.sum(axis=(0, 1)))
    for r, c in random_pixels(data.shape[:2], 50):
        ax2.plot(x, data[r, c, :])
    ax1.set_xlabel('Trimbit [1]')
    ax1.set_ylabel('Counts [1]')
    ax1.set_title('Sum')
    ax2.set_xlabel('Trimbit [1]')
    ax2.set_ylabel('Counts [1]')
    ax2.set_title('Sample pixels')
    fig.suptitle('Trimbit scan')
    fig.tight_layout()
    return fig


def trimbit_histogram(trimbits):
    h = histogram(trimbits, 0, 64, 64)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.step(h['x'], h['y'], where='post')
    ax.set_xlabel('Trimbits')
    ax.set_ylabel('Pixels')
    return fig
