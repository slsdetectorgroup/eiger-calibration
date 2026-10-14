"""Reading and writing Eiger trimbit files.

File format, all ``int32``::

    dacs     : ndacs values (16 dacs, iodelay, tau)
    trimbits : 256 * 1024 values, one half module
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import HALFMODULE_SHAPE

NDACS = 18
_NPIXELS = HALFMODULE_SHAPE[0] * HALFMODULE_SHAPE[1]


def write_trimbit_file(fname: str | Path, trimbits: np.ndarray, dacs: np.ndarray,
                       ndacs: int = NDACS) -> None:
    """Write the trimbits of one half module and its dacs to ``fname``.

    Raises
    ------
    ValueError
        If ``dacs`` does not hold ``ndacs`` values or ``trimbits`` is not one
        half module.
    """
    dacs = np.asarray(dacs)
    trimbits = np.asarray(trimbits)
    if dacs.size != ndacs:
        raise ValueError(f'Expected {ndacs} dacs, got {dacs.size}')
    if trimbits.size != _NPIXELS:
        raise ValueError(f'Expected {_NPIXELS} trimbits (one half module), got {trimbits.size}')
    with open(fname, 'wb') as f:
        f.write(dacs.astype(np.int32).tobytes())
        f.write(trimbits.astype(np.int32).tobytes())


def read_trimbit_file(fname: str | Path, ndacs: int = NDACS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(trimbits[256, 1024], dacs[ndacs])`` from a trimbit file.

    Raises
    ------
    ValueError
        If the file size does not match one half module plus ``ndacs`` dacs.
    """
    data = np.fromfile(fname, dtype=np.int32)
    if data.size != _NPIXELS + ndacs:
        raise ValueError(f'Wrong file size, expected {_NPIXELS + ndacs} int32 values, got {data.size}')
    return data[ndacs:].reshape(HALFMODULE_SHAPE), data[:ndacs]


def file_dacs(detector_dacs: np.ndarray) -> np.ndarray:
    """Dacs as stored in the trimbit file, ``[ndacs, n_halfmodules]``.

    ``detector.dacs.get_asarray()`` returns the 16 dacs and iodelay; the file
    also holds tau, which is written as 0.
    """
    detector_dacs = np.asarray(detector_dacs)
    return np.vstack((detector_dacs, np.zeros((1, detector_dacs.shape[1]))))


def clip_trimbits(trimbits: np.ndarray, fill: int = 32) -> np.ndarray:
    """Round to ``int32`` in 0..63, replacing NaN with ``fill``."""
    tb = np.where(np.isnan(trimbits), fill, trimbits)
    return np.clip(np.round(tb), 0, 63).astype(np.int32)
