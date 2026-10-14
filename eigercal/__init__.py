"""Eiger XRF calibration driven from Python, with the per-pixel fits done by aare.

The calibration is a sequence of steps in :mod:`eigercal.calibration`, each of
which reads and writes files in the data directory of the
:class:`~eigercal.config.Calibration` it is given, so a run can be resumed at
any step. ``scripts/eiger_calibration.py`` shows the full sequence.
"""
from .config import Calibration, XRF_ENERGY, setup_logging
from .geometry import GEOMETRIES, CHIPS, EigerGeometry
from . import calibration, fit, plot, scans, trimbits

__all__ = [
    'Calibration', 'XRF_ENERGY', 'setup_logging',
    'GEOMETRIES', 'CHIPS', 'EigerGeometry',
    'calibration', 'fit', 'plot', 'scans', 'trimbits',
]
