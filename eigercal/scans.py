"""Detector control: setup, the three scans and setting the chip thresholds.

``detector`` is an ``slsdet.Eiger``; nothing here imports slsdet so the
functions can be exercised with a stand-in.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import Calibration
from .geometry import EigerGeometry
from .receiver import setup_measurement

log = logging.getLogger(__name__)


def setup_detector(detector, cal: Calibration) -> None:
    """Put the detector in the state the calibration expects: frames, period,
    exposure time, dynamic range, trimbits, vtrim and vrshaper."""
    detector.frames = cal.nframes
    detector.period = cal.period
    detector.exptime = cal.exptime
    detector.dr = cal.dynamic_range
    detector.trimval = cal.trimval
    detector.dacs.vtrim = cal.vtr
    detector.dacs.vrshaper = cal.vrs


def _scan(detector, cal: Calibration, geometry: EigerGeometry, values: np.ndarray,
          apply, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Acquire one frame per value, after passing it to ``apply``."""
    size = detector.detsize
    data = np.zeros((size.y, size.x, values.size))
    with setup_measurement(detector, cal.zmq_host, geometry) as receiver:
        for i, value in enumerate(values):
            apply(value)
            log.info('%s = %s', label, value)
            detector.acquire()
            data[:, :, i] = receiver.get_frame()
    return data, values


def vrf_scan(detector, cal: Calibration, geometry: EigerGeometry,
             start: int = 1500, stop: int = 3800, step: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Scan vrpreamp at a fixed threshold with a short exposure, in 16 bit."""
    dr = detector.dr
    detector.dr = 16
    detector.vthreshold = cal.threshold
    detector.exptime = cal.vrf_scan_exptime

    def apply(value):
        detector.dacs.vrpreamp = int(value)

    try:
        return _scan(detector, cal, geometry, np.arange(start, stop, step), apply, 'vrpreamp')
    finally:
        detector.dr = dr


def threshold_scan(detector, cal: Calibration, geometry: EigerGeometry,
                   start: int = 0, stop: int = 2001, step: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Scan vthreshold, the S-curve data of the calibration."""
    detector.dr = cal.dynamic_range
    detector.exptime = cal.exptime

    def apply(value):
        detector.vthreshold = int(value)

    return _scan(detector, cal, geometry, np.arange(start, stop, step), apply, 'vthreshold')


def trimbit_scan(detector, cal: Calibration, geometry: EigerGeometry,
                 step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Scan the trimbit value of all pixels."""
    detector.exptime = cal.exptime

    def apply(value):
        detector.trimval = int(value)

    return _scan(detector, cal, geometry, np.arange(0, 64, step), apply, 'trimval')


def set_vcmp_from_fit(detector, fit_result: np.ndarray, geometry: EigerGeometry,
                      mu_range: tuple[float, float] = (100, 1900)) -> tuple[np.ndarray, np.ndarray]:
    """Set the threshold of each chip to the mean inflection point of its
    pixels, and vcp of each half module to the mean of its chips.

    Pixels with ``mu`` outside ``mu_range`` are ignored. A chip without such
    pixels gets 0. Pass ``detector=None`` to only compute the values.

    Returns ``vcmp[n_modules, 8]`` and ``vcp[n_modules, 2]``.
    """
    mu = fit_result['mu']
    vcmp = np.zeros((geometry.n_modules, 8), dtype=np.int64)
    vcp = np.zeros((geometry.n_modules, 2), dtype=np.int64)
    for j, module in enumerate(geometry.module):
        for i, chip in enumerate(geometry.chips(module)):
            m = mu[chip]
            m = m[(m > mu_range[0]) & (m < mu_range[1])]
            vcmp[j, i] = int(m.mean()) if m.size else 0
        for h in range(2):
            v = vcmp[j, 4 * h:4 * h + 4]
            v = v[v > 0]
            vcp[j, h] = int(v.mean()) if v.size else 0
    log.info('vcmp per chip: %s', vcmp.tolist())
    log.info('vcp per half module: %s', vcp.tolist())
    if detector is not None:
        detector.vcmp = [int(v) for v in vcmp.flat]
        detector.dacs.vcp = [int(v) for v in vcp.flat]
    return vcmp, vcp


def load_trimbits(detector, cal: Calibration) -> None:
    """Load the trimbit files of this calibration into the detector."""
    log.info('Loading trimbits: %s', cal.trimbit_base)
    detector.trimbits = str(cal.trimbit_base)
