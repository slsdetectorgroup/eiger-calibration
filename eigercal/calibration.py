"""The calibration steps, in the order they are run.

Every step saves its result in the data directory of the
:class:`~eigercal.config.Calibration` and later steps load from there, so a
run can be resumed after any step. ``xraybox`` is any object with
``xrf_open()`` and ``xrf_close()``, such as ``slsdetbox.BigXrayBox``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import fit, plot, scans
from . import trimbits as tbfile
from .config import Calibration
from .geometry import GEOMETRIES, EigerGeometry

log = logging.getLogger(__name__)


def geometry_of(cal: Calibration) -> EigerGeometry:
    return GEOMETRIES[cal.geometry]


def _mask(data: np.ndarray, pixelmask) -> np.ndarray:
    """Zero the pixels that are True in ``pixelmask`` at every scan point."""
    if pixelmask is not None:
        data[np.asarray(pixelmask, dtype=bool)] = 0
    return data


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    log.info('Wrote %s', path)


# ---------------------------------------------------------------- vrf scan

def clean_vrf_data(data, min_threshold: float = 50.0) -> np.ndarray:
    """Zero the pixels that exceed three times the median of the counting
    pixels at any point of the scan. Returns a copy."""
    data = np.array(data, dtype=np.double)
    hot = np.zeros(data.shape[:2], dtype=bool)
    for i in range(data.shape[2]):
        frame = data[:, :, i]
        counting = frame[frame > 0]
        threshold = max(3 * np.median(counting), min_threshold) if counting.size else min_threshold
        hot |= frame > threshold
    data[hot] = 0
    return data


def fit_vrf_scan(x, data, geometry: EigerGeometry) -> tuple[list[int], list[np.ndarray]]:
    """vrpreamp per half module: the centre of a Gaussian fitted to the
    derivative of the summed counts. Also returns the fit parameters."""
    vrf, pars = [], []
    for hm in geometry.halfmodule:
        y = data[hm].sum(axis=(0, 1))
        par = fit.fit_vrf(x, np.gradient(y))
        pars.append(par)
        vrf.append(int(np.round(par[1])))
    return vrf, pars


def do_vrf_scan(detector, xraybox, cal: Calibration, start: int = 1500, stop: int = 3800,
                step: int = 30, pixelmask=None) -> tuple[list[int], float, list[float]]:
    """Scan vrpreamp and choose its value per half module.

    Returns the vrf values, a suggested exposure time for the threshold scan
    that gives about 1000 counts per pixel, and the mean counts at the chosen
    vrf per half module.
    """
    geometry = geometry_of(cal)
    xraybox.xrf_open()
    try:
        data, x = scans.vrf_scan(detector, cal, geometry, start, stop, step)
    finally:
        xraybox.xrf_close()
    _mask(data, pixelmask)
    data = clean_vrf_data(data)
    vrf, pars = fit_vrf_scan(x, data, geometry)
    counts = [float(data[:, :, np.argmin(np.abs(x - v))].mean()) for v in vrf]
    exptime = 1000 * cal.vrf_scan_exptime / min(counts)
    log.info('vrf: %s, counts at vrf: %s, suggested exptime: %.2f s', vrf, counts, exptime)
    np.savez(cal.vrf_file, data=data, x=x, vrf=vrf)
    if cal.plot:
        fig = plot.vrf_scan(x, data, geometry, pars, detector.hostname,
                            f'{cal.det_id} vrf scan: {cal.target}')
        _save(fig, cal.png('vrf'))
    return vrf, exptime, counts


def load_vrpreamp(cal: Calibration) -> np.ndarray:
    with np.load(cal.vrf_file) as f:
        return f['vrf']


# ---------------------------------------------------------- threshold scan

def do_scurve(detector, xraybox, cal: Calibration, start: int = 0, stop: int = 2001,
              step: int = 40, pixelmask=None) -> tuple[np.ndarray, np.ndarray]:
    """Take the threshold scan and save it. The data is saved unmasked."""
    geometry = geometry_of(cal)
    xraybox.xrf_open()
    try:
        data, x = scans.threshold_scan(detector, cal, geometry, start, stop, step)
    finally:
        xraybox.xrf_close()
    np.savez(cal.scurve_file, data=data, x=x)
    _mask(data, pixelmask)
    if cal.plot:
        fig = plot.scurve_scan(x, data, geometry, f'{cal.det_id} threshold scan: {cal.target}')
        _save(fig, cal.png('vcmp'))
    return data, x


def load_scurve(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cal.scurve_file) as f:
        return f['data'], f['x']


def do_scurve_fit(cal: Calibration, pixelmask=None, start=fit.SCURVE_START,
                  minimizer=fit.SCURVE_MINIMIZER) -> np.ndarray:
    """Fit an S-curve to every pixel of the saved threshold scan, scaled to
    1000 counts at the last point, and save the result."""
    data, x = load_scurve(cal)
    _mask(data, pixelmask)
    data = fit.scale_scurve_data(data)
    t0 = time.perf_counter()
    result = fit.fit_scurves(x, data, start=start, minimizer=minimizer, n_threads=cal.n_threads)
    log.info('S-curve fit of %d pixels in %.2f s, %d failed', result.size,
             time.perf_counter() - t0, int((result['sigma'] == 0).sum()))
    np.save(cal.fit_file, result)
    if cal.plot:
        mean, std, fig = plot.chip_histograms(result['mu'], geometry_of(cal))
        fig.axes[0].set_xlabel('Vcmp [DAC LSB]')
        fig.axes[0].set_ylabel('Number of pixels')
        _save(fig, cal.png('vcmp', '_fit'))
    return result


def load_fit_result(cal: Calibration) -> np.ndarray:
    return np.load(cal.fit_file)


# ------------------------------------------------------------ trimbit scan

def do_trimbit_scan(detector, xraybox, cal: Calibration, step: int = 2,
                    pixelmask=None) -> tuple[np.ndarray, np.ndarray]:
    """Set the chip thresholds from the S-curve fit, then scan the trimbits."""
    geometry = geometry_of(cal)
    scans.set_vcmp_from_fit(detector, load_fit_result(cal), geometry)
    xraybox.xrf_open()
    try:
        data, x = scans.trimbit_scan(detector, cal, geometry, step)
    finally:
        xraybox.xrf_close()
    _mask(data, pixelmask)
    np.savez(cal.tbscan_file, data=data, x=x)
    if cal.plot:
        _save(plot.trimbit_scan(x, data), cal.png('tbscan'))
    return data, x


def load_trimbit_scan(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cal.tbscan_file) as f:
        return f['data'], f['x']


def find_trimbits(cal: Calibration, pixelmask=None, fallback: bool = True):
    """Fit the saved trimbit scan and find the trimbit per pixel that puts its
    threshold at the inflection point found in the threshold scan.

    Saves the trimbits and the fit. Returns the trimbits as ``int32``, the
    target counts, the scaled scan data, its x values and the fit records.
    """
    scurve_data, _ = load_scurve(cal)
    data, x = load_trimbit_scan(cal)
    _mask(data, pixelmask)
    data = fit.scale_trimbit_data(data, scurve_data[:, :, -1])
    target = fit.scurve_at_mu(load_fit_result(cal))
    t0 = time.perf_counter()
    result, info = fit.find_trimbits(x, data, target, n_threads=cal.n_threads, fallback=fallback)
    log.info('Trimbit fit of %d pixels in %.2f s: %d fits failed, %d without crossing, '
             '%d from interpolation, %d without trimbit', result.size, time.perf_counter() - t0,
             info['fit_failed'], info['no_crossing'], info['fallback'], info['failed'])
    failed = np.isnan(result['trimbits'])
    tb = tbfile.clip_trimbits(result['trimbits'])
    np.savez(cal.trimbit_file, trimbits=tb, fit=result)
    if cal.plot:
        fig, _, _ = plot.image(failed)
        _save(fig, cal.png('tbscan', '_tbfail'))
        fig, _, _ = plot.image(tb, clim=(0, 63))
        _save(fig, cal.png('tbscan', '_image'))
        _save(plot.trimbit_histogram(tb), cal.png('tbscan', '_hist'))
    return tb, target, data, x, result


def write_trimbit_files(cal: Calibration, trimbits: np.ndarray, dacs: np.ndarray,
                        hostnames: list[str]) -> list[Path]:
    """Write one trimbit file per half module, named after its hostname.

    ``dacs`` is ``detector.dacs.get_asarray()``, one column per half module.
    """
    geometry = geometry_of(cal)
    if len(hostnames) != geometry.n_halfmodules:
        raise ValueError(f'{len(hostnames)} hostnames for {geometry.n_halfmodules} half modules')
    dacs = tbfile.file_dacs(dacs)
    paths = []
    for i, hm in enumerate(geometry.halfmodule):
        path = cal.trimbit_file_for(hostnames[i])
        tbfile.write_trimbit_file(path, trimbits[hm], dacs[:, i])
        log.info('Wrote %s', path)
        paths.append(path)
    return paths


def find_and_write_trimbits(detector, cal: Calibration, pixelmask=None, fallback: bool = True):
    """:func:`find_trimbits` followed by :func:`write_trimbit_files` with the
    dacs and hostnames of the detector."""
    tb, target, data, x, result = find_trimbits(cal, pixelmask, fallback)
    write_trimbit_files(cal, tb, detector.dacs.get_asarray(), detector.hostname)
    return tb, target, data, x, result


def load_trimbit_result(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """The saved trimbits and fit records."""
    with np.load(cal.trimbit_file) as f:
        return f['trimbits'], f['fit']


def load_trimbits(detector, cal: Calibration) -> None:
    scans.load_trimbits(detector, cal)


# ----------------------------------------------------------------- checks

def acquire_image(detector, xraybox, cal: Calibration) -> np.ndarray:
    """One image with the fluorescence shutter open, for a quick look."""
    with scans.setup_measurement(detector, cal.zmq_host, geometry_of(cal)) as receiver:
        xraybox.xrf_open()
        try:
            detector.acquire()
        finally:
            xraybox.xrf_close()
        return receiver.get_frame()
