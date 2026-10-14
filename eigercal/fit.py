"""Per-pixel fits of the calibration data, done by aare.

Three fits are needed:

* a Gaussian to the derivative of the vrf scan, one per half module;
* an S-curve per pixel of the threshold scan, which gives the inflection
  point ``mu`` of every pixel;
* an S-curve per pixel of the trimbit scan, solved for the trimbit at which
  the pixel counts as many photons as it did at ``mu``.

The S-curve is aare's ``RisingScurve``::

    f(x) = (p0 + p1*x) + 0.5*(1 + erf((x - mu) / (sqrt(2)*sigma))) * (A + C*(x - mu))

Start values and limits are carried over from the ROOT based fits in
``sls_detector_tools`` and were checked to reproduce them, see ``PLAN.md``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

import aare

#: Parameters of the S-curve, in the order of aare's RisingScurve.
SCURVE_PAR = ('p0', 'p1', 'mu', 'sigma', 'A', 'C')
#: Layout of a saved S-curve fit, one record per pixel.
SCURVE_DTYPE = np.dtype([(name, np.double) for name in SCURVE_PAR])
#: Layout of a saved trimbit fit: the S-curve parameters and the trimbit.
TRIMBIT_DTYPE = np.dtype([(name, np.double) for name in SCURVE_PAR] + [('trimbits', np.double)])

SCURVE_START = (60.0, 0.01, 1114.95212, 198.609468, 594.207866, 0.44786038)
SCURVE_MU_LIMITS = (0.0, 2000.0)
SCURVE_SIGMA_LIMITS = (50.0, 300.0)
SCURVE_MINIMIZER = aare.Minimizer.VarPro

#: Start values for p0, p1, mu, sigma and A; C is fixed to 0 for trimbits.
TRIMBIT_START = (9.35, 2.08, 33.5, 14.47, 507.0)
TRIMBIT_MU_LIMITS = (0.0, 64.0)
TRIMBIT_MINIMIZER = aare.Minimizer.LevenbergMarquardt
#: Trimbits are searched in this range, slightly wider than 0..63.
TRIMBIT_RANGE = (-10.0, 74.0)


def scurve(x, p0, p1, mu, sigma, A, C):
    """The S-curve model evaluated with numpy."""
    return (p0 + p1 * x) + 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma))) * (A + C * (x - mu))


def gaussian(x, A, mu, sigma):
    """Gaussian with the parameter order of aare's Gaussian model."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def to_structured(par: np.ndarray, dtype: np.dtype = SCURVE_DTYPE) -> np.ndarray:
    """Convert a parameter cube ``[rows, cols, npar]`` to one record per pixel."""
    out = np.zeros(par.shape[:-1], dtype)
    for i, name in enumerate(dtype.names[:par.shape[-1]]):
        out[name] = par[..., i]
    return out


def to_cube(fit: np.ndarray, names=SCURVE_PAR) -> np.ndarray:
    """The inverse of :func:`to_structured`."""
    return np.stack([fit[name] for name in names], axis=-1)


def scurve_at_mu(fit: np.ndarray) -> np.ndarray:
    """Counts at the inflection point of each fitted S-curve.

    This is the target for trimming: the erf term is zero at ``mu`` so the
    value is ``p0 + p1*mu + A/2``.
    """
    return fit['p0'] + fit['p1'] * fit['mu'] + 0.5 * fit['A']


def _cube(data) -> np.ndarray:
    data = np.asarray(data, dtype=np.double)
    if data.ndim != 3:
        raise ValueError(f'Expected data as [rows, cols, n_scan], got shape {data.shape}')
    return np.ascontiguousarray(data)


def _empty(data: np.ndarray) -> np.ndarray:
    """Pixels without a single count; they are not fitted, like in the old code."""
    return ~(data != 0).any(axis=2)


def scale_scurve_data(data) -> np.ndarray:
    """Normalise every pixel to 1000 counts at the last point of the scan.

    Pixels without counts at the last point are set to zero and fail the fit.
    """
    data = _cube(data)
    last = data[:, :, -1]
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = data / last[:, :, None] * 1000.0
    scaled[~np.isfinite(scaled).all(axis=2)] = 0.0
    return scaled


def scale_trimbit_data(data, scurve_last) -> np.ndarray:
    """Scale a trimbit scan per pixel by the plateau of the threshold scan.

    This puts the trimbit scan in the units of the S-curve fit, 1000 counts at
    the plateau, so that the target from :func:`scurve_at_mu` applies.
    """
    data = _cube(data)
    scale = np.asarray(scurve_last, dtype=np.double) / 1000.0
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = data / scale[:, :, None]
    scaled[~np.isfinite(scaled).all(axis=2)] = 0.0
    return scaled


def vrf_window(x, y_derivative) -> tuple[float, float]:
    """Fit window around the steepest point of a vrf scan: four points before
    it and three after, as the old code chose."""
    x = np.asarray(x)
    n = len(x)
    center = int(np.argmax(y_derivative))
    if center > n - 4:
        return x[n - 4], x[-1]
    return x[max(center - 4, 0)], x[center + 3]


def fit_vrf(x, y_derivative, window: tuple[float, float] | None = None) -> np.ndarray:
    """Gaussian fit of the derivative of a vrf scan, returns ``(A, mu, sigma)``."""
    x = np.asarray(x, dtype=np.double)
    y = np.asarray(y_derivative, dtype=np.double)
    if window is None:
        window = vrf_window(x, y)
    sel = (x >= window[0]) & (x <= window[1])
    return aare.Gaussian().fit(x[sel], y[sel])['par']


def fit_scurves(x, data, start=SCURVE_START, mu_limits=SCURVE_MU_LIMITS,
                sigma_limits=SCURVE_SIGMA_LIMITS, minimizer=SCURVE_MINIMIZER,
                n_threads: int = 4) -> np.ndarray:
    """Fit an S-curve to every pixel of ``data[rows, cols, n_scan]``.

    Returns one :data:`SCURVE_DTYPE` record per pixel. Pixels without counts
    are not fitted; they and failed fits have all parameters at zero.
    """
    model = aare.RisingScurve(minimizer=minimizer)
    model.SetParLimits('mu', *mu_limits)
    model.SetParLimits('sigma', *sigma_limits)
    for i, value in enumerate(start):
        model.SetParameter(i, float(value))
    data = _cube(data)
    par = model.fit(np.asarray(x, dtype=np.double), data, n_threads=n_threads)['par']
    par[_empty(data)] = 0.0
    return to_structured(par)


def fit_trimbit_scans(x, data, start=TRIMBIT_START, mu_limits=TRIMBIT_MU_LIMITS,
                      minimizer=TRIMBIT_MINIMIZER, n_threads: int = 4) -> np.ndarray:
    """Fit an S-curve without charge sharing term to every pixel of a trimbit
    scan. Returns the parameter cube ``[rows, cols, 6]`` with ``C`` at zero;
    pixels without counts and failed fits are all zero."""
    model = aare.RisingScurve(minimizer=minimizer)
    model.FixParameter('C', 0.0)
    model.SetParLimits('mu', *mu_limits)
    for i, value in enumerate(start):
        model.SetParameter(i, float(value))
    data = _cube(data)
    par = model.fit(np.asarray(x, dtype=np.double), data, n_threads=n_threads)['par']
    par[_empty(data)] = 0.0
    return par


def solve_trimbits(par: np.ndarray, target, search=TRIMBIT_RANGE, n_grid: int = 200,
                   n_iter: int = 40, chunk: int = 32) -> np.ndarray:
    """Trimbit at which each fitted S-curve reaches ``target``.

    Like ROOT's ``TF1::GetX``, which the old code used: a grid of ``n_grid``
    intervals over ``search`` is scanned from the low end and the first
    interval in which the curve crosses the target is refined by bisection.
    For a monotonic curve this is the only crossing; for the garbage fits of
    bad pixels it picks the same crossing as the old code did.

    NaN where the fit failed or the curve does not reach the target inside
    ``search``. Rows are processed in chunks of ``chunk`` to bound memory.
    """
    target = np.asarray(target, dtype=np.double)
    out = np.full(target.shape, np.nan)
    grid = np.linspace(search[0], search[1], n_grid + 1)
    for r0 in range(0, target.shape[0], chunk):
        rows = slice(r0, r0 + chunk)
        p = [par[rows, :, k] for k in range(6)]
        t = target[rows]
        with np.errstate(all='ignore'):
            g = scurve(grid, *[q[..., None] for q in p]) - t[..., None]
            sign = np.sign(g)
            crossing = (sign[..., :-1] * sign[..., 1:] <= 0) & np.isfinite(g[..., :-1]) & np.isfinite(g[..., 1:])
            found = crossing.any(axis=-1) & (p[3] > 0)
            first = np.argmax(crossing, axis=-1)
            lo = grid[first]
            hi = grid[first + 1]
            f_lo = np.take_along_axis(g, first[..., None], axis=-1)[..., 0]
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                f_mid = scurve(mid, *p) - t
                same = np.sign(f_mid) == np.sign(f_lo)
                lo = np.where(same, mid, lo)
                f_lo = np.where(same, f_mid, f_lo)
                hi = np.where(same, hi, mid)
        result = 0.5 * (lo + hi)
        result[~found] = np.nan
        out[rows] = result
    return out


def interpolate_crossing(x, data, target) -> np.ndarray:
    """Model free trimbit: linear interpolation between the scan points around
    the first one that reaches ``target``.

    NaN where the target is never reached or is reached at the first point.
    """
    x = np.asarray(x, dtype=np.double)
    data = _cube(data)
    target = np.asarray(target, dtype=np.double)
    above = data >= target[:, :, None]
    first = np.argmax(above, axis=2)
    never = ~above.any(axis=2)
    i1 = np.clip(first, 1, None)
    i0 = i1 - 1
    rows, cols = np.indices(target.shape)
    y0 = data[rows, cols, i0]
    y1 = data[rows, cols, i1]
    with np.errstate(all='ignore'):
        tb = x[i0] + (target - y0) / (y1 - y0) * (x[i1] - x[i0])
    tb[never | (first == 0) | ~np.isfinite(tb)] = np.nan
    return tb


def find_trimbits(x, data, target, start=TRIMBIT_START, minimizer=TRIMBIT_MINIMIZER,
                  n_threads: int = 4, fallback: bool = True) -> tuple[np.ndarray, dict]:
    """Fit the trimbit scan and solve for the trimbit of every pixel.

    With ``fallback`` the pixels whose fit gives no trimbit take the
    interpolated crossing of the measured data instead.

    Returns one :data:`TRIMBIT_DTYPE` record per pixel, with ``trimbits`` NaN
    where no value was found, and a dict with the counts of failed fits,
    missing crossings and fallbacks used.
    """
    par = fit_trimbit_scans(x, data, start=start, minimizer=minimizer, n_threads=n_threads)
    tb = solve_trimbits(par, target)
    info = {'fit_failed': int((par[..., 3] == 0).sum()),
            'no_crossing': int(np.isnan(tb).sum()),
            'fallback': 0}
    if fallback:
        missing = np.isnan(tb)
        if missing.any():
            interp = interpolate_crossing(x, data, target)
            use = missing & np.isfinite(interp)
            tb[use] = interp[use]
            info['fallback'] = int(use.sum())
    result = np.zeros(tb.shape, TRIMBIT_DTYPE)
    for i, name in enumerate(SCURVE_PAR):
        result[name] = par[..., i]
    result['trimbits'] = tb
    info['failed'] = int(np.isnan(tb).sum())
    return result, info
