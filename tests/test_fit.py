import numpy as np
import pytest

from eigercal import fit, trimbits


def test_vrf_fit_matches_root(crop):
    for i in range(2):
        par = fit.fit_vrf(crop['vrf_x'], np.gradient(crop['vrf_y'][i]))
        assert int(np.round(par[1])) == crop['vrf'][i]


def test_scurve_at_mu_is_the_model_at_mu():
    rng = np.random.default_rng(1)
    f = np.zeros((3, 4), fit.SCURVE_DTYPE)
    for name in fit.SCURVE_PAR:
        f[name] = rng.uniform(0.5, 2.0, f.shape)
    expected = fit.scurve(f['mu'], f['p0'], f['p1'], f['mu'], f['sigma'], f['A'], f['C'])
    assert np.allclose(fit.scurve_at_mu(f), expected)


def test_scale_scurve_data_zeroes_pixels_without_plateau():
    data = np.ones((2, 2, 5))
    data[0, 0, :] = 0
    scaled = fit.scale_scurve_data(data)
    assert (scaled[0, 0] == 0).all()
    assert (scaled[1, 1] == 1000).all()


def test_scurve_fit_matches_root(crop):
    ref = crop['scurve_fit']
    data = fit.scale_scurve_data(crop['scurve_data'])
    result = fit.fit_scurves(crop['scurve_x'], data, n_threads=4)
    assert result.dtype == fit.SCURVE_DTYPE
    good = (ref['mu'] > 100) & (ref['mu'] < 1900)
    failed = result['sigma'] == 0
    assert (failed & good).sum() <= 0.002 * good.sum()
    dmu = np.abs(result['mu'] - ref['mu'])[good & ~failed]
    assert np.median(dmu) < 0.01
    assert np.percentile(dmu, 99) < 1.0
    assert (dmu > 5).mean() < 0.005


def test_scurve_fit_of_empty_pixel_is_zero(crop):
    data = fit.scale_scurve_data(crop['scurve_data'][:2, :2])
    data[0, 0] = 0
    result = fit.fit_scurves(crop['scurve_x'], data, n_threads=1)
    assert all(result[name][0, 0] == 0 for name in fit.SCURVE_PAR)


def test_round_trip_structured_cube():
    rng = np.random.default_rng(2)
    cube = rng.uniform(size=(3, 5, 6))
    assert np.array_equal(fit.to_cube(fit.to_structured(cube)), cube)


def test_solve_trimbits_recovers_known_crossing():
    par = np.zeros((2, 3, 6))
    par[..., 0] = 10      # p0
    par[..., 1] = 2       # p1
    par[..., 2] = 30      # mu
    par[..., 3] = 12      # sigma
    par[..., 4] = 500     # A
    tb_true = np.array([[5.0, 20.0, 40.5], [0.0, 63.0, 70.0]])
    target = fit.scurve(tb_true, *[par[..., k] for k in range(6)])
    tb = fit.solve_trimbits(par, target)
    assert np.allclose(tb, tb_true, atol=1e-6)


def test_solve_trimbits_nan_when_failed_or_out_of_range():
    par = np.zeros((1, 2, 6))
    par[0, 0] = [10, 2, 30, 12, 500, 0]
    par[0, 1] = [10, 2, 30, 0, 500, 0]      # failed fit, sigma 0
    target = np.array([[1e6, 300.0]])        # never reached / failed fit
    tb = fit.solve_trimbits(par, target)
    assert np.isnan(tb).all()


def test_interpolate_crossing():
    x = np.arange(0, 64, 2, dtype=float)
    data = np.zeros((1, 3, x.size))
    data[0, 0] = 10 * x                      # crosses 150 at x=15
    data[0, 1] = 1000                        # above target already at x=0
    data[0, 2] = 1                           # never reaches the target
    tb = fit.interpolate_crossing(x, data, np.array([[150.0, 150.0, 150.0]]))
    assert tb[0, 0] == pytest.approx(15.0)
    assert np.isnan(tb[0, 1]) and np.isnan(tb[0, 2])


def test_trimbits_match_root(crop):
    ref = crop['tb_fit']
    data = fit.scale_trimbit_data(crop['tb_data'], crop['scurve_data'][:, :, -1])
    target = fit.scurve_at_mu(crop['scurve_fit'])
    result, info = fit.find_trimbits(crop['tb_x'], data, target, n_threads=4, fallback=False)
    assert result.dtype == fit.TRIMBIT_DTYPE
    # The old code overwrote NaN trimbits with 32 in place before saving
    root_ok = (ref['trimbits'] != 32.0) & (ref['sigma'] != 0)
    both = root_ok & np.isfinite(result['trimbits'])
    assert both.sum() > 0.97 * root_ok.sum()
    dtb = np.abs(result['trimbits'] - ref['trimbits'])[both]
    assert np.median(dtb) < 0.05
    tb = trimbits.clip_trimbits(result['trimbits'])
    assert (tb[both] == crop['trimbits'][both]).mean() > 0.9


def test_find_trimbits_fallback_fills_missing(crop):
    data = fit.scale_trimbit_data(crop['tb_data'], crop['scurve_data'][:, :, -1])
    target = fit.scurve_at_mu(crop['scurve_fit'])
    without, info_without = fit.find_trimbits(crop['tb_x'], data, target, n_threads=4, fallback=False)
    with_, info_with = fit.find_trimbits(crop['tb_x'], data, target, n_threads=4, fallback=True)
    assert info_with['failed'] <= info_without['failed']
    assert info_with['fallback'] == info_without['failed'] - info_with['failed']
    filled = np.isnan(without['trimbits']) & np.isfinite(with_['trimbits'])
    assert filled.sum() == info_with['fallback']
