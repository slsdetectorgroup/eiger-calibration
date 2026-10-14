"""The steps that need no detector, run on the T65 crop in a temporary
data directory."""
import numpy as np
import pytest

from eigercal import Calibration, calibration, fit, plot, trimbits
from eigercal.geometry import GEOMETRIES, EigerGeometry


@pytest.fixture
def cal(tmp_path, crop):
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path=tmp_path,
                      plot=False, n_threads=4)
    np.savez(cal.scurve_file, data=crop['scurve_data'], x=crop['scurve_x'])
    np.savez(cal.tbscan_file, data=crop['tb_data'], x=crop['tb_x'])
    return cal


def test_scurve_fit_step_writes_the_fit_file(cal, crop):
    result = calibration.do_scurve_fit(cal)
    assert cal.fit_file.exists()
    saved = calibration.load_fit_result(cal)
    assert saved.dtype == fit.SCURVE_DTYPE
    assert np.array_equal(saved, result)
    assert saved.shape == crop['scurve_data'].shape[:2]


def test_find_trimbits_step(cal, crop):
    np.save(cal.fit_file, crop['scurve_fit'])
    tb, target, data, x, result = calibration.find_trimbits(cal)
    assert tb.dtype == np.int32 and tb.shape == (64, 64)
    assert tb.min() >= 0 and tb.max() <= 63
    saved_tb, saved_fit = calibration.load_trimbit_result(cal)
    assert np.array_equal(saved_tb, tb)
    assert saved_fit.dtype == fit.TRIMBIT_DTYPE
    assert np.array_equal(target, fit.scurve_at_mu(crop['scurve_fit']))


def test_pixelmask_zeroes_pixels(cal, crop):
    mask = np.zeros((64, 64), dtype=bool)
    mask[3, 7] = True
    result = calibration.do_scurve_fit(cal, pixelmask=mask)
    assert all(result[name][3, 7] == 0 for name in fit.SCURVE_PAR)


def test_write_trimbit_files(tmp_path):
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path=tmp_path)
    rng = np.random.default_rng(3)
    tb = rng.integers(0, 64, (512, 1024))
    dacs = rng.integers(0, 4000, (17, 2))
    paths = calibration.write_trimbit_files(cal, tb, dacs, ['beb003', 'beb043'])
    assert [p.name for p in paths] == ['T65_CuXRF_gain5.sn003', 'T65_CuXRF_gain5.sn043']
    top, top_dacs = trimbits.read_trimbit_file(paths[0])
    bottom, bottom_dacs = trimbits.read_trimbit_file(paths[1])
    assert np.array_equal(top, tb[256:512])
    assert np.array_equal(bottom, tb[0:256])
    assert top_dacs.tolist() == dacs[:, 0].tolist() + [0]
    assert bottom_dacs.tolist() == dacs[:, 1].tolist() + [0]


def test_write_trimbit_files_needs_one_hostname_per_halfmodule(tmp_path):
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path=tmp_path)
    with pytest.raises(ValueError):
        calibration.write_trimbit_files(cal, np.zeros((512, 1024)), np.zeros((17, 2)), ['beb003'])


def test_clean_vrf_data_zeroes_hot_pixels():
    data = np.full((4, 4, 3), 100.0)
    data[1, 1, 2] = 1000.0
    cleaned = calibration.clean_vrf_data(data)
    assert (cleaned[1, 1] == 0).all()
    assert (cleaned[0, 0] == 100).all()
    assert (data[1, 1] != 0).all()


def test_fit_vrf_scan_per_halfmodule(crop):
    geometry = EigerGeometry('test', 64, 64, (), ((slice(32, 64), slice(0, 64)), (slice(0, 32), slice(0, 64))), ())
    data = np.zeros((64, 64, crop['vrf_x'].size))
    for i, hm in enumerate(geometry.halfmodule):
        data[hm] = crop['vrf_y'][i] / (32 * 64)
    vrf, pars = calibration.fit_vrf_scan(crop['vrf_x'], data, geometry)
    assert vrf == crop['vrf'].tolist()
    fig = plot.vrf_scan(crop['vrf_x'], data, geometry, pars, ['beb003', 'beb043'], 'test')
    assert fig is not None


def test_plots_run_on_a_500k_image():
    g = GEOMETRIES['500k']
    rng = np.random.default_rng(4)
    mu = rng.normal(1200, 50, g.shape)
    mean, std, fig = plot.chip_histograms(mu, g)
    assert len(mean) == 8 and all(abs(m - 1200) < 10 for m in mean)
    plot.image(mu, clim=(1000, 1400))
    plot.trimbit_histogram(rng.integers(0, 64, g.shape))
    plot.scurve_scan(np.arange(5), rng.uniform(size=(*g.shape, 5)), g, 'test')
    plot.trimbit_scan(np.arange(5), rng.uniform(size=(*g.shape, 5)))
