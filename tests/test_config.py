from pathlib import Path

import pytest

from eigercal import Calibration


def test_file_names_follow_the_old_convention():
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path='/data/T65/gain5')
    assert cal.vrf_file == Path('/data/T65/gain5/T65_vrf_CuXRF_0.npz')
    assert cal.scurve_file.name == 'T65_vcmp_CuXRF_0.npz'
    assert cal.fit_file.name == 'T65_vcmp_CuXRF_0_fit.npy'
    assert cal.tbscan_file.name == 'T65_tbscan_CuXRF_0.npz'
    assert cal.trimbit_base == Path('/data/T65/gain5/T65_CuXRF_gain5')
    assert cal.trimbit_file.name == 'T65_CuXRF_gain5.npz'
    assert cal.trimbit_file_for('beb003').name == 'T65_CuXRF_gain5.sn003'
    assert cal.png('tbscan', '_hist').name == 'T65_tbscan_CuXRF_0_hist.png'
    assert cal.png('vrf').name == 'T65_vrf_CuXRF_0.png'


def test_run_id_is_part_of_the_scan_names():
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path='/data', run_id=1)
    assert cal.scurve_file.name == 'T65_vcmp_CuXRF_1.npz'
    assert cal.trimbit_file.name == 'T65_CuXRF_gain5.npz'


def test_dots_in_det_id_survive():
    cal = Calibration(det_id='1.5M', gain='gain5', target='Cu', data_path='/data')
    assert cal.trimbit_file.name == '1.5M_CuXRF_gain5.npz'
    assert cal.fit_file.name == '1.5M_vcmp_CuXRF_0_fit.npy'


def test_energy_from_target():
    cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path='/data')
    assert cal.energy == 8.0


def test_unknown_target():
    with pytest.raises(ValueError):
        Calibration(det_id='T65', gain='gain5', target='Xx', data_path='/data')
