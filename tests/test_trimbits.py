import numpy as np
import pytest

from eigercal import trimbits


def test_write_and_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    tb = rng.integers(0, 64, (256, 1024))
    dacs = rng.integers(0, 4000, 18)
    fname = tmp_path / 'test.sn003'
    trimbits.write_trimbit_file(fname, tb, dacs)
    assert fname.stat().st_size == 4 * (256 * 1024 + 18)
    tb_read, dacs_read = trimbits.read_trimbit_file(fname)
    assert (tb_read == tb).all()
    assert (dacs_read == dacs).all()
    assert tb_read.dtype == np.int32


def test_write_rejects_wrong_sizes(tmp_path):
    with pytest.raises(ValueError):
        trimbits.write_trimbit_file(tmp_path / 'a', np.zeros((256, 1024)), np.zeros(17))
    with pytest.raises(ValueError):
        trimbits.write_trimbit_file(tmp_path / 'b', np.zeros((512, 1024)), np.zeros(18))


def test_read_rejects_wrong_file_size(tmp_path):
    fname = tmp_path / 'short.sn003'
    np.zeros(100, dtype=np.int32).tofile(fname)
    with pytest.raises(ValueError):
        trimbits.read_trimbit_file(fname)


def test_file_dacs_appends_tau():
    detector_dacs = np.arange(34).reshape(17, 2)
    dacs = trimbits.file_dacs(detector_dacs)
    assert dacs.shape == (18, 2)
    assert (dacs[:17] == detector_dacs).all()
    assert (dacs[17] == 0).all()


def test_clip_trimbits():
    tb = trimbits.clip_trimbits(np.array([np.nan, 70.0, -3.0, 12.6, 63.4, 0.49]))
    assert tb.dtype == np.int32
    assert tb.tolist() == [32, 63, 0, 13, 63, 0]
