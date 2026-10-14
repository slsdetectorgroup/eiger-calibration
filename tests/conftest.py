from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

DATA = Path(__file__).parent / 'data' / 't65_gain5_crop.npz'


@pytest.fixture(scope='session')
def crop():
    """64x64 pixel crop of the T65 gain5 Cu calibration with the results of
    the ROOT based fits of sls_detector_tools."""
    with np.load(DATA) as f:
        return {k: f[k] for k in f.files}
