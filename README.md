# eiger-calibration

Calibration of EIGER modules with X-ray fluorescence, driven from Python.
The `eigercal` package holds the steps; `scripts/eiger_calibration.py` runs
them for one gain of one detector.

The per-pixel fits are done by [aare](https://github.com/slsdetectorgroup/aare).
This replaces the ROOT based extension of `sls_detector_tools`, from which
this package was extracted; see `PLAN.md` for what was ported and how the
fits were checked against the old ones.

## Environment

The package needs `numpy`, `scipy`, `matplotlib`, `pyzmq` and `aare`
(2026.9.30 or newer, on PyPI and on the `slsdetectorgroup` conda channel).
Running a calibration also needs `slsdet` (conda, `slsdetectorgroup`
channel) and `slsdetbox` for the X-ray box, both only available from conda.

```bash
conda activate root-env            # or any env with slsdet and slsdetbox
conda install -c slsdetectorgroup aare
pip install -e .
```

To use a local build of aare instead, put its build directory on
`PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/aare/build
```

## Running a calibration

Edit the configuration block at the top of `scripts/eiger_calibration.py`
and run it. Every step writes its result to the data directory, so a step can
be commented out and its result loaded instead, for example to refit a saved
scan:

```python
from eigercal import Calibration, calibration

cal = Calibration(det_id='T65', gain='gain5', target='Cu', data_path='/mnt/ssd/calibration/T65/gain5')
fit_result = calibration.do_scurve_fit(cal)
tb, target, data, x, result = calibration.find_trimbits(cal)
```

The steps are:

| Step | Function | Writes |
|---|---|---|
| vrf scan and fit | `calibration.do_vrf_scan` | `{det}_vrf_{target}XRF_{run}.npz` |
| threshold scan | `calibration.do_scurve` | `{det}_vcmp_{target}XRF_{run}.npz` |
| S-curve fit per pixel | `calibration.do_scurve_fit` | `{det}_vcmp_{target}XRF_{run}_fit.npy` |
| set chip thresholds, trimbit scan | `calibration.do_trimbit_scan` | `{det}_tbscan_{target}XRF_{run}.npz` |
| trimbit fit, trimbit files | `calibration.find_and_write_trimbits` | `{det}_{target}XRF_{gain}.npz`, `.sn{beb}` |

The file names are those of `sls_detector_tools`, and the saved fits use the
same record layout, so data directories from the old package can be loaded.

## Tests

```bash
python -m pytest
```

The fit tests compare against the ROOT fits of a 64x64 pixel crop of the
T65 gain5 Cu calibration in `tests/data`. The detector and receiver code is
not covered by the tests.
