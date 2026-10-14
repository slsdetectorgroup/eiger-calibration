# Moving the Eiger calibration out of sls_detector_tools

Status as of 2026-10-14. The offline part of the port is done and tested
against the ROOT results; the detector side is ported but has not yet run on
hardware.

## Why

`sls_detector_tools` fits the calibration scans in a compiled extension that
links ROOT, which ties the calibration to a ROOT conda environment and to a
build step. aare has the same S-curve model with four minimizers and a
threaded per-pixel fit, so the calibration can run in any environment that
has `slsdet`, `aare` and `pyzmq`.

## What was ported

Only the path used by `scripts/eiger_calibration.py` of `sls_detector_tools`
for an XRF calibration was moved:

| Old | New |
|---|---|
| `config` module globals | `eigercal.config.Calibration` dataclass, file names as properties |
| `mask` geometry classes | `eigercal.geometry` (500k, 1M, 2M, 9M) |
| `receiver.ZmqReceiver`, `calibration.setup_measurement` | `eigercal.receiver` |
| `calibration.setup_detector`, `_vrf_scan`, `_threshold_scan`, `_trimbit_scan`, `find_mean_and_set_vcmp`, `load_trimbits` | `eigercal.scans` |
| `_sls_cmodule.vrf_fit`, `mpfit.fit`, `mpfit.find_trimbits`, `function.scurve` | `eigercal.fit` on top of aare |
| `io.write_trimbit_file`, `io.read_trimbit_file` | `eigercal.trimbits` |
| `plot.imshow`, `plot.histogram`, `plot.chip_histograms`, the scan plots | `eigercal.plot` |
| `calibration.do_vrf_scan`, `do_scurve`, `do_scurve_fit_scaled`, `do_trimbit_scan`, `find_and_write_trimbits_scaled` | `eigercal.calibration` |

Dropped: everything else in `calibration.py` (unscaled fits, global
calibration, reports, rewriting and copying of calibration files, mask
generation), the test pulse and beam calibration types, `root_helper`,
`mpfit`, the ROOT histogram functions, and the 250k, 1.5M and 1.5M OMNY
geometries, whose definitions in `mask.py` were inconsistent. The 9M script
stays on the old package.

File names and the record layout of the saved fits are unchanged, so old
data directories load with the new package and the reverse.

## How the fits were replaced

Checked on the T65 gain5 Cu calibration of 2026-10-14 (512 x 1024 pixels),
whose ROOT results were on disk. Rerunning the old fit reproduced the saved
reference exactly, so the numbers compare against the production code.

| Step | Old | New | Result |
|---|---|---|---|
| vrf | ROOT `gaus` on the 8 points around the steepest point | `aare.Gaussian` on the same points | 2957 / 2956, identical |
| S-curve per pixel | 64 processes, 8.9 s | `RisingScurve`, VarPro, limits mu 0..2000 and sigma 50..300, old start values; 0.27 s on 32 threads | median difference in mu 0.0004 DAC; 2 failed fits among 521k good pixels; 0.1 % of pixels differ by more than 5 DAC, with chi2 better on either side |
| trimbit per pixel | per-pixel start values and limits for p0 and A, then `TF1::GetX` | `RisingScurve`, Levenberg-Marquardt, C fixed, mu limited to 0..64, old start values; then the first crossing of the target on [-10, 74], found as `TF1::GetX` did; 0.4 s plus 4 s | integer trimbits identical to ROOT on 99.0 % of the pixels, 99.2 % within one; distance to the measured crossing of the target: median 0.199 for both, more than 1 trimbit off: ROOT 4258 pixels, aare 4053 |

Notes from the comparison:

* VarPro is the right minimizer for the S-curves but not for the trimbits.
  Without the per-pixel limits of the old code its exact solution of the
  linear parameters wanders on noisy pixels, and its 95th percentile is
  worse than Levenberg-Marquardt. Migrad fails on 56k trimbit pixels with
  the global start vector.
* The target count is the S-curve at its own inflection point, which is
  `p0 + p1 * mu + A / 2`. No error function is needed for it.
* Pixels without counts are not fitted and get zeros, as before.
* The old code overwrote the NaN trimbits with 32 inside the array it then
  saved, so the saved fit does not show which pixels failed. The new code
  keeps NaN in the saved fit and only fills 32 in the integer trimbits.
* The trimbit is the first crossing of the target from the low end of the
  search range, as `TF1::GetX` returned it. The bottom half module of T65
  has 22k pixels whose counts are above the target already at trimbit 0 and
  dip in the middle of the scan. Both fits return the same garbage curve
  there, which crosses the target more than once, and only this rule makes
  the reported trimbits agree. A plain bisection picked another crossing on
  those pixels and agreed with ROOT on 90 % of that half module instead of
  98 %.
* Pixels whose fitted curve never reaches the target fall back to the
  interpolated crossing of the measured data instead of a flat 32. On T65
  that recovers 741 of the 40k pixels without a value; the others never
  reach the target within the scan.
* On the bottom half module of T65, where 65k pixels never reach the
  target, about 2100 pixels remain on which the old fit lands closer to the
  measured crossing than the new one, against about 200 the other way. The
  per-pixel start values and limits of the old fit are the likely reason.
  Global non-negativity limits on p0 and A, a sigma limit, and normalising
  each pixel by its plateau to turn the per-pixel limits into global ones
  were tried; none helped, and the normalisation made the healthy half
  module worse. On the healthy top half module the new fit is closer to
  the data more often than the old one.

## Steps

1. Repository skeleton, `pyproject.toml`, README. Done.
2. Offline core: config, geometry, fit, trimbit files, the offline
   calibration steps, with regression tests against a 64 x 64 crop of the
   T65 data in `tests/data`. Done, 39 tests pass. The full T65 module was
   also run through the offline steps in a scratch directory: the S-curve
   step takes 0.7 s and the trimbit step 5 s, and the trimbit files have
   the same dac header as the old ones.
3. Detector side: receiver, scans, chip thresholds. Ported, not run on
   hardware.
4. Plots and the step functions, with the old file names. Done.
5. `scripts/eiger_calibration.py` with the same stage sequence. Done.
6. First run on a module with the ROOT results still on disk, side by side.
   Open.

## Open points for the first run on hardware

* The receiver flips a port when the `flipRows` flag in the ZMQ header is
  set. The old receiver flipped the ports of every bottom half module
  unconditionally; `ZmqReceiver(..., flip_bottom=True)` restores that if the
  header flag turns out not to be set for this receiver version.
* Only the 500k geometry has been validated. The chip threshold setting is
  one generic loop over modules now instead of a branch per geometry.
* aare is not installed in the `root-env` environment yet; the tests were
  run with aare's local build on `PYTHONPATH`.
* `root-env` has no pytest.
* If the remaining difference on bad pixels matters, aare would need
  per-pixel start values or limits for the trimbit fit.
