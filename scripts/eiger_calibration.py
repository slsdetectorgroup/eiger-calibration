# -*- coding: utf-8 -*-
"""
Calibrate one gain of an EIGER module in the big X-ray box.

Every step saves its result in the data directory, so a step can be skipped
by commenting it out and its result loaded instead.

Calibration plan::

    Name      vcmp       target
    gain0     1500       In 3.3keV
    gain2     1200       Ti 4.5keV
    gain3     1200       Cr 5.4keV
    gain4     1200       Fe 6.4keV
    gain5     1200       Cu 8.0keV
    gain6     1200       Ge 9.9keV
    gain7     1200       Zr
    gain8                Mo
    gain9                Ag
    gain10               Sn
"""
from pathlib import Path

import matplotlib.pyplot as plt
from slsdet import Eiger
from slsdetbox import BigXrayBox

from eigercal import Calibration, calibration, scans, setup_logging

# ---------------------------------------------------------- Configuration
cal = Calibration(
    det_id='T65',
    geometry='500k',       # 500k, 1M, 2M, 9M
    gain='gain5',
    target='Cu',
    run_id=0,
    data_path=Path('/mnt/ssd/calibration') / 'T65' / 'gain5',
    threshold=1200,
    vrf_scan_exptime=0.1,
    vtr=2400,
    zmq_host='mpc2932',
)
setup_logging(cal)

# ---------------------------------------------------------- X-ray box and detector
box = BigXrayBox()
d = Eiger()
scans.setup_detector(d, cal)
d.parallel = False
d.dacs.vtrim = cal.vtr
d.vthreshold = cal.threshold

# ---------------------------------------------------------- vrf scan
vrf, t, counts = calibration.do_vrf_scan(d, box, cal, start=2500, stop=3700)
# vrf = calibration.load_vrpreamp(cal)
d.dacs.vrpreamp = vrf
cal.exptime = 300
# cal.exptime = t

# ---------------------------------------------------------- Threshold scan and fit
data, x = calibration.do_scurve(d, box, cal)
fit_result = calibration.do_scurve_fit(cal)
# fit_result = calibration.load_fit_result(cal)

# ---------------------------------------------------------- Trimbit scan and trimbits
data, x = calibration.do_trimbit_scan(d, box, cal)
tb, target, data, x, result = calibration.find_and_write_trimbits(d, cal)
calibration.load_trimbits(d, cal)

# ---------------------------------------------------------- Checks
# One image with the trimmed module
# img = calibration.acquire_image(d, box, cal)
# fig, ax = plt.subplots()
# im = ax.imshow(img)
# im.set_clim(0, 500)
# plt.show()

# Threshold scan of the trimmed module
# cal.run_id = 1
# data, x = calibration.do_scurve(d, box, cal)
# fit_result = calibration.do_scurve_fit(cal)
