"""Settings and file names for one calibration run."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Fluorescence energies in keV of the targets in the big X-ray box.
XRF_ENERGY = {
    'In': 3.29, 'Sc': 4.1, 'Ti': 4.5, 'Cr': 5.4, 'Fe': 6.4, 'Cu': 8.00,
    'Ge': 9.9, 'Se': 11.22, 'Zr': 15.8, 'Mo': 17.5, 'Ag': 22.2, 'Sn': 25.3,
}


def _default_threads() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class Calibration:
    """Settings for calibrating one gain of one detector.

    The file names follow the convention of ``sls_detector_tools`` so that
    data directories written by the old package can be read by this one:

    ===================  =========================================
    vrf scan             ``{det_id}_vrf_{target}XRF_{run_id}.npz``
    threshold scan       ``{det_id}_vcmp_{target}XRF_{run_id}.npz``
    S-curve fit          ``{det_id}_vcmp_{target}XRF_{run_id}_fit.npy``
    trimbit scan         ``{det_id}_tbscan_{target}XRF_{run_id}.npz``
    trimbit result       ``{det_id}_{target}XRF_{gain}.npz``
    trimbit files        ``{det_id}_{target}XRF_{gain}.sn{beb}``
    ===================  =========================================
    """

    det_id: str
    gain: str
    target: str
    data_path: Path
    geometry: str = '500k'
    run_id: int = 0

    # Acquisition
    exptime: float = 3.0
    vrf_scan_exptime: float = 0.1
    nframes: int = 1
    period: float = 0.0
    dynamic_range: int = 32

    # DACs applied before scanning
    vtr: int = 2500
    vrs: int = 1600
    threshold: int = 1200
    #: Trimbit value during the threshold scan; trimming is centred on it
    trimval: int = 32

    # Processing
    n_threads: int = field(default_factory=_default_threads)
    plot: bool = True

    #: Host running the receivers, used for the ZMQ stream
    zmq_host: str = 'localhost'

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        if self.target not in XRF_ENERGY:
            raise ValueError(f'Unknown target: {self.target}')

    @property
    def energy(self) -> float:
        """Fluorescence energy of the target in keV."""
        return XRF_ENERGY[self.target]

    def _base(self, kind: str) -> Path:
        return self.data_path / f'{self.det_id}_{kind}_{self.target}XRF_{self.run_id}'

    @property
    def vrf_file(self) -> Path:
        return Path(f'{self._base("vrf")}.npz')

    @property
    def scurve_file(self) -> Path:
        return Path(f'{self._base("vcmp")}.npz')

    @property
    def fit_file(self) -> Path:
        return Path(f'{self._base("vcmp")}_fit.npy')

    @property
    def tbscan_file(self) -> Path:
        return Path(f'{self._base("tbscan")}.npz')

    @property
    def trimbit_base(self) -> Path:
        """Trimbit file name without extension, as passed to ``detector.trimbits``."""
        return self.data_path / f'{self.det_id}_{self.target}XRF_{self.gain}'

    @property
    def trimbit_file(self) -> Path:
        return Path(f'{self.trimbit_base}.npz')

    def trimbit_file_for(self, hostname: str) -> Path:
        """Trimbit file of one half module, named after its beb number."""
        return Path(f'{self.trimbit_base}.sn{hostname[3:]}')

    def png(self, kind: str, suffix: str = '') -> Path:
        """Path of a plot belonging to the ``vrf``, ``vcmp`` or ``tbscan`` data."""
        return Path(f'{self._base(kind)}{suffix}.png')


def setup_logging(cal: Calibration, fname: str = 'calibration.log',
                  stream: bool = True, level: int = logging.INFO) -> logging.Logger:
    """Log to ``fname`` in the data directory and optionally to stdout."""
    cal.data_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s:%(funcName)s() %(message)s')
    handler = logging.FileHandler(cal.data_path / fname)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.info('New measurement: %s %s %s run %d', cal.det_id, cal.gain,
                cal.target, cal.run_id)
    return logger
