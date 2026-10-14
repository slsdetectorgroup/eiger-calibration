"""Receive one frame per acquisition from the ZMQ streams of the receivers."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

import numpy as np
import zmq

from .geometry import EigerGeometry

log = logging.getLogger(__name__)


def dtype_for(bitmode: int) -> np.dtype:
    """NumPy dtype that holds a pixel of the given dynamic range."""
    if bitmode == 32:
        return np.dtype(np.int32)
    if bitmode == 16:
        return np.dtype(np.int16)
    if bitmode in (8, 4):
        return np.dtype(np.uint8)
    raise ValueError(f'Unsupported bitmode: {bitmode}')


def decode(data: bytes, bitmode: int) -> np.ndarray:
    """Pixels of one port as a flat array, unpacking 4 bit data."""
    raw = np.frombuffer(data, dtype=dtype_for(bitmode))
    if bitmode == 4:
        pixels = np.empty(raw.size * 2, dtype=np.uint8)
        pixels[0::2] = raw & 0x0F
        pixels[1::2] = raw >> 4
        return pixels
    return raw


def _inside(inner, outer) -> bool:
    return all(o.start <= i.start and i.stop <= o.stop for i, o in zip(inner, outer))


class ZmqReceiver:
    """Assemble the frames of all ports of a detector into one image.

    Each port streams a JSON header, the data and a JSON end-of-acquisition
    message. Only single frame acquisitions are supported.

    The rows of the bottom half module of each module come out of the chips
    in reverse order. By default a port is flipped when the ``flipRows`` flag
    of its header is set. ``flip_bottom=True`` flips the ports of every bottom
    half module regardless of the header, which is what ``sls_detector_tools``
    did, and ``flip_bottom=False`` never flips.
    """

    def __init__(self, host: str, detector, geometry: EigerGeometry,
                 flip_bottom: bool | None = None, hwm: int = 1000):
        ports = list(detector.rx_zmqport)
        if len(ports) != len(geometry.port):
            raise ValueError(f'{len(ports)} zmq ports for a geometry with {len(geometry.port)} ports')
        size = detector.detsize
        if (size.y, size.x) != geometry.shape:
            raise ValueError(f'Detector size {(size.y, size.x)} does not match geometry {geometry.shape}')
        self.geometry = geometry
        self.shape = geometry.shape
        self.flip_bottom = flip_bottom
        self.bottom_ports = {
            i for i, port in enumerate(geometry.port)
            if any(_inside(port, hm) for hm in geometry.halfmodule[1::2])
        }
        self.context = zmq.Context()
        self.sockets = []
        for port in ports:
            socket = self.context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, hwm)
            endpoint = f'tcp://{host}:{port}'
            log.info('Connecting to %s', endpoint)
            socket.connect(endpoint)
            socket.setsockopt(zmq.SUBSCRIBE, b'')
            self.sockets.append(socket)

    def get_frame(self) -> np.ndarray:
        """Read one frame from every port and return the assembled image."""
        image = np.zeros(self.shape)
        for i, (region, socket) in enumerate(zip(self.geometry.port, self.sockets)):
            header = json.loads(socket.recv())
            data = socket.recv()
            json.loads(socket.recv())  # end of acquisition
            rows = region[0].stop - region[0].start
            cols = region[1].stop - region[1].start
            block = decode(data, header['bitmode']).reshape(rows, cols)
            if self.flip_bottom is None:
                flip = bool(header.get('flipRows', 0))
            else:
                flip = self.flip_bottom and i in self.bottom_ports
            image[region] = block[::-1, :] if flip else block
        return image

    def close(self) -> None:
        for socket in self.sockets:
            socket.close()
        self.context.term()


@contextmanager
def setup_measurement(detector, host: str, geometry: EigerGeometry, flip_bottom: bool | None = None):
    """Restart the ZMQ stream, disable file writing and yield a receiver.

    The dacs are saved on entry and restored on exit. ::

        with setup_measurement(detector, 'mpc2932', geometry) as receiver:
            detector.acquire()
            image = receiver.get_frame()
    """
    detector.rx_zmqstream = False
    time.sleep(1)
    detector.rx_zmqstream = True
    detector.fwrite = False
    dacs = detector.dacs.get_asarray()
    receiver = ZmqReceiver(host, detector, geometry, flip_bottom=flip_bottom)
    try:
        yield receiver
    finally:
        receiver.close()
        detector.dacs.set_from_array(dacs)
