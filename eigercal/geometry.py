"""Pixel layout of Eiger systems as numpy slices.

All slices are ``(rows, cols)`` tuples into the assembled image. Half modules
are listed top first within each module, which matches the order of
``detector.hostname`` for a module configured as ``[top, bottom]``.

Only the 500k geometry has been validated after the move from
``sls_detector_tools``; the others are carried over as data.
"""
from __future__ import annotations

from dataclasses import dataclass

Slice2D = tuple[slice, slice]

#: Chips of one module in module coordinates: the top row of chips is at
#: rows 256:512, listed left to right, then the bottom row.
CHIPS: tuple[Slice2D, ...] = tuple(
    (r, c)
    for r in (slice(256, 512), slice(0, 256))
    for c in (slice(0, 256), slice(256, 512), slice(512, 768), slice(768, 1024))
)

MODULE_SHAPE = (512, 1024)
HALFMODULE_SHAPE = (256, 1024)

_VCMP_TOP = ('vcmp_ll', 'vcmp_lr', 'vcmp_rl', 'vcmp_rr')
_VCMP_BOTTOM = _VCMP_TOP[::-1]


@dataclass(frozen=True)
class EigerGeometry:
    name: str
    nrow: int
    ncol: int
    #: Full modules (512 x 1024)
    module: tuple[Slice2D, ...]
    #: Half modules (256 x 1024), top of each module first
    halfmodule: tuple[Slice2D, ...]
    #: UDP/ZMQ ports (256 x 512), in the order of ``detector.rx_zmqport``
    port: tuple[Slice2D, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrow, self.ncol)

    @property
    def n_modules(self) -> int:
        return len(self.module)

    @property
    def n_halfmodules(self) -> int:
        return len(self.halfmodule)

    @property
    def vcmp_names(self) -> list[str]:
        """DAC names ``'{halfmodule}:vcmp_xx'`` in the order of the chips in
        :func:`chips` of each module, which is the order ``detector.vcmp`` uses."""
        names = []
        for j in range(self.n_modules):
            names += [f'{2 * j}:{v}' for v in _VCMP_TOP]
            names += [f'{2 * j + 1}:{v}' for v in _VCMP_BOTTOM]
        return names

    def chips(self, module: Slice2D) -> tuple[Slice2D, ...]:
        """The eight chips of ``module`` as slices into the full image."""
        r0, c0 = module[0].start, module[1].start
        return tuple(
            (slice(r0 + c[0].start, r0 + c[0].stop), slice(c0 + c[1].start, c0 + c[1].stop))
            for c in CHIPS
        )


def _vertical_stack(name: str, n_modules: int) -> EigerGeometry:
    """Modules stacked vertically, module 0 at the top of the image."""
    n_half = 2 * n_modules
    port = []
    for row in range(n_half, 0, -1):
        rows = slice(256 * (row - 1), 256 * row)
        port.append((rows, slice(0, 512)))
        port.append((rows, slice(512, 1024)))
    halfmodule = tuple((slice(256 * (i - 1), 256 * i), slice(0, 1024)) for i in range(n_half, 0, -1))
    module = tuple((slice(512 * (i - 1), 512 * i), slice(0, 1024)) for i in range(n_modules, 0, -1))
    return EigerGeometry(name, 512 * n_modules, 1024, module, halfmodule, tuple(port))


def _eiger9M() -> EigerGeometry:
    """cSAXS 9M: three columns of six modules."""
    port = []
    for col in range(3):
        for row in range(12, 0, -1):
            rows = slice(256 * (row - 1), 256 * row)
            port.append((rows, slice(1024 * col, 1024 * col + 512)))
            port.append((rows, slice(1024 * col + 512, 1024 * (col + 1))))
    halfmodule = tuple(
        (slice(256 * (i - 1), 256 * i), slice(1024 * c, 1024 * (c + 1)))
        for c in range(3) for i in range(12, 0, -1)
    )
    module = tuple(
        (slice(512 * (i - 1), 512 * i), slice(1024 * c, 1024 * (c + 1)))
        for c in range(3) for i in range(6, 0, -1)
    )
    return EigerGeometry('9M', 3072, 3072, module, halfmodule, tuple(port))


GEOMETRIES: dict[str, EigerGeometry] = {
    '500k': _vertical_stack('500k', 1),
    '1M': _vertical_stack('1M', 2),
    '2M': _vertical_stack('2M', 4),
    '9M': _eiger9M(),
}
