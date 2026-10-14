import numpy as np
import pytest

from eigercal.geometry import CHIPS, GEOMETRIES


def coverage(shape, regions):
    count = np.zeros(shape, dtype=int)
    for region in regions:
        count[region] += 1
    return count


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_ports_halfmodules_and_modules_tile_the_image(name):
    g = GEOMETRIES[name]
    for regions in (g.port, g.halfmodule, g.module):
        assert (coverage(g.shape, regions) == 1).all()
    assert g.n_halfmodules == 2 * g.n_modules
    assert len(g.port) == 2 * g.n_halfmodules


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_chips_tile_each_module(name):
    g = GEOMETRIES[name]
    for module in g.module:
        count = coverage(g.shape, g.chips(module))
        assert count[module].sum() == 512 * 1024
        assert (count[module] == 1).all()


def test_500k_layout_matches_sls_detector_tools():
    g = GEOMETRIES['500k']
    assert g.shape == (512, 1024)
    assert g.halfmodule == ((slice(256, 512), slice(0, 1024)), (slice(0, 256), slice(0, 1024)))
    assert g.port == (
        (slice(256, 512), slice(0, 512)), (slice(256, 512), slice(512, 1024)),
        (slice(0, 256), slice(0, 512)), (slice(0, 256), slice(512, 1024)),
    )
    assert g.chips(g.module[0]) == CHIPS
    assert g.vcmp_names == [
        '0:vcmp_ll', '0:vcmp_lr', '0:vcmp_rl', '0:vcmp_rr',
        '1:vcmp_rr', '1:vcmp_rl', '1:vcmp_lr', '1:vcmp_ll',
    ]


def test_1M_bottom_halfmodules_are_the_ones_the_old_receiver_flipped():
    g = GEOMETRIES['1M']
    bottoms = [hm[0] for hm in g.halfmodule[1::2]]
    assert bottoms == [slice(512, 768), slice(0, 256)]
