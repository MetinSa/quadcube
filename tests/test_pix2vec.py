from typing import Tuple, Dict

import numpy as np

from quadcube import pix2vec

Pixel = int
UnitVector = Tuple[float, float, float]

N = 2323

TABULATED_SOLUTION: Dict[Pixel, UnitVector] = {
    261757113: (-0.559628725, 0.387791693, 0.732415915),
    312550434: (0.727091610, -0.683203876, -6.76029772e-02),
    202497551: (-1.57909375e-02, 0.133706689, 0.990895152),
    139816313: (-8.51026177e-03, -0.367106944, 0.930139780),
    445394658: (0.651691854, -0.579139113, 0.489791423),
    280631281: (0.747721672, -0.580979407, -0.321520150),
    448335255: (0.645362616, -0.492870629, 0.583597064),
    302025660: (0.682050526, -0.678715825, -0.272308588),
    131586426: (0.113142319, 0.699876368, 0.705245972),
    98477345: (0.371478260, 0.622480452, 0.688855469),
    109661907: (0.131059259, 3.77952382e-02, 0.990653813),
    97989306: (0.408581167, 0.632068694, 0.658445716),
    157889605: (-0.140370801, -0.147884607, 0.978992403),
    125474172: (0.159124821, 0.626980782, 0.762610316),
    92213103: (0.447914273, 0.618587375, 0.645540416),
    160274414: (-0.231547326, -0.275684714, 0.932943702),
    41104145: (0.222546697, -0.409764171, 0.884627700),
    5105435: (0.601013243, -0.378140688, 0.704125464),
    315879751: (0.904772580, -0.405359566, -0.130652979),
    238749305: (-0.515449226, 0.103380442, 0.850661278),
    408302824: (0.917558432, -0.396650761, 2.74718441e-02),
    290970621: (0.740303993, -2.32418417e-03, -0.672268212),
    427780527: (0.985613048, -9.49372910e-03, 0.168751016),
    376877674: (0.923506141, 0.314976096, -0.218920842),
    387452141: (0.854496479, 0.434947431, -0.284000993),
}


def test_input_types_and_shapes() -> None:
    assert pix2vec(1).shape == (3, 1)
    assert pix2vec([1]).shape == (3, 1)
    assert pix2vec(np.array([1])).shape == (3, 1)
    assert pix2vec([idx for idx in range(N)]).shape == (3, N)
    assert pix2vec(np.arange(N)).shape == (3, N)


def test_compare_pix2vec_with_tabulated() -> None:
    computed_vectors = pix2vec(list(TABULATED_SOLUTION.keys()))
    for computed_vec, tabulated_vec in zip(
        computed_vectors.transpose(), TABULATED_SOLUTION.values()
    ):
        assert np.allclose(computed_vec, tabulated_vec)