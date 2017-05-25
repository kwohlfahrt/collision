import pytest
import numpy as np
from collision.misc import *

@pytest.mark.parametrize("x,base,expected", [
    (4, 5, 5), (5, 5, 5), (0, 5, 0),
    (4, 2, 4), (5, 2, 6), (0, 2, 0),
])
def test_roundUp(x, base, expected):
    assert roundUp(x, base) == expected


@pytest.mark.parametrize("x,expected", [
    (1, 1), (2, 2), (3, 4), (5, 8), (6, 8)
])
def test_nextPowerOf2(x, expected):
    assert nextPowerOf2(x) == expected


def test_np_dtype():
    assert dtype_decl(np.dtype('uint32')) == 'uint'
    assert dtype_decl(np.dtype('float16')) == 'half'
    assert dtype_decl(np.dtype(('float16', 4))) == 'half4'
    with pytest.raises(ValueError):
        dtype_decl(np.dtype(('float16', 5)))
    with pytest.raises(ValueError):
        dtype_decl(np.dtype(('float16', (4, 4))))
