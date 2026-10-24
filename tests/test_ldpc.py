import numpy as np
from adaptive_ofdm.ldpc import NRLDPC


def test_nr_ldpc_no_noise_roundtrip_and_linearity():
    rng=np.random.default_rng(31)
    codec=NRLDPC(160,320,iterations=12)
    info=rng.integers(0,2,(6,160),dtype=np.uint8)
    coded=codec.encode(info)
    recovered=codec.decode(30*(1-2*coded.astype(float)))
    np.testing.assert_array_equal(recovered,info)
    np.testing.assert_array_equal(codec.encode(info[:3]^info[3:]),coded[:3]^coded[3:])
    np.testing.assert_array_equal(codec.encode(np.zeros((1,160),np.uint8)),np.zeros((1,320),np.uint8))
