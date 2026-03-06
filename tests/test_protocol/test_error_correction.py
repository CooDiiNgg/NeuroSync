"""Tests for NeuroSync.protocol.error_correction."""
import numpy as np
from NeuroSync.protocol.error_correction import ParityMatrix


class TestParityMatrix:
    def test_encode_shape(self):
        pm = ParityMatrix(96)
        assert len(pm.encode(np.random.choice([-1.0, 1.0], 96))) == pm.total_bits

    def test_decode_no_error(self):
        pm = ParityMatrix(96)
        data = np.random.choice([-1.0, 1.0], 96)
        decoded, syn = pm.decode(pm.encode(data))
        assert syn == 0
        assert np.array_equal(decoded, data)

    def test_single_bit_correction(self):
        pm = ParityMatrix(96)
        data = np.random.choice([-1.0, 1.0], 96)
        encoded = pm.encode(data)
        encoded[5] *= -1.0
        decoded, syn = pm.decode(encoded)
        assert syn > 0
        assert np.array_equal(decoded, data)

    def test_small_data(self):
        pm = ParityMatrix(8)
        data = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        decoded, syn = pm.decode(pm.encode(data))
        assert syn == 0
