import unittest

from amaranth import signed

from dsp_sandbox.cic import UpsamplingCICFilter, DownsamplingCICFilter
from dsp_sandbox.types.fixed_point import Q
from dsp_sandbox.types.complex import ComplexConst
from itertools import zip_longest
from stream_helper import stream_process

import numpy as np
from scipy.signal import lfilter

from math import floor, ceil, log2

def zero_upsample(samples, factor):
    upsampled = np.zeros(factor*len(samples), dtype=np.complex64)
    upsampled[factor*np.arange(len(samples))] = samples
    return upsampled

def cic_upsample(samples, rate, M, stages):
    out = zero_upsample(samples, rate)
    for _ in range(stages):
        out = lfilter([1] * M, 1, out)
    return out

def cic_downsample(samples, rate, M, stages):
    out = samples
    for _ in range(stages):
        out = lfilter([1] * M, 1, out)
    return out[::rate]

class TestCIC(unittest.TestCase):

    def test_upsampling_cic(self):
        M = 20
        rate = 5
        stages = 3
        shape = Q(12, 0)
        dut = UpsamplingCICFilter(M=M, stages=stages, rate=rate, width_in=len(shape), width_out=len(shape)+10)

        samples = np.random.uniform(-1, 1, 1000) + 1j * np.random.uniform(-1, 1, 1000)
        samples *= ((1 << 11) - 1)
        samples = np.round(samples)

        input_sequence = map(lambda x: ComplexConst(shape=shape, value=x), samples)
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=6000)
        expected = cic_upsample(samples, rate, M, stages)
        expected = [ x / (2 ** 3) for x in expected ]
        expected = [ (floor(x.real) + 1j*floor(x.imag)) for x in expected ]
        print(out[:10])
        print(expected[:10])
        self.assertTrue(np.array_equal(out, expected))

    def test_downsampling_cic(self):
        M = 20
        rate = 5
        stages = 3
        shape = Q(12, 0)
        dut = DownsamplingCICFilter(M=M, stages=stages, rate=rate, width_in=len(shape), width_out=23)

        samples = np.random.uniform(-1, 1, 1000) + 1j * np.random.uniform(-1, 1, 1000)
        samples *= ((1 << 11) - 1)
        samples = np.round(samples)

        input_sequence = map(lambda x: ComplexConst(shape=shape, value=x), samples)
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=6000, vcd_file="cic.vcd")
        expected = cic_downsample(samples, rate, M, stages)

        full_g = ceil(stages * log2(rate * M))

        expected = [ x / (2 ** (9)) for x in expected ]
        expected = [ (floor(x.real) + 1j*floor(x.imag)) for x in expected ]

        for a,b in zip(out, expected):
            #print(a,b)
            assert abs(a - b) < 2
        #self.assertTrue(np.array_equal(out, expected))

if __name__ == "__main__":
    unittest.main()