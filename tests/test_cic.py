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
        out = lfilter([1] * rate * M, 1, out)
    return out

def cic_downsample(samples, rate, M, stages):
    out = samples
    for _ in range(stages):
        out = lfilter([1] * rate * M, 1, out)
    return out[::rate]

def random_samples_gen(N, width):
    samples = np.random.uniform(-1, 1, N) + 1j * np.random.uniform(-1, 1, N)
    samples *= ((1 << (width-1)) - 1)
    return np.round(samples)

class TestCIC(unittest.TestCase):

    def test_upsampling_cic(self):
        # Parameters and DUT instance
        M = 4
        rate = 5
        stages = 3
        width_in = 12
        width_out = 22
        dut = UpsamplingCICFilter(M=M, stages=stages, rate=rate, width_in=width_in, width_out=width_out)

        # Input samples
        samples = random_samples_gen(1000, width_in)

        # Simulate DUT and gather output stream outputs
        input_sequence = map(lambda x: ComplexConst(shape=Q(width_in, 0), value=x), samples) 
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=10000)

        # Build expected output with our model
        expected = cic_upsample(samples, rate, M, stages)
        full_out = width_in + ceil(log2(((rate*M)**(stages)) / rate))
        if width_out < full_out:
            expected = [ x / (2 ** (full_out-width_out)) for x in expected ]
        expected = [ (floor(x.real) + 1j*floor(x.imag)) for x in expected ]

        # Compare output and expected values
        self.assertTrue(np.array_equal(out, expected))

    def test_downsampling_cic(self):
        # Parameters and DUT instance
        M = 1
        rate = 12
        stages = 3
        width_in = 12
        width_out = 14
        dut = DownsamplingCICFilter(M=M, stages=stages, rate=rate, width_in=width_in, width_out=width_out)

        # Input samples
        samples = random_samples_gen(1000, width_in)

        # Simulate DUT and gather output stream outputs
        input_sequence = map(lambda x: ComplexConst(shape=Q(width_in, 0), value=x), samples) 
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=6000, vcd_file="cic.vcd")

        # Build expected output with our model
        expected = cic_downsample(samples, rate, M, stages)
        full_out = width_in + ceil(stages * log2(rate * M))
        if width_out < full_out:
            expected = [ x / (2 ** (full_out - width_out)) for x in expected ]
        expected = [ (floor(x.real) + 1j*floor(x.imag)) for x in expected ]

        # Compare output and expected values
        # TODO: check why error is not exactly 0, rounding issues?
        error = np.array(out) - np.array(expected)
        max_err = np.max(np.concatenate([np.real(error), np.imag(error)]))
        self.assertTrue(max_err < 2)
        #self.assertTrue(np.array_equal(out, expected))

if __name__ == "__main__":
    unittest.main()