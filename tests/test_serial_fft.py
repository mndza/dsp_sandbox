import unittest

from dsp_sandbox.serial_fft import SerialFFT, SDFRadix2Stage
from dsp_sandbox.types.fixed_point import Q
from dsp_sandbox.types.complex import ComplexConst
from numpy.fft import fft as np_fft
from itertools import zip_longest
from stream_helper import stream_process

class TestSerialFFT(unittest.TestCase):

    def test_stage(self):
        N = 4
        shape = Q(5,0)
        dut = SDFRadix2Stage(N, shape)
        samples = [ i for i in range(4) ]
        input_sequence = map(lambda x: ComplexConst(shape=shape, value=x), samples)
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=30)
        self.assertListEqual(out, [2, 4, -2, -2])

    def fft_testbench(self, input_idle_cycles, output_stall_cycles, cycles):
        N = 128
        shape=Q(1, 10)
        samples = [ i/N for i in range(N) ]
        dut = SerialFFT(N=N, shape=shape)
        input_sequence = map(lambda x: ComplexConst(shape=shape, value=x), samples)
        out = stream_process(dut, dut.input, dut.output, input_sequence, input_idle_cycles=input_idle_cycles, output_stall_cycles=output_stall_cycles, cycles=cycles)
        expected = np_fft(samples, n=N)
        for x,y in zip_longest(out, expected):
            self.assertAlmostEqual(x, y, delta=0.02)

    def test_fft_streams(self):
        for input_idle_cycles in [0, 1, 2, 3]:
            for output_stall_cycles in [0, 1, 2, 3]:
                self.fft_testbench(input_idle_cycles, output_stall_cycles, cycles=20*128)

if __name__ == "__main__":
    unittest.main()