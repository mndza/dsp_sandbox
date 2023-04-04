import unittest
from amaranth import unsigned, C
from dsp_sandbox.window import CyclicStream, Window
from dsp_sandbox.types.complex import ComplexConst
from dsp_sandbox.types.fixed_point import Q
from stream_helper import stream_process

class TestWindow(unittest.TestCase):
    def test_cyclic_stream(self):
        samples = [ C(i, 5) for i in range(32) ]
        dut = CyclicStream(unsigned(5), samples)
        out = stream_process(dut, None, dut.output, None, output_stall_cycles=1, cycles=200)
        self.assertListEqual(out[:2*len(samples)], 2*list(range(32)))

    def test_window(self):
        N = 32
        shape = Q(5, 10)
        dut = Window(shape, 32)
        samples = [ 1 for i in range(N) ]
        input_seq = map(lambda x: ComplexConst(shape=shape, value=x), samples)
        out = stream_process(dut, dut.input, dut.output, input_seq, cycles=200)
        # TODO: compare to expected output

if __name__ == "__main__":
    unittest.main()