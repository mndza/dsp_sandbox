import unittest
from math import ceil, log2
from amaranth import unsigned
from dsp_sandbox.bit_exchange import SerialBitReversal, SerialBitExchange
from stream_helper import stream_process

def binrev(v, n):
    nbits = ceil(log2(n))
    indexes = [ int(f'{i:0{nbits}b}'[::-1], 2) for i in range(n) ]
    return [ v[indexes[i]] for i in range(n) ]

class TestSerialBitReversal(unittest.TestCase):
    def test_serial_bit_reversal(self):
        N = 32
        shape = unsigned(5)
        expected = binrev(list(range(N)), N)
        dut = SerialBitReversal(shape, N)
        input_sequence = list(range(32))
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=100)
        self.assertListEqual(expected, out)

    def test_serial_bit_exchange(self):
        N = 8
        shape = unsigned(5)
        expected = [0, 4, 2, 6, 1, 5, 3, 7]
        dut = SerialBitExchange(shape, 2, 0)
        input_sequence = list(range(N))
        out = stream_process(dut, dut.input, dut.output, input_sequence, cycles=100)
        self.assertListEqual(expected, out)
    
if __name__ == "__main__":
    unittest.main()