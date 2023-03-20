import unittest
from math import ceil, log2

from amaranth import unsigned
from amaranth.sim import Simulator
from dsp_sandbox.bit_exchange import SerialBitReversal

def binrev(v, n):
    nbits = ceil(log2(n))
    indexes = [ int(f'{i:0{nbits}b}'[::-1], 2) for i in range(n) ]
    return [ v[indexes[i]] for i in range(n) ]

class TestSerialBitReversal(unittest.TestCase):
    def test_serial_bit_reversal(self):
        N = 32
        shape = unsigned(5)
        expected = binrev(list(range(N)), N)

        out = []

        def process():
            yield dut.input_valid.eq(1)
            for x in range(32):
                yield dut.input.eq(x)
                yield
                valid_out = yield dut.output_valid
                if valid_out:
                    val = yield dut.output
                    out.append(val)

            yield dut.input_valid.eq(0)
            
            for i in range(2*32):
                yield
                valid_out = yield dut.output_valid
                if valid_out:
                    val = yield dut.output
                    out.append(val)
            
        dut = SerialBitReversal(shape, N)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_sync_process(process)
        sim.run()

        self.assertListEqual(expected, out)

    
if __name__ == "__main__":
    unittest.main()