import unittest

from amaranth.sim import Simulator
from dsp_sandbox.serial_fft import SerialFFT
from dsp_sandbox.types.fixed_point import Q
from dsp_sandbox.types.complex import ComplexConst
from numpy.fft import fft as np_fft

def compute_fft(N, shape, samples):
    ''' Retrieve output samples from simulation '''
    out = []

    def process():
        yield FFT.input_valid.eq(1)
        for x in samples:
            val = ComplexConst(shape=shape, value=x)
            yield FFT.input.real.value.eq(val.real.value)
            yield FFT.input.imag.value.eq(val.imag.value)
            yield
            valid_out = yield FFT.output_valid
            if valid_out:
                val = yield from FFT.output.to_complex()
                out.append(val)

        yield FFT.input_valid.eq(0)
        
        for i in range(2*N):
            yield
            valid_out = yield FFT.output_valid
            if valid_out:
                val = yield from FFT.output.to_complex()
                out.append(val)
    
    FFT = SerialFFT(N=N, shape=shape)
    sim = Simulator(FFT)
    sim.add_clock(1e-6)
    sim.add_sync_process(process)
    with sim.write_vcd(vcd_file="sdf.vcd"):
        sim.run()
    
    return out


class TestSerialFFT(unittest.TestCase):
    def test_fft(self):
        N = 128
        samples = [ i / N for i in range(N) ]
        out = compute_fft(N=N, shape=Q(1,10), samples=samples)
        expected = np_fft(samples, n=N)

        for x,y in zip(out, expected):
            self.assertAlmostEqual(x, y, delta=0.02)


if __name__ == "__main__":
    unittest.main()