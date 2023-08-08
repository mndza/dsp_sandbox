from amaranth import Elaboratable, Module, Signal, Cat, Shape, Mux, Value, C
from dsp_sandbox.streams import ComplexStream
from dsp_sandbox.delay import StreamDelay
from dsp_sandbox.types.fixed_point import Q, FixedPointConst, FixedPointRounding
from dsp_sandbox.types.complex import Complex, ComplexConst
from dsp_sandbox.cic import Delay, Downsampler
from amalthea.gateware.stream import StreamFork
from amaranth import EnableInserter

import numpy as np

from dsp_sandbox.fir import FIRFilter

class FIRHalfBandDecimator(Elaboratable):
    def __init__(self, Fpb, N, shape_in, shape_out):
        self.Fpb     = Fpb
        self.N       = N
        self.input   = ComplexStream(shape_in)
        self.output  = ComplexStream(shape_out)

    @staticmethod
    def get_fir_taps(N, Fpb):
        # Algorithm described in "A Trick for the Design of FIR Half-Band Filters":
        # https://resolver.caltech.edu/CaltechAUTHORS:VAIieeetcs87a
        assert Fpb < 0.25, "A half-band filter requires that Fpb is smaller than 0.25"
        assert N % 2 == 0, "Filter order N must be a multiple of 2"
        assert N % 4 != 0, "Filter order N must not be a multiple of 4"
        taps = signal.remez(N//2+1, [0., 2*Fpb, .5, .5], [1, 0], [1, 1])
        taps /= 2
        return taps

    def elaborate(self, platform):
        m = Module()

        odd = Signal()
        with m.If(self.input.consume):
            m.d.sync += odd.eq(~odd)

        shape_taps = Q(1, 16)
        fir   = FIRFilter(self.get_fir_taps(self.N, self.Fpb), self.input.shape, self.output.shape, shape_taps=shape_taps)
        delay = EnableInserter(odd & self.input.valid)(Delay(2*Shape.cast(self.input.shape).width, self.N//4+1))

        m.submodules += fir
        m.submodules += delay

        delayed = Complex(shape=self.input.shape, value=delay.output)

        m.d.comb += [
            # Switch input between FIR and delay
            fir.input.payload   .eq(self.input.payload.reshape(fir.input.shape)),
            fir.input.valid     .eq(self.input.valid & ~odd),
            delay.input         .eq(self.input.payload),
            self.input.ready    .eq(fir.input.ready | odd),

            # Wire output to sum of both arms
            self.output         .stream_eq(fir.output, omit=("payload",)),
            self.output.payload .eq((fir.output.payload + (delayed >> 1)).reshape(self.output.shape)),
        ]

        return m
    

class FIRHalfBandInterpolator(Elaboratable):
    def __init__(self, Fpb, N, shape_in, shape_out):
        self.Fpb     = Fpb
        self.N       = N
        self.input   = ComplexStream(shape_in)
        self.output  = ComplexStream(shape_out)

    @staticmethod
    def get_fir_taps(N, Fpb):
        # Algorithm described in "A Trick for the Design of FIR Half-Band Filters":
        # https://resolver.caltech.edu/CaltechAUTHORS:VAIieeetcs87a
        assert Fpb < 0.25, "A half-band filter requires that Fpb is smaller than 0.25"
        assert N % 2 == 0, "Filter order N must be a multiple of 2"
        assert N % 4 != 0, "Filter order N must not be a multiple of 4"
        taps = signal.remez(N//2+1, [0., 2*Fpb, .5, .5], [1, 0], [1, 1])
        taps /= 2
        return taps

    def elaborate(self, platform):
        m = Module()

        odd = Signal()
        with m.If(self.output.consume):
            m.d.sync += odd.eq(~odd)

        shape_taps = Q(1, 16)
        fork  = StreamFork(self.input, 2)
        fir   = FIRFilter(self.get_fir_taps(self.N, self.Fpb), self.input.shape, self.output.shape, shape_taps=shape_taps)
        delay = EnableInserter(fork.outputs[1].valid)(Delay(2*Shape.cast(self.input.shape).width, self.N//4+1))

        m.submodules += fork
        m.submodules += fir
        m.submodules += delay

        delayed = Complex(shape=self.input.shape, value=delay.output)

        m.d.comb += [
            # Wire to both arms (FIR filter and delay)
            fir.input.payload       .eq(fork.outputs[0].payload),
            fir.input.valid         .eq(fork.outputs[0].valid),
            fork.outputs[0].ready   .eq(fir.input.ready),

            delay.input             .eq(fork.outputs[1].payload),
            fork.outputs[1].ready   .eq(1),

            # Switch output between arms
            self.output.payload     .eq( Mux(odd, fir.output.payload, (delayed >> 1).reshape(self.output.shape)) ),
            self.output.valid       .eq(Mux(odd, fir.output.valid, self.output.ready)),
            fir.output.ready        .eq(Mux(odd, self.output.ready, 0)),
        ]

        return m






def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)


from scipy import signal

# Fs : sample frequency
# Fpb: pass-band frequency. 
#      Half band filters are symmatric around Fs/4, so Fpb must be smaller than that.
# N  : filter order (number of taps-1)
#      N must be a multiple of 2, but preferable not a multiple of 4
#
# For a half-band filter, the stop-band frequency Fsb = Fs/2 - Fpb
# This function uses the algorithm described in "A Trick for the Design of FIR Half-Band Filters":
# https://resolver.caltech.edu/CaltechAUTHORS:VAIieeetcs87a
#
# Ideally, N/2 should be odd, because otherwise the outer coefficients of the filter will be 0
# by definition anyway.
def half_band_calc_filter(Fs, Fpb, N):
    assert Fpb < Fs/4, "A half-band filter requires that Fpb is smaller than Fs/4"
    assert N % 2 == 0, "Filter order N must be a multiple of 2"
    assert N % 4 != 0, "Filter order N must not be a multiple of 4"

    g = signal.remez(
            N//2+1,
            [0., 2*Fpb/Fs, .5, .5],
            [1, 0],
            [1, 1]
            )

    zeros = np.zeros(N//2+1)

    h = [item for sublist in zip(g, zeros) for item in sublist][:-1]
    h[N//2] = 1.0
    h = np.array(h)/2

    (w,H) = signal.freqz(h)

    Fsb = Fs/2-Fpb
    
    Hpb_min = min(np.abs(H[0:int(Fpb/Fs*2 * len(H))]))
    Hpb_max = max(np.abs(H[0:int(Fpb/Fs*2 * len(H))]))
    Rpb = 1 - (Hpb_max - Hpb_min)
    
    Hsb_max = max(np.abs(H[int(Fsb/Fs*2 * len(H)+1):len(H)]))
    Rsb = Hsb_max
    
    print("Rpb: %fdB" % (-dB20(Rpb)))
    print("Rsb: %fdB" % -dB20(Rsb))

    return (h, w, H, Rpb, Rsb, Hpb_min, Hpb_max, Hsb_max)

def half_band_find_optimal_N(Fs, Fpb, Apb, Asb, Nmin = 2, Nmax = 1000):
    for N in range(Nmin, Nmax, 4):
        print("Trying N=%d" % N)
        (h, w, H, Rpb, Rsb, Hpb_min, Hpb_max, Hsb_max) = half_band_calc_filter(Fs, Fpb, N)
        if -dB20(Rpb) <= Apb and -dB20(Rsb) >= Asb:
            return N

    return None



