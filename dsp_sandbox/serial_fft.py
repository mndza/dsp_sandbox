from amaranth import Elaboratable, Module, Signal, Memory, Cat

from cmath import exp, pi
from math import ceil, log2
from enum import IntEnum

from .types.fixed_point import Q
from .types.complex import Complex, ComplexConst
from .streams import ComplexStream
from .bit_exchange import SerialBitReversal
from .delay import StreamDelay

# TODO:
# - Add optional scaling / rounding
# - Make twiddle shape configurable
# - Add more tests
# - Add option for Memory-backed Delay module
# - Optional digit slicing / shift-and-add multipliers
# - Simplify control signals using a global counter?

class FFTScaling(IntEnum):
    UNSCALED = 0
    SCALED   = 1

class SerialFFT(Elaboratable):
    '''
    Single-path Delay Feedback FFT
    Radix-2^2
    Decimation in frequency (DIF)
    '''
    def __init__(self, *, N, shape=Q(1,15), natural_order=True):
        assert N & (N-1) == 0, "N must be a power of 2"
        # Internal properties
        self.N              = N
        self.shape          = shape
        self.natural_order  = natural_order
        self.strategy       = FFTScaling.UNSCALED  # force for the moment
        if self.strategy == FFTScaling.UNSCALED:
            output_shape = Q(ceil(log2(N)) + shape.integer_bits, shape.fraction_bits)
        else:
            output_shape = shape
        # Ports
        self.input          = ComplexStream(shape=shape)
        self.output         = ComplexStream(shape=output_shape)

    def elaborate(self, platform):
        m = Module()

        N = self.N

        # Define sequence of butterfly and twiddle stages
        stages    = []
        shape     = self.shape

        # Radix-2^2 stages
        while N >= 4:
            # First butterfly
            stages += [ SDFRadix2Stage(N, shape) ]
            shape = stages[-1].output.shape
            # Trivial twiddle factors (1, -1j)
            stages += [ R22TwiddleStage(N=N, shape=shape) ]
            # Second butterfly
            stages += [ SDFRadix2Stage(N//2, shape) ]
            shape = stages[-1].output.shape
            if N == 4: N = 1; break
            # Twiddle factors
            w = []
            for k1 in range(2):
                for k2 in range(2):
                    w += [ (n3*(k1+2*k2), N) for n3 in range(N//4) ]
            stages += [ TwiddleStage(factors=w, shape=shape) ]
            N = N // 4

        # Radix-2 stages
        while N >= 2:
            # Butterfly
            stages += [ SDFRadix2Stage(N, shape) ]
            shape = stages[-1].output.shape
            if N == 2: N = 1; break
            # Twiddle factors
            w = [ (0, N) ] * (N//2) + [ (k, N) for k in range(N//2) ]
            stages += [ TwiddleStage(factors=w, shape=shape) ]
            N = N // 2

        # Optional bit reversal stage at the end
        if self.natural_order:
            stages += [ SerialBitReversal(2*len(shape), self.N) ]

        # Add all stages as submodules
        m.submodules += stages

        # Connect all stages and input/output
        last = self.input
        for stage in stages:
            m.d.comb += stage.input.stream_eq(last)
            last = stage.output
        m.d.comb += self.output.stream_eq(last)

        return m


class SDFRadix2Butterfly(Elaboratable):
    '''
    Radix-2 butterfly for Single-path Delay Feedback FFT
    '''
    def __init__(self, shape, shape_out=None):
        shape_out = shape_out or Q(1 + shape.integer_bits, shape.fraction_bits)
        self.s = Signal()
        self.a = ComplexStream(shape=shape_out)
        self.b = ComplexStream(shape=shape)
        self.c = ComplexStream(shape=shape_out)
        self.d = ComplexStream(shape=shape_out)
        self.o = Signal()

    def elaborate(self, platform):
        m = Module()

        a, b, c, d = self.a, self.b, self.c, self.d

        with m.If(self.s):
            # Butterfly mode
            m.d.comb += [
                c.payload.eq((a.payload - b.payload).reshape(c.shape)),
                c.valid.eq(a.valid & b.valid),
                d.payload.eq((a.payload + b.payload).reshape(d.shape)),
                d.valid.eq(a.valid & b.valid),
                a.ready.eq(d.produce & b.valid),
                b.ready.eq(c.produce & d.produce & a.valid),
            ]
        with m.Else():
            # Switch mode
            m.d.comb += [
                c.payload.eq(b.payload.reshape(c.shape)),
                c.valid.eq(b.valid),
                d.payload.eq(a.payload.reshape(d.shape)),
                d.valid.eq(a.valid & self.o),
                a.ready.eq(d.produce & self.o),
                b.ready.eq(c.produce),
            ]

        return m

class SDFRadix2Stage(Elaboratable):
    def __init__(self, N, shape, shape_out=None):
        shape_out   = shape_out or Q(1 + shape.integer_bits, shape.fraction_bits)
        self.N      = N
        self.input  = ComplexStream(shape=shape)
        self.output = ComplexStream(shape=shape_out)

    def elaborate(self, platform):
        m = Module()

        N = self.N
        input_shape, output_shape = self.input.shape, self.output.shape

        # Internal counter to generate buterfly control signal
        counter = Signal(range(N))
        with m.If(self.input.consume):
            m.d.sync += counter.eq(counter + 1)
        s = counter[-1]

        # Feedback delay
        m.submodules.delay = delay = StreamDelay(2*len(output_shape)+1, self.N // 2)

        # Butterfly
        m.submodules.butterfly = butterfly = SDFRadix2Butterfly(input_shape, shape_out=output_shape)
        m.d.comb += [
            butterfly.s  .eq(s),
            butterfly.b  .stream_eq(self.input),
            self.output  .stream_eq(butterfly.d),

            delay.input  .stream_eq(butterfly.c, omit="payload"),
            delay.input.payload.eq(Cat(butterfly.c.payload, s)),  # flag samples with s

            butterfly.a  .stream_eq(delay.output, omit="payload"),
            # Top bit of the delay output contains a flag to tell if sample
            # is ready for output
            Cat(butterfly.a.payload, butterfly.o).eq(delay.output.payload),
        ]

        return m

class TwiddleStage(Elaboratable):
    def __init__(self, factors, shape, shape_out=None):
        self.factors   = factors
        self.shape     = shape
        self.shape_out = shape_out or shape
        self.input     = ComplexStream(shape=shape)
        self.output    = ComplexStream(shape=self.shape_out)

    def elaborate(self, platform):
        m = Module()

        twiddle_shape = Q(2, 11)  # this greatly affects output accuracy

        # Internal counter selects current twiddle factor
        counter = Signal(range(len(self.factors)))

        # Twiddle ROM instance
        factors = [ComplexConst(twiddle_shape, exp(-1j*2*pi*k/N)).value() for k,N in self.factors]
        twiddle_rom = Memory(width=2*len(twiddle_shape), depth=len(factors), init=factors)
        m.submodules.twiddle_rd = twiddle_rd = twiddle_rom.read_port(domain="comb")
        factor = Complex(shape=twiddle_shape)
        m.d.comb += [
            twiddle_rd.addr .eq(counter),
            factor          .eq(twiddle_rd.data),
        ]

        # Complex rotate and increment counter per input
        m.d.comb += self.input.ready.eq(self.output.produce)
        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                m.d.sync += self.output.payload.eq((self.input.payload * factor).reshape(self.output.shape))
                m.d.sync += counter.eq(counter + 1)

        return m

class R22TwiddleStage(Elaboratable):
    '''
    Trivial twiddle stage for Radix-2^2, rotates last quarter of the N samples by -1j
    '''
    def __init__(self, N, shape):
        self.N      = N
        self.shape  = shape
        self.input  = ComplexStream(shape=shape)
        self.output = ComplexStream(shape=shape)

    def elaborate(self, platform):
        m = Module()

        counter = Signal(range(self.N))
        
        m.d.comb += self.input.ready.eq(self.output.produce)
        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                with m.If(counter[-1] & counter[-2]):  # last quarter
                    m.d.sync += self.output.real.eq( self.input.imag)
                    m.d.sync += self.output.imag.eq(-self.input.real)
                with m.Else():
                    m.d.sync += self.output.payload.eq(self.input.payload)
                m.d.sync += counter.eq(counter + 1)

        return m
