from amaranth import *
from .types.fixed_point import FixedPointConst, FixedPointValue
from .streams import ComplexStream, SampleStream
from scipy.signal import get_window

class Window(Elaboratable):
    def __init__(self, shape, N, window="hann", coeff_shape=None):
        self.N      = N
        self.window = window
        self.cshape = coeff_shape or shape
        self.input  = ComplexStream(shape)
        self.output = ComplexStream(shape)

    def window_coefficients(self):
        w = get_window(self.window, self.N, fftbins=False)  # symmetric window
        return [ FixedPointConst(self.cshape, s) for s in w ]
        
    def elaborate(self, platform):
        m = Module()

        m.submodules.win = win = CyclicStream(self.cshape, self.window_coefficients())

        m.d.comb += self.input.ready.eq(self.output.produce & win.output.valid)
        m.d.comb += win.output.ready.eq(self.output.produce & self.input.valid)

        x, w = self.input, win.output
        w_fp = FixedPointValue(self.cshape, value=w.payload)
        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(x.valid & w.valid)
            with m.If(x.valid & w.valid):
                m.d.sync += self.output.payload.eq((x.payload * w_fp)
                                                    .reshape(self.output.shape))

        return m
    
class CyclicStream(Elaboratable):
    def __init__(self, shape, samples):
        self.samples = samples
        self.output  = SampleStream(shape)
    
    def elaborate(self, platform):
        m = Module()

        # Stream ROM
        samp_width = Shape.cast(self.output.payload.shape()).width
        init_values = [ s.value for s in self.samples ]
        mem = Memory(width=samp_width, depth=len(self.samples), init=init_values)
        m.submodules.mem_rd = mem_rd = mem.read_port(domain="sync", transparent=False)

        addr = Signal(range(len(self.samples)))
        with m.If(self.output.produce):
            m.d.sync += [
                self.output.valid.eq(1),
                addr             .eq(_incr(addr, len(self.samples))),
            ]

        m.d.comb += [
            mem_rd.addr         .eq(addr),
            mem_rd.en           .eq(self.output.produce),
            self.output.payload .eq(mem_rd.data),
        ]

        return m

def _incr(signal, modulo):
    if modulo == 2 ** len(signal):
        return signal + 1
    else:
        return Mux(signal == modulo - 1, 0, signal + 1)