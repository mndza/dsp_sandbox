from amaranth import *
from .streams import SampleStream

# TODO: add option for Memory-backed Delay

class StreamDelay(Elaboratable):
    def __init__(self, shape, delay):
        self.delay        = delay
        self.shape        = shape
        self.input        = SampleStream(shape)
        self.output       = SampleStream(shape)

    def elaborate(self, platform):
        m = Module()

        delay_line = [ StreamRegister(self.shape) for _ in range(self.delay) ]
        m.submodules += delay_line

        last = self.input
        for delay in delay_line:
            m.d.comb += delay.input.stream_eq(last)
            last = delay.output
        m.d.comb += self.output.stream_eq(last)

        return m
    
class StreamRegister(Elaboratable):
    def __init__(self, shape):
        self.input  = SampleStream(shape)
        self.output = SampleStream(shape)

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.input.ready.eq(self.output.produce)
        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid)
            m.d.sync += self.output.payload.eq(self.input.payload)
        return m
