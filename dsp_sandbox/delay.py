from amaranth import *
from amaranth.lib.fifo import SyncFIFO
from .streams import SampleStream

class StreamDelay(Elaboratable):
    def __init__(self, shape, delay, storage="auto"):
        self.depth        = delay
        self.shape        = shape
        self.storage      = storage
        self.input        = SampleStream(shape)
        self.output       = SampleStream(shape)

    def elaborate(self, platform):
        storage = self.storage
        if storage == "auto":
            # Decide to use FFs or RAM depending on depth
            storage = "bram" if self.depth > 2 else "distributed"

        if storage == "distributed":
            return self.elaborate_distributed(platform)
        elif storage == "bram":
            return self.elaborate_bram(platform)
        
    def elaborate_bram(self, platform):
        m = Module()

        w_data = self.input.payload
        r_data = self.output.payload

        m.submodules.fifo = fifo = SyncFIFO(width=len(w_data), depth=self.depth, fwft=False)
        m.d.comb += [
            # Write signaling
            fifo.w_data      .eq(w_data),
            fifo.w_en        .eq(self.input.valid),
            self.input.ready .eq(fifo.w_rdy),
            # Read signaling
            fifo.r_en        .eq(self.output.produce),
            r_data           .eq(fifo.r_data),
        ]
        with m.If(fifo.r_en):
            m.d.sync += self.output.valid.eq(fifo.r_rdy)

        return m

    def elaborate_distributed(self, platform):
        m = Module()

        reg_line = [ StreamRegister(self.shape) for _ in range(self.depth) ]
        m.submodules += reg_line

        last = self.input
        for reg in reg_line:
            m.d.comb += reg.input.stream_eq(last)
            last = reg.output
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
