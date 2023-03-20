from amaranth import *
from .delay import Delay
from math import ceil, log2

# TODO: Add documentation
# Reference:
# Mario Garrido Gálvez, Jesus Grajal and Oscar Gustafsson, Optimum Circuits for Bit
# Reversal, 2011, IEEE Transactions on Circuits and Systems - II - Express Briefs, (58), 10,
# 657-661.
# http://urn.kb.se/resolve?urn=urn:nbn:se:liu:diva-71782

class SerialBitExchange(Elaboratable):
    def __init__(self, shape, j, k, internal_s=True):
        assert j > k
        self.shape        = shape
        self.j            = j
        self.k            = k
        self.L            = 2**j - 2**k
        self.internal_s   = internal_s
        # signals
        self.input        = Signal(shape)
        self.input_valid  = Signal()
        self.output       = Signal(shape)
        self.output_valid = Signal()
        if not internal_s:
            self.s        = Signal()

    def elaborate(self, platform):
        m = Module()

        # Internal delay
        m.submodules.delay = delay = Delay(shape=self.shape, delay=self.L)

        if self.internal_s:
            j, k = self.j, self.k
            counter = Signal(range(2**(j+1)))
            with m.If(self.input_valid):
                m.d.sync += counter.eq(counter + 1)
            s = (~counter[j]) | counter[k]

        # Connections
        m.d.comb += [
            # Delay and output muxes
            delay.input       .eq(Mux(s, self.input, delay.output)),
            self.output       .eq(Mux(s, delay.output, self.input)),
            # Valid signals
            delay.input_valid .eq(Mux(s, self.input_valid, delay.output_valid)),
            self.output_valid .eq(Mux(s, delay.output_valid, self.input_valid)),
        ]

        return m

class SerialBitReversal(Elaboratable):
    def __init__(self, shape, N):
        assert N & (N-1) == 0, "N must be a power of two"
        self.shape = shape
        self.N = N
        # signals
        self.input        = Signal(shape)
        self.input_valid  = Signal()
        self.output       = Signal(shape)
        self.output_valid = Signal()
        self.s            = Signal()

    def elaborate(self, platform):
        m = Module()

        bits = ceil(log2(self.N))

        j, k = bits-1, 0
        
        stages = []
        while j != k and j > k:
            bx_stage = SerialBitExchange(self.shape, j, k)
            m.submodules[f'bx_{j}_{k}'] = bx_stage
            stages.append(bx_stage)
            j, k = j-1, k+1

        # Connect stages
        last_data  = self.input
        last_valid = self.input_valid
        for stage in stages:
            m.d.comb += [
                stage.input       .eq(last_data),
                stage.input_valid .eq(last_valid),
            ]
            last_data  = stage.output
            last_valid = stage.output_valid
        m.d.comb += [
            self.output       .eq(last_data),
            self.output_valid .eq(last_valid),
        ]

        return m
