from amaranth import *
from math import ceil, log2

from .streams import SampleStream
from .delay import StreamDelay

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
        self.input        = SampleStream(shape)
        self.output       = SampleStream(shape)
        if not internal_s:
            self.s1       = Signal()
            self.s2       = Signal()

    def elaborate(self, platform):
        m = Module()

        # Internal delay
        m.submodules.delay = delay = StreamDelay(shape=self.shape, delay=self.L)

        if self.internal_s:
            s1 = Signal()
            s2 = Signal()
            j, k = self.j, self.k
            counter_i = Signal(range(2**(j+1)))
            counter_o = Signal.like(counter_i, reset=self.L)
            with m.If(delay.input.consume):
                m.d.sync += counter_i.eq(counter_i + 1)
            with m.If(self.output.consume):
                m.d.sync += counter_o.eq(counter_o + 1)
            m.d.comb += [
                s1.eq((~counter_i[j]) | counter_i[k]),
                s2.eq((~counter_o[j]) | counter_o[k]),
            ]

        # Generation of signals for driving the multiplexers
        # This additional logic synchronizes multiplexers for legal
        # state changes and avoids stalls
        m1 = Signal(2)  # additional bit to allow "disabling" muxes
        m2 = Signal(2)
        with m.FSM(name="mux_ctl") as fsm:
            with m.State("0"):
                m.d.comb += [ m1.eq(0), m2.eq(0) ]
                with m.If(   s1 & ~s2): m.d.comb += [ m1.eq(2), m2.eq(0) ]
                with m.Elif(~s1 &  s2): m.d.comb += [ m1.eq(0), m2.eq(2) ]
                with m.Elif( s1 &  s2):
                    m.d.comb += [ m1.eq(1), m2.eq(1) ]
                    m.next = "1"
            with m.State("1"):
                m.d.comb += [ m1.eq(1), m2.eq(1) ]
                with m.If(  ~s1 &  s2): m.d.comb += [ m1.eq(2), m2.eq(1) ]
                with m.Elif( s1 & ~s2): m.d.comb += [ m1.eq(1), m2.eq(2) ]
                with m.Elif(~s1 & ~s2):
                    m.d.comb += [ m1.eq(0), m2.eq(0) ]
                    m.next = "0"

        # Delay input multiplexer
        with m.Switch(m1):
            with m.Case(0):
                m.d.comb += [
                    delay.input.stream_eq(delay.output, omit="ready"),
                    delay.output.ready.eq(1),  # avoid combinatorial loop
                ]
            with m.Case(1):
                m.d.comb += delay.input.stream_eq(self.input)
        
        # Output multiplexer
        with m.Switch(m2):
            with m.Case(0):
                m.d.comb += self.output.stream_eq(self.input)
            with m.Case(1):
                m.d.comb += self.output.stream_eq(delay.output)

        return m

class SerialBitReversal(Elaboratable):
    def __init__(self, shape, N):
        assert N & (N-1) == 0, "N must be a power of two"
        self.shape = shape
        self.N = N
        # signals
        self.input  = SampleStream(shape)
        self.output = SampleStream(shape)
        self.s      = Signal()

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
        last = self.input
        for stage in stages:
            m.d.comb += stage.input.stream_eq(last)
            last = stage.output
        m.d.comb += self.output.stream_eq(last)

        return m
