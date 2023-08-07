from itertools import zip_longest
from amaranth import Elaboratable, Module, Signal, Cat

from dsp_sandbox.streams import ComplexStream
from dsp_sandbox.types.complex import Complex


class FIRFilter(Elaboratable):
    def __init__(self, taps, shape_in, shape_out, *, shape_taps):
        self.taps       = list(taps)
        self.shape_taps = shape_taps
        self.input      = ComplexStream(shape_in)
        self.output     = ComplexStream(shape_out)

    def elaborate(self, platform):
        m = Module()

        # TODO:
        # - Multiple windows / sample histories
        # - Multiple tap sets
        # - Configurable multiplier folding factor

        # History of previous samples
        delay_line = [ Complex(shape=self.input.shape) for _ in range(len(self.taps) - 1) ]

        # Convert floating point taps to fixed point constants
        taps      = self.taps
        symmetric = taps == taps[::-1]
        if symmetric:
            taps = taps[:(len(taps)+1)//2]
        taps = [ self.shape_taps.const(tap) for tap in taps ]

        # Sample window for multiplication with taps
        window = [self.input.payload] + delay_line
        if symmetric:
            new_window = [ window[i] + window[-i-1] for i in range(len(window)//2) ]
            if len(window) % 2 == 1:
                new_window.append(window[len(window)//2])            
            window = new_window

        # Multiplication stage: definitions and stream processing
        muls_val = [ b * a for a, b in zip(taps, window) ]
        muls_reg = [ Complex(shape=m.shape, name="mul") for m in muls_val ]

        muls_valid = Signal()
        muls_ready = Signal()

        m.d.comb += self.input.ready.eq(~muls_valid | muls_ready)

        with m.If(self.input.ready):
            m.d.sync += muls_valid.eq(self.input.valid)
            with m.If(self.input.valid):
                # Multiply current window and store results
                for reg, value in zip(muls_reg, muls_val):
                    m.d.sync += reg.eq(value)
                # Update sample history
                m.d.sync += Cat(delay_line).eq(Cat(self.input.payload, *delay_line))
                
        # Adder tree stages, with ceil(log2(N)) levels
        level, level_valid, level_ready = muls_reg, muls_valid, muls_ready
        while len(level) > 1:
            even = level[0::2]
            odd  = level[1::2]
            results = [ a+b if b is not None else a for a,b in zip_longest(even, odd) ]
            new_level = [ Complex(shape=r.shape) for r in results ]
            new_valid = Signal()
            new_ready = Signal()
            m.d.comb += level_ready.eq(~new_valid | new_ready)
            with m.If(level_ready):
                m.d.sync += new_valid.eq(level_valid)
                with m.If(level_valid):
                    for reg, value in zip(new_level, results):
                        m.d.sync += reg.eq(value)
            level, level_valid, level_ready = new_level, new_valid, new_ready

        # Output wiring
        m.d.comb += self.output.payload.eq(level[0].reshape(self.output.payload.shape))
        m.d.comb += self.output.valid  .eq(level_valid)
        m.d.comb += level_ready        .eq(self.output.ready)

        return m
