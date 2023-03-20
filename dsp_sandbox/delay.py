from amaranth import *

# TODO: add option for Memory-backed Delay

class Delay(Elaboratable):
    def __init__(self, shape, delay):
        self.delay        = delay
        self.input        = Signal(shape)
        self.input_valid  = Signal()
        self.output       = Signal(shape)
        self.output_valid = Signal()

    def elaborate(self, platform):
        m = Module()

        data_in  = Cat(self.input, self.input_valid)
        data_out = Cat(self.output, self.output_valid)

        delay_line = [ Signal(len(data_in), name=f"delay{i}") for i in range(self.delay) ]
        m.d.comb += data_out.eq(delay_line[-1])

        m.d.sync += delay_line[0].eq(data_in)
        for i in range(self.delay-1):
            m.d.sync += delay_line[i+1].eq(delay_line[i])

        return m