from amaranth import *
from contextlib import nullcontext

class StreamSkidBuffer(Elaboratable):
    def __init__(self, stream_class, shape, reg_output=False):
        self.input  = stream_class(shape)
        self.output = stream_class(shape)
        self.reg_output = reg_output

    def elaborate(self, platform):
        m = Module()

        # Internal signals
        r_valid     = Signal()
        in_payload  = self.input.payload
        out_payload = self.output.payload
        r_payload   = Signal.like(in_payload, reset_less=True)

        # Internal storage is only valid when there is incoming
        # data but the consumer is not ready
        with m.If(self.input.consume & ~self.output.produce):
            m.d.sync += r_valid.eq(1)
        with m.Elif(self.output.ready):
            m.d.sync += r_valid.eq(0)

        # Keep storing input data
        with m.If(self.input.consume):
            m.d.sync += r_payload.eq(in_payload)
        
        # As long as our internal buffer is empty, we accept a new sample
        # This internal buffer provides the "elasticity" needed due to
        # the register delay in the `ready` signal path.
        m.d.comb += self.input.ready.eq(~r_valid)

        # Drive output valid and data signals
        # Changes between our internal/buffered signals or direct connection
        out_domain  = m.d.sync if self.reg_output else m.d.comb
        out_context = m.If(self.output.produce) if self.reg_output else nullcontext()
        with out_context:
            out_domain += self.output.valid.eq(self.input.valid | r_valid)
            out_domain += out_payload.eq(Mux(r_valid, r_payload, in_payload))

        return m
