from amaranth.sim import Simulator
from itertools import repeat, chain, cycle
from contextlib import nullcontext
from dsp_sandbox.types.complex import Complex

def stream_process(
        dut,
        input_stream,
        output_stream,
        input_sequence,
        cycles=25*32,
        input_idle_cycles=0,
        output_stall_cycles=0,
        vcd_file=None,
        gtkw_file=None):

    out = []
    def output_receiver():
        for _ in range(cycles):
            yield
            ready = yield output_stream.ready
            valid = yield output_stream.valid
            if valid & ready:
                if isinstance(output_stream.payload, Complex):
                    value = yield from output_stream.payload.to_complex()
                else:
                    value = yield output_stream.payload
                out.append(value)

    def output_stall_control():
        counter = 0
        yield output_stream.ready.eq(0)
        for _ in range(cycles):
            yield
            yield output_stream.ready.eq(counter == 0)
            counter = (counter + 1) % (output_stall_cycles + 1)

    def input_sender():
        # Build a generator that delivers values for payload and valid signals
        if input_idle_cycles == 0:
            source = zip(input_sequence, repeat(1))
        else:
            pld = chain.from_iterable(map(lambda x: repeat(x, input_idle_cycles+1), input_sequence))
            source = zip(pld, cycle([1] + [0]*input_idle_cycles))
        
        for _ in range(cycles):
            ready = yield input_stream.ready
            valid = yield input_stream.valid
            if ready or not valid:
                try:
                    payload, valid = next(source)
                except StopIteration:
                    yield input_stream.valid.eq(0)
                    break
                yield input_stream.payload.eq(payload)
                yield input_stream.valid.eq(valid)
            yield
        
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(input_sender)
    sim.add_sync_process(output_receiver)
    sim.add_sync_process(output_stall_control)

    if vcd_file is not None:
        sim_context = sim.write_vcd(vcd_file=vcd_file, gtkw_file=gtkw_file)
    else:
        sim_context = nullcontext()
    
    with sim_context:
        sim.run()

    return out