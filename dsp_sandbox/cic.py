from amaranth import Elaboratable, Module, Shape, Signal, Mux, EnableInserter
from .streams import ComplexStream
from .types.fixed_point import Q, FixedPointConst
from .types.complex import Complex, ComplexConst

class UpsamplingCICFilter(Elaboratable):
    def __init__(self, M, stages, rate, width_in, width_out=None):
        self.M            = M
        self.stages       = stages
        self.rate         = rate
        self.width_in     = width_in
        self.width_out    = width_out or (width_in + self.bit_growth_summary()[-1])
        assert M % rate == 0  # allow M multiple of rate
        self.input        = ComplexStream(Q(self.width_in, 0))
        #growth = self.bit_growth_summary()[-1]
        self.output       = ComplexStream(Q(self.width_out, 0))

    def bit_growth_summary(self):
        # Comb stages add +1 bit each
        #bit_growths = [ self.width + 1 + i for i in range(self.stages) ]
        # Integrator stages are a bit more complex and we get the max growths by simulation
        #bit_growths += get_integrator_bit_growths(stages=self.stages, M=self.M)
        bit_growths = cic_growth(N=self.stages, M=self.M, R=1)
        return bit_growths
    
    #def plot_response(self):

    def elaborate(self, platform):
        m = Module()

        bit_growths = self.bit_growth_summary()
        #print(f'bit growths {bit_growths}')
        input_widths = [self.width_in + g for g in [0]+bit_growths[:-1]]
    
        # Comb stages
        comb_in_w = input_widths[:len(input_widths)//2]
        stages  = [ CombStage(self.M // self.rate, w) for w in comb_in_w]
        # Upsampling
        if self.rate != 1:
            upsampler_in_w = input_widths[len(input_widths)//2]
            stages += [ Upsampler(upsampler_in_w, self.rate) ]
        # Integrator stages
        int_in_w = input_widths[len(input_widths)//2:]
        stages += [ IntegratorStage(in_w, self.width_in + g) for in_w, g in zip(int_in_w, bit_growths[len(input_widths)//2:]) ]

        # Connect stages
        m.submodules += stages
        last = self.input
        for stage in stages:
            m.d.comb += stage.input.stream_eq(last)
            last = stage.output
        m.d.comb += self.output.stream_eq(last)

        return m


class DownsamplingCICFilter(Elaboratable):
    def __init__(self, M, stages, rate, width_in, width_out=None):
        self.width_in     = width_in
        self.width_out    = width_out or (self.width_in + ceil(stages * log2(rate * M)))
        print(self.width_out)
        self.M            = M
        self.stages       = stages
        self.rate         = rate
        assert M % rate == 0  # allow M multiple of rate
        self.input        = ComplexStream(Q(self.width_in, 0))
        self.output       = ComplexStream(Q(self.width_out, 0))

    def truncation_summary(self):
        return cic_truncation(N=self.stages, R=self.rate, M=self.M, Bin=self.width_in, Bout=self.width_out)

    #def plot_response(self):

    def elaborate(self, platform):
        m = Module()

        stages = []

        trunc = self.truncation_summary()
        full_width = self.width_in + ceil(self.stages * log2(self.rate * self.M))
        je = [full_width - x for x in trunc]
        trunc2 = [ (je[i-1] if i>0 else full_width) - je[i] for i in range(len(je)) ]

        # Integrator stages
        stage_width = full_width
        for i in range(self.stages):
            stage_width = full_width - trunc[i]
            stages += [ IntegratorStage(stage_width, stage_width) ]

        print('LEL')

        # Downsampling
        if self.rate != 1:
            stages += [ Downsampler(stage_width, self.rate) ]

        # Comb stages
        for i in range(self.stages):
            stage_width = full_width - trunc[self.stages + i]
            stages += [ CombStage(self.M // self.rate, stage_width, stage_width) ]

        # Connect stages
        m.submodules += stages
        last = self.input
        for i, stage in enumerate(stages):
            if i == self.stages:
                m.d.comb += stage.input.payload.eq(last.payload.reshape(stage.input.shape))
            elif i < self.stages:
                m.d.comb += stage.input.payload.eq((last.payload >> (trunc2[i])).reshape(stage.input.shape))
                print(trunc[i])
            else:
                m.d.comb += stage.input.payload.eq((last.payload >> trunc2[i-1]).reshape(stage.input.shape))
                print(trunc[i-1])
            m.d.comb += stage.input.stream_eq(last, omit="payload")
            last = stage.output
        m.d.comb += self.output.payload.eq((last.payload >> trunc2[-1]).reshape(self.output.shape))
        m.d.comb += self.output.stream_eq(last, omit="payload")

        return m


class CombStage(Elaboratable):
    def __init__(self, M, width_in, width_out=None, truncate=0):
        self.M         = M
        self.width_in  = width_in
        self.width_out = width_out or width_in + 1
        self.input     = ComplexStream(Q(self.width_in, 0))
        self.output    = ComplexStream(Q(self.width_out, 0))  # 1-bit growth
        self.trunc     = truncate

    def bit_growth(self):
        return 1
    
    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.input.ready.eq(self.output.produce)

        m.submodules.delay = delay = EnableInserter(self.input.consume)(Delay(2*self.width_in, self.M))
        m.d.comb += delay.input.eq(self.input.payload)
        delayed = Complex(shape=self.input.shape, value=delay.output)

        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                m.d.sync += self.output.payload.eq(((self.input.payload - delayed)).reshape(self.output.shape))
        
        return m

class IntegratorStage(Elaboratable):
    def __init__(self, width_in, width_out, truncate=0):
        self.input  = ComplexStream(Q(width_in, 0))
        self.output = ComplexStream(Q(width_out, 0))
        self.trunc = truncate

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.input.ready.eq(self.output.produce)

        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                m.d.sync += self.output.payload.eq(((self.output.payload + self.input.payload)).reshape(self.output.shape))
        
        return m
    
class Upsampler(Elaboratable):
    def __init__(self, width, factor):
        self.factor = factor
        self.input  = ComplexStream(Q(width, 0))
        self.output = ComplexStream(Q(width, 0))

    def elaborate(self, platform):
        m = Module()

        counter = Signal(range(self.factor))
        counter_next = _incr(counter, self.factor)
        m.d.comb += self.input.ready.eq(self.output.produce & (counter == 0))

        with m.If(self.output.produce):
            with m.If(counter == 0):
                m.d.sync += self.output.valid.eq(self.input.valid)
                with m.If(self.input.valid):
                    m.d.sync += self.output.payload.eq(self.input.payload)
                    m.d.sync += counter.eq(counter_next)
            with m.Else():
                m.d.sync += self.output.valid.eq(1)
                m.d.sync += self.output.payload.eq(ComplexConst(self.input.shape, value=0))
                m.d.sync += counter.eq(counter_next)

        return m


class Downsampler(Elaboratable):
    def __init__(self, width, factor):
        self.factor = factor
        self.input  = ComplexStream(Q(width, 0))
        self.output = ComplexStream(Q(width, 0))

    def elaborate(self, platform):
        m = Module()

        counter = Signal(range(self.factor))
        m.d.comb += self.input.ready.eq(self.output.produce | (counter != 0))

        with m.If(self.input.consume):
            m.d.sync += counter.eq(_incr(counter, self.factor))

        with m.If(self.output.produce):
            m.d.sync += self.output.valid.eq(self.input.valid & (counter == 0))
            with m.If(self.input.valid & (counter == 0)):
                m.d.sync += self.output.payload.eq(self.input.payload)

        return m


class Delay(Elaboratable):
    def __init__(self, shape, delay):
        self.delay  = delay
        self.input  = Signal(shape)
        self.output = Signal(shape)

    def elaborate(self, platform):
        m = Module()

        data_in, data_out = self.input, self.output

        delay_line = [ Signal(len(data_in), name=f"delay{i}") for i in range(self.delay) ]
        m.d.comb += data_out.eq(delay_line[-1])

        m.d.sync += delay_line[0].eq(data_in)
        for i in range(self.delay-1):
            m.d.sync += delay_line[i+1].eq(delay_line[i])

        return m

    
def _incr(signal, modulo):
    if modulo == 2 ** len(signal):
        return signal + 1
    else:
        return Mux(signal == modulo - 1, 0, signal + 1)
    


# Refs:
# [1]: Eugene Hogenauer, "An Economical Class of Digital Filters For Decimation and Interpolation,"
#      IEEE Trans. Acoust. Speech and Signal Proc., Vol. ASSP-29, April 1981, pp. 155-162. 
# [2]: Rick Lyons, "Computing CIC filter register pruning using MATLAB"
#      https://www.dsprelated.com/showcode/269.php
# [3]: Peter Thorwartl, "Implementation of Filters", Part 3, lecture notes.
#      https://www.so-logic.net/documents/trainings/03_so_implementation_of_filters.pdf 

from math import floor, log2, ceil, comb

# CIC downsamplers / decimators
# How much can we prune / truncate every stage output given a desired output width ?
# Calculate how many bits we can discard after each intermediate stage such that the quantization 
# error introduced is not greater than the one introduced by truncating/rounding at the filter 
# output.

def F_sq(N, R, M, i):
    assert i <= 2*N + 1
    if i == 2*N + 1: return 1  # eq. (16b) from [1]
    # h(k) and L (range of k), eq. (9b) from [1]
    if i <= N:
        # integrator stage
        L = N * (R * M - 1) + i - 1
        h = lambda k: sum((-1)**l * comb(N, l) * comb(N-i+k-R*M*l, k-R*M*l)
                        for l in range(k//(R*M)+1))
    else:
        # comb stage
        L = 2*N + 1 - i
        h = lambda k: (-1)**k * comb(2*N+1-i, k)
    # Compute standard deviation error gain from stage i to output
    F_i_sq = sum(h(k)**2 for k in range(L+1))
    return F_i_sq

def cic_truncation(N, R, M, Bin, Bout=None):
    full_width = Bin + ceil(N * log2(R * M))  # maximum width at output
    Bout = Bout or full_width                 # allow to specify full width
    B_last = full_width - Bout                # number of bits discarded at last stage
    t = log2(2**(2*B_last)/12) + log2(6 / N)  # Last two terms of (21) from [1]
    truncation = []
    for stage in range(2*N):
        ou = F_sq(N, R, M, stage+1)
        B_i = floor(0.5 * (-log2(ou) + t))    # Eq. (21) from [1]
        truncation.append(max(0, B_i))
    truncation.append(max(0, B_last))
    # From [2]: fix case where input is truncated prior to any filtering
    if truncation[0] == 1:
        truncation[0] = 0
        truncation[1] += 1
    return truncation

# CIC upsamplers / interpolators
# How much bit growth there is per intermediate stage?
# In the interpolator case, we cannot discard bits in intermediate stages: small errors in the 
# interpolator stages causes the variance of the error to grow without bound resulting in an 
# unstable filter.

def cic_growth(N, R, M):
    growths = []
    for i in range(2*N):
        j=i+1
        if j <= N: G_i = 2**j                             # comb stage
        else:      G_i = (2**(2*N-j) * (R*M)**(j-N)) / R  # integration stage
        growths.append(ceil(log2(G_i)))
    return growths
