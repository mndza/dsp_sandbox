from amaranth import Shape, tracer
from luna.gateware.stream import StreamInterface
from .types.complex import Complex

class StreamProperties:
    @property
    def produce(self):
        return self.ready | ~self.valid
    
    @property
    def consume(self):
        return self.ready & self.valid

class ComplexStream(StreamInterface, StreamProperties):
    def __init__(self, shape):
        name = tracer.get_var_name(depth=2, default=None)
        super().__init__(name=name, payload_width=2*Shape.cast(shape).width)
        self.payload = Complex(shape=shape, value=self.payload)
        self.real = self.payload.real
        self.imag = self.payload.imag

    @property
    def shape(self):
        return self.payload.shape
    
class SampleStream(StreamInterface, StreamProperties):
    def __init__(self, shape):
        name = tracer.get_var_name(depth=2, default=None)
        super().__init__(name=name, payload_width=Shape.cast(shape).width)
