from amaranth import *
from amaranth import tracer
from amaranth.hdl.ast import ValueCastable
from .fixed_point import FixedPointConst, FixedPointValue, Q

# Copied from Amalthea with some changes

class ComplexConst(ValueCastable):
    def __init__(self, shape, value):
        self.shape = shape
        self.real = FixedPointConst(shape, value.real)
        self.imag = FixedPointConst(shape, value.imag)

    @ValueCastable.lowermethod
    def as_value(self):
        return Cat(self.real, self.imag)

    def value(self):
        mask = int(2**len(self.real.shape)-1)
        return (self.real.value & mask) | ((self.imag.value & mask) << len(self.real.shape))

class Complex(ValueCastable):
    def __init__(self, *, shape=None, value=None, name=None):
        self.shape = shape
        self.name = name or tracer.get_var_name(depth=2, default="Complex")
        if value is None:
            if shape is None:
                raise ValueError(f"must specify `shape` argument")
            self.real = Signal(shape, name=self.name+'_real')
            self.imag = Signal(shape, name=self.name+'_imag')
        elif isinstance(value, complex):
            if shape is None:
                raise ValueError(f"must specify `shape` argument for complex value '{value}'")
            self.real = Const(value.real, shape, name=self.name+'_real')
            self.imag = Const(value.imag, shape, name=self.name+'_imag')
        elif isinstance(value, tuple) and isinstance(value[0], FixedPointValue) and isinstance(value[1], FixedPointValue):
            assert shape is None
            self.real = value[0]
            self.imag = value[1]
            assert self.real.shape == self.imag.shape
            self.shape = self.real.shape
        elif isinstance(value, Value):
            assert len(value) == 2*len(shape)
            l = int(len(value)/2)
            real = value[0:l]
            imag = value[l:]
            if shape.signed:
                real = real.as_signed()
                imag = imag.as_signed()
            self.real = shape(real)
            self.imag = shape(imag)
            self.shape = shape
        else:
            raise TypeError(f"unsupported value {type(value)}")

    def eq(self, other):
        assert len(self.as_value()) == len(Value.cast(other))
        return self.as_value().eq(other)

    def reshape(self, shape, rounding=None):
        real = self.real.reshape(shape, rounding=rounding)
        imag = self.imag.reshape(shape, rounding=rounding)
        return Complex(value=(real, imag))

    def to_complex(self):
        real = (yield from self.real.to_float())
        imag = (yield from self.imag.to_float())
        return complex(real, imag)
        
    @ValueCastable.lowermethod
    def as_value(self):
        return Cat(self.real, self.imag)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        assert real.shape == imag.shape
        return Complex(value=(real, imag))

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        assert real.shape == imag.shape
        return Complex(value=(real, imag))

    def __mul__(self, other):
        if isinstance(other, (Complex, ComplexConst)):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            assert real.shape == imag.shape
        elif isinstance(other, (FixedPointValue, FixedPointConst)):
            real = self.real * other
            imag = self.imag * other
        return Complex(value=(real, imag))

    def __rshift__(self, shift):
        real = self.real >> shift
        imag = self.imag >> shift
        return Complex(value=(real, imag))