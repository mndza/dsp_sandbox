from amaranth import Cat, Const, Module, Shape, Signal, Value, tracer
from amaranth.hdl.ast import ValueCastable, ShapeCastable

# Copied from Amalthea with some changes

class FixedPointShape(ShapeCastable):
    def __init__(self, integer_bits, fraction_bits, signed=True):
        if integer_bits < 0:
            raise ValueError("integer_bits must be >= 0")
        if fraction_bits < 0:
            raise ValueError("fraction_bits must be >= 0")
        if signed and integer_bits < 1:
            raise ValueError("integer_bits must be at least 1 for a signed value")
        self.integer_bits = integer_bits
        self.fraction_bits = fraction_bits
        self.signed = signed

    def as_shape(self):
        """Convert the representation defined by the layout to an unsigned :class:`Shape`."""
        return Shape(len(self), self.signed)

    def value(self, value=None, name=None):
        """Create a FixedPointValue with this shape"""
        return FixedPointValue(self, value=value, name=name)

    def __eq__(self, other):
        return self.integer_bits == other.integer_bits \
            and self.fraction_bits == other.fraction_bits\
            and self.signed == other.signed

    def __len__(self):
        return self.integer_bits + self.fraction_bits

    def __repr__(self):
        return f"Q({self.integer_bits}, {self.fraction_bits}, signed={self.signed})"


def Q(integer_bits, fraction_bits, signed=True):
    return FixedPointShape(integer_bits, fraction_bits, signed)

class FixedPointConst(ValueCastable):
    def __init__(self, shape, value):
        self.shape = shape
        self.value = round(value * (2**self.shape.fraction_bits))

    @ValueCastable.lowermethod
    def as_value(self):
        return Const(self.value, self.shape)
    
    def __len__(self):
        return len(self.shape)



class FixedPointValue(ValueCastable):
    def __init__(self, shape, value=None, name=None):
        self.shape = shape
        self.name = name or tracer.get_var_name(depth=2, default="FixedPoint")
        if value is None:
            self.value = Signal(shape, name=self.name)
        elif isinstance(value, Value):
            self.value = value
        elif isinstance(value, (int, float)):
            val = FixedPointConst(value=value, shape=shape)
            self.value = Const(val.value, shape=val.shape)
        else:
            raise TypeError(f"cannot create FixedPointValue from {value}")

    def eq(self, other):
        if isinstance(other, FixedPointValue):
            assert self.shape == other.shape
            return [self.value.eq(other.value)]
        elif isinstance(other, (Value, ValueCastable)):
            assert len(self.value) == len(other)
            return [self.value.eq(other)]
        else:
            raise TypeError(f"unsupported {type(other)}")

    @ValueCastable.lowermethod
    def as_value(self):
        return self.value

    def to_float(self):
        """Convert to float (only usable in simulation)"""
        value = (yield self.value)
        return float(value) / (2**self.shape.fraction_bits)

    def reshape(self, new_shape):
        if self.shape == new_shape:
            return self
        integer_diff = new_shape.integer_bits - self.shape.integer_bits
        fraction_diff = new_shape.fraction_bits - self.shape.fraction_bits

        # Extend or reduce fraction bits
        value = self.value.shift_left(fraction_diff)

        if integer_diff > 0:
            # Positive difference, extend integer bits
            # sign-extend if needed
            top_bit = value[-1] if self.shape.signed else 0
            value = Cat(value, [top_bit]*integer_diff)
        elif integer_diff < 0:
            # Negative difference, slice away extra integer bits
            value = value[:integer_diff]

        if new_shape.signed:
            value = value.as_signed()
        return FixedPointValue(new_shape, value)


    def _align(self, other):
        """Align the decimal point in two fixed-point values"""
        fraction_bits = max(self.shape.fraction_bits, other.shape.fraction_bits)
        return (
            self.reshape(Q(self.shape.integer_bits, fraction_bits, self.shape.signed)),
            other.reshape(Q(other.shape.integer_bits, fraction_bits, other.shape.signed)),
        )

    def __add__(self, other):
        self, other = self._align(other)
        integer_bits = max(self.shape.integer_bits, other.shape.integer_bits)+1
        signed = self.shape.signed or other.shape.signed
        new_shape = Q(integer_bits, self.shape.fraction_bits, signed)
        return new_shape.value(self.value + other.value)

    def __sub__(self, other):
        self, other = self._align(other)
        integer_bits = max(self.shape.integer_bits, other.shape.integer_bits)+1
        signed = self.shape.signed or other.shape.signed
        new_shape = Q(integer_bits, self.shape.fraction_bits, signed)
        return new_shape.value(self.value - other.value)

    def __mul__(self, other):
        integer_bits = self.shape.integer_bits + other.shape.integer_bits
        fraction_bits = self.shape.fraction_bits + other.shape.fraction_bits
        signed = self.shape.signed or other.shape.signed
        new_shape = Q(integer_bits, fraction_bits, signed)
        return new_shape.value(self.value * other.value)

    def __neg__(self):
        if not self.shape.signed:
            new_shape = Q(self.shape.integer_bits + 1, self.shape.fraction_bits)
        else:
            new_shape = self.shape
        return new_shape.value(-self.value)

    def __rshift__(self, shift):
        return self.shape.value(self.value >> shift)