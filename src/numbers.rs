use num_traits::FromPrimitive;

/// The `Number` trait provides a unified interface for numeric types (integers and floats).
///
/// This trait is designed to allow generic programming over both integer and floating-point types,
/// supporting basic arithmetic operations, logical comparisons, and some mathematical functions commonly used in neural networks.
///
/// # Required Methods
/// - `zero()` and `one()`: Return the additive and multiplicative identity for the type.
/// - `exp(self)`: Exponential function. Only implemented for floating-point types; panics for integers.
/// - `tanh(self)`: Hyperbolic tangent function. Only implemented for floating-point types; panics for integers.
/// - Logical comparisons: `and`, `or`, `not`, `eq`, `ne`, `gt`, `lt`, `ge`, `le`
///
/// # Implementations
/// - `f32`, `f64`: Fully supported, including `exp`, `tanh`, and logical comparisons.
/// - `i32`, `i64`, `usize`: Supported for arithmetic, identity, and logical comparisons, but `exp` and `tanh` will panic if called.
///
pub trait Number:
    Copy
    + Default
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + PartialEq
{
    /// Returns the additive identity (zero) for the type.
    fn zero() -> Self;
    /// Returns the multiplicative identity (one) for the type.
    fn one() -> Self;
    /// Returns the exponential of the value.
    /// Only implemented for floating-point types; panics for integers.
    fn exp(self) -> Self;
    /// Returns the hyperbolic tangent of the value.
    /// Only implemented for floating-point types; panics for integers.
    fn tanh(self) -> Self;

    fn ln(self) -> Self;

    /// Logical AND: returns one if both are non-zero, else zero.
    fn and(self, rhs: Self) -> Self;
    /// Logical OR: returns one if either is non-zero, else zero.
    fn or(self, rhs: Self) -> Self;
    /// Logical NOT: returns one if zero, else zero.
    fn not(self) -> Self;

    /// Equality comparison.
    fn eq(self, rhs: Self) -> bool;
    /// Inequality comparison.
    fn ne(self, rhs: Self) -> bool;
    /// Greater than.
    fn gt(self, rhs: Self) -> bool;
    /// Less than.
    fn lt(self, rhs: Self) -> bool;
    /// Greater than or equal.
    fn ge(self, rhs: Self) -> bool;
    /// Less than or equal.
    fn le(self, rhs: Self) -> bool;
    fn to_number<T: Number + FromPrimitive>(x: f64) -> T;
}


impl Number for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn exp(self) -> Self { self.exp() }
    fn tanh(self) -> Self { self.tanh() }
    fn ln(self) -> Self { self.ln() }

    fn and(self, rhs: Self) -> Self {
        if self != 0.0 && rhs != 0.0 { Self::one() } else { Self::zero() }
    }
    fn or(self, rhs: Self) -> Self {
        if self != 0.0 || rhs != 0.0 { Self::one() } else { Self::zero() }
    }
    fn not(self) -> Self {
        if self == 0.0 { Self::one() } else { Self::zero() }
    }

    fn eq(self, rhs: Self) -> bool { self == rhs }
    fn ne(self, rhs: Self) -> bool { self != rhs }
    fn gt(self, rhs: Self) -> bool { self > rhs }
    fn lt(self, rhs: Self) -> bool { self < rhs }
    fn ge(self, rhs: Self) -> bool { self >= rhs }
    fn le(self, rhs: Self) -> bool { self <= rhs }
    fn to_number<T: Number + FromPrimitive>(x: f64) -> T {
        T::from_f64(x).unwrap()
    }
}

impl Number for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn exp(self) -> Self { self.exp() }
    fn tanh(self) -> Self { self.tanh() }
    fn ln(self) -> Self { self.ln() }

    fn and(self, rhs: Self) -> Self {
        if self != 0.0 && rhs != 0.0 { Self::one() } else { Self::zero() }
    }
    fn or(self, rhs: Self) -> Self {
        if self != 0.0 || rhs != 0.0 { Self::one() } else { Self::zero() }
    }
    fn not(self) -> Self {
        if self == 0.0 { Self::one() } else { Self::zero() }
    }

    fn eq(self, rhs: Self) -> bool { self == rhs }
    fn ne(self, rhs: Self) -> bool { self != rhs }
    fn gt(self, rhs: Self) -> bool { self > rhs }
    fn lt(self, rhs: Self) -> bool { self < rhs }
    fn ge(self, rhs: Self) -> bool { self >= rhs }
    fn le(self, rhs: Self) -> bool { self <= rhs }
    fn to_number<T: Number + FromPrimitive>(x: f64) -> T {
        T::from_f64(x).unwrap()
    }
}

impl Number for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn exp(self) -> Self { panic!("exp not supported for i32") }
    fn tanh(self) -> Self { panic!("tanh not supported for i32") }
    fn ln(self) -> Self { panic!("ln not supported for i32") }

    fn and(self, rhs: Self) -> Self {
        if self != 0 && rhs != 0 { Self::one() } else { Self::zero() }
    }
    fn or(self, rhs: Self) -> Self {
        if self != 0 || rhs != 0 { Self::one() } else { Self::zero() }
    }
    fn not(self) -> Self {
        if self == 0 { Self::one() } else { Self::zero() }
    }

    fn eq(self, rhs: Self) -> bool { self == rhs }
    fn ne(self, rhs: Self) -> bool { self != rhs }
    fn gt(self, rhs: Self) -> bool { self > rhs }
    fn lt(self, rhs: Self) -> bool { self < rhs }
    fn ge(self, rhs: Self) -> bool { self >= rhs }
    fn le(self, rhs: Self) -> bool { self <= rhs }
    fn to_number<T: Number + FromPrimitive>(x: f64) -> T {
        T::from_f64(x).unwrap()
    }
}

impl Number for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn exp(self) -> Self { panic!("exp not supported for i64") }
    fn tanh(self) -> Self { panic!("tanh not supported for i64") }
    fn ln(self) -> Self { panic!("ln not supported for i64") }

    fn and(self, rhs: Self) -> Self {
        if self != 0 && rhs != 0 { Self::one() } else { Self::zero() }
    }
    fn or(self, rhs: Self) -> Self {
        if self != 0 || rhs != 0 { Self::one() } else { Self::zero() }
    }
    fn not(self) -> Self {
        if self == 0 { Self::one() } else { Self::zero() }
    }

    fn eq(self, rhs: Self) -> bool { self == rhs }
    fn ne(self, rhs: Self) -> bool { self != rhs }
    fn gt(self, rhs: Self) -> bool { self > rhs }
    fn lt(self, rhs: Self) -> bool { self < rhs }
    fn ge(self, rhs: Self) -> bool { self >= rhs }
    fn le(self, rhs: Self) -> bool { self <= rhs }
    fn to_number<T: Number + FromPrimitive>(x: f64) -> T {
        T::from_f64(x).unwrap()
    }
}
