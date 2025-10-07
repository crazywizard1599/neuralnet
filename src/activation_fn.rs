use crate::numbers::*;

/// Computes the sigmoid activation for a single value.
///
/// # Arguments
/// * `x` - Input value of type implementing `Number`.
///
/// # Returns
/// * Sigmoid activation: `1 / (1 + exp(-x))`
///
/// # Panics
/// Panics for integer types, as `exp` is not implemented for them.
fn sigmoid<T: Number>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// Applies the sigmoid activation function element-wise to an array.
///
/// # Arguments
/// * `inputs` - Array of input values.
///
/// # Returns
/// * Array of sigmoid-activated values.
pub fn sigmoid_layer<T: Number, const N: usize>(inputs: &[T; N]) -> [T; N] {
    let mut outputs = [T::zero(); N];
    for i in 0..N {
        outputs[i] = sigmoid(inputs[i]);
    }
    outputs
}

/// Computes the ReLU (Rectified Linear Unit) activation for a single value.
///
/// # Arguments
/// * `x` - Input value of type implementing `Number`.
///
/// # Returns
/// * If `x > 0`, returns `x`; otherwise returns zero.
fn relu<T: Number>(x: T) -> T {
    if x.gt(T::zero()) { x } else { T::zero() }
}

/// Applies the ReLU activation function element-wise to an array.
///
/// # Arguments
/// * `inputs` - Array of input values.
///
/// # Returns
/// * Array of ReLU-activated values.
pub fn relu_layer<T: Number, const N: usize>(inputs: &[T; N]) -> [T; N] {
    let mut outputs = [T::zero(); N];
    for i in 0..N {
        outputs[i] = relu(inputs[i]);
    }
    outputs
}

/// Computes the hyperbolic tangent (tanh) activation for a single value.
///
/// # Arguments
/// * `x` - Input value of type implementing `Number`.
///
/// # Returns
/// * Tanh activation: `tanh(x)`
///
/// # Panics
/// Panics for integer types, as `tanh` is not implemented for them.
fn tanh<T: Number>(x: T) -> T {
    x.tanh()
}

/// Applies the tanh activation function element-wise to an array.
///
/// # Arguments
/// * `inputs` - Array of input values.
///
/// # Returns
/// * Array of tanh-activated values.
pub fn tanh_layer<T: Number, const N: usize>(inputs: &[T; N]) -> [T; N] {
    let mut outputs = [T::zero(); N];
    for i in 0..N {
        outputs[i] = tanh(inputs[i]);
    }
    outputs
}

pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
}

impl Activation {
    pub fn forward<T: Number, const N: usize>(&self, inputs: &[T; N]) -> [T; N] {
        match self {
            Activation::Sigmoid => sigmoid_layer(inputs),
            Activation::ReLU => relu_layer(inputs),
            Activation::Tanh => tanh_layer(inputs),
        }
    }

    pub fn derivative<T: Number>(&self, x: T) -> T {
        match self {
            Activation::Sigmoid => {
                let sig = sigmoid(x);
                sig * (T::one() - sig)
            }
            Activation::ReLU => {
                if x.gt(T::zero()) { T::one() } else { T::zero() }
            }
            Activation::Tanh => {
                let t = tanh(x);
                T::one() - t * t
            }
        }
    }
}