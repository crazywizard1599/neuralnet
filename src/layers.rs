use crate::numbers::*;
use crate::forward_propagation::*;

/// Fully-connected layer with OUT outputs and IN inputs.
/// weights[i][j] is weight for output i and input j.
pub struct Layer1D<T: Number, const OUT: usize, const IN: usize> {
    pub weights: [[T; IN]; OUT],
    pub biases: [T; OUT],
}

impl<T: Number, const OUT: usize, const IN: usize> Layer1D<T, OUT, IN> {
    pub fn new(weights: [[T; IN]; OUT], biases: [T; OUT]) -> Self {
        Layer1D { weights, biases }
    }

    /// Forward pass: compute outputs = biases + W * inputs
    pub fn forward(&self, inputs: &[T; IN]) -> [T; OUT] {
        crate::forward_propagation::dense_linear(inputs, self)
    }

    /// Update weights and biases in-place given gradients and learning rate.
    /// weight_grads has same shape as weights: [OUT][IN], bias_grads length OUT.
    pub fn update_weights(&mut self, weight_grads: &[[T; IN]; OUT], bias_grads: &[T; OUT], learning_rate: T) {
        for i in 0..OUT {
            self.biases[i] = self.biases[i] - bias_grads[i] * learning_rate;
            for j in 0..IN {
                self.weights[i][j] = self.weights[i][j] - weight_grads[i][j] * learning_rate;
            }
        }
    }
}

pub struct Layer2D<T: Number, const FILTERS: usize, const FILTER_SIZE: usize> {
    pub filters: [[T; FILTER_SIZE]; FILTERS],
    pub biases: [T; FILTERS],
}

impl<T: Number, const FILTERS: usize, const FILTER_SIZE: usize> Layer2D<T, FILTERS, FILTER_SIZE> {
    pub fn forward(&self, inputs: &[T; FILTER_SIZE]) -> [T; FILTERS] { 
        dense_conv2d(inputs, self)
    }
}

/// Creates a fixed-size array representing a linear (fully connected) layer.
///
/// # Arguments
/// * `values` - Slice of input values to fill the layer's neurons.
///
/// # Type Parameters
/// * `T` - Numeric type implementing the `Number` trait.
/// * `N` - Number of neurons in the layer (array size).
///
/// # Returns
/// * Tuple containing:
///   - `[T; N]`: Array of neuron values (weights).
///   - `[T; N]`: Array of bias values (initialized to zero).
///
/// # Behavior
/// - Copies up to `N` values from the input slice into the output array.
/// - If `values.len() < N`, remaining elements are filled with `T::zero()`.
/// - If `values.len() > N`, excess values are ignored.
/// - Bias array is always initialized to zeros.
///
pub fn linear<T: Number, const N: usize, const IN: usize>(values: &[T]) -> Layer1D<T, N, IN> {
    // Create a weight matrix with shape [N][IN] and fill it from the flattened `values` slice.
    // If `values` is shorter than N*IN the remaining entries stay as T::zero().
    // If `values` is longer, excess values are ignored.
    let mut weights = [[T::zero(); IN]; N];
    let mut idx = 0usize;
    for i in 0..N {
        for j in 0..IN {
            if idx < values.len() {
                weights[i][j] = values[idx];
                idx += 1;
            } else {
                // leave default zero
            }
        }
    }

    Layer1D {
        weights,
        biases: [T::zero(); N],
    }
}

/// Creates a fixed-size 2D array representing a Conv2D layer's filters.
///
/// # Arguments
/// * `values` - Slice of input values to fill the filters.
/// 
/// # Type Parameters
/// * `T` - Numeric type implementing the `Number` trait.
/// * `FILTERS` - Number of filters (rows in the output array).
/// * `FILTER_SIZE` - Size of each filter (columns in the output array).
///
/// # Returns
/// * Tuple containing:
///   - `[[T; FILTER_SIZE]; FILTERS]`: 2D array of filter weights.
///   - `[[T; FILTER_SIZE]; FILTERS]`: 2D array of filter biases (initialized to zero).
///
/// # Behavior
/// - Fills each filter with `FILTER_SIZE` values from the input slice.
/// - If `values.len() < FILTERS * FILTER_SIZE`, remaining elements are filled with `T::zero()`.
/// - If `values.len() > FILTERS * FILTER_SIZE`, excess values are ignored.
/// - Bias array is always initialized to zeros.
///
pub fn conv2d<T: Number, const FILTERS: usize, const FILTER_SIZE: usize>(values: &[T]) -> Layer2D<T, FILTERS, FILTER_SIZE> {
    let mut arr = [[T::zero(); FILTER_SIZE]; FILTERS];
    let mut idx = 0;
    for i in 0..FILTERS {
        for j in 0..FILTER_SIZE {
            if idx < values.len() {
                arr[i][j] = values[idx];
                idx += 1;
            }
        }
    }
    Layer2D {
        filters: arr,
        biases: [T::zero(); FILTERS],
    }
}