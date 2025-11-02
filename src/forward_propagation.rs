use crate::numbers::Number;
use crate::layers::{Layer1D, Layer2D};

/// Performs forward propagation for a dense (fully connected) linear layer.
///
/// # Arguments
/// * `inputs` - Array of input values of length `IN`.
/// * `layer` - Reference to a `Layer1D` struct containing:
///     - `weights`: 2D array `[OUT][IN]` where each row contains the weights for one output neuron.
///     - `biases`: Array `[OUT]` containing the bias for each output neuron.
///
/// # Returns
/// * `[T; OUT]` - Array of output values for each neuron in the next layer.
///
/// # Steps
/// 1. For each output neuron `i`:
///     - Initialize the output with its bias: `outputs[i] = biases[i]`.
///     - For each input neuron `j`, multiply the input value by the corresponding weight and add to the output:
///       `outputs[i] += inputs[j] * weights[i][j]`.
/// 2. Repeat for all output neurons.
/// 3. Return the array of computed outputs.
///
pub fn dense_linear<T: Number, const IN: usize, const OUT: usize>(
    inputs: &[T; IN],
    layer: &Layer1D<T, OUT, IN>,
) -> [T; OUT]
where
    T: Number,
{
    let Layer1D { weights, biases } = layer;
    let mut outputs = [T::zero(); OUT];
    for i in 0..OUT {
        // Step 1: Initialize output with bias for neuron i
        outputs[i] = biases[i];
        for j in 0..IN {
            // Step 2: Add weighted input: inputs[j] * weights[i][j]
            outputs[i] = outputs[i] + inputs[j] * weights[i][j];
        }
    }
    // Step 3: Return outputs
    outputs
}

/// Performs forward propagation for a dense 1D convolutional layer.
///
/// # Arguments
/// * `inputs` - Array of input values of length `IN`.
/// * `layer` - Reference to a `Layer2D` struct containing:
///     - `filters`: 2D array `[OUT][FILTER_SIZE]` where each row is a filter for one output neuron.
///     - `biases`: Array `[OUT]` containing the bias for each output neuron.
///
/// # Returns
/// * `[T; OUT]` - Array of output values for each filter/output neuron.
///
/// # Steps
/// 1. For each output neuron (filter) `i`:
///     - Initialize the output with its bias: `outputs[i] = biases[i]`.
///     - For each filter position `j`, multiply the input value by the filter weight and add to the output:
///       `outputs[i] += inputs[j] * filters[i][j]` (only if `j < IN` to avoid out-of-bounds).
/// 2. Repeat for all output neurons.
/// 3. Return the array of computed outputs.
///
pub fn dense_conv2d<T: Number, const IN: usize, const OUT: usize, const FILTER_SIZE: usize>(
    inputs: &[T; IN],
    layer: &Layer2D<T, OUT, FILTER_SIZE>,
) -> [T; OUT]
where
    T: Number,
{
    let Layer2D { filters, biases } = layer;
    let mut outputs = [T::zero(); OUT];
    for i in 0..OUT {
        // Step 1: Initialize output with bias for filter i
        outputs[i] = biases[i];
        for j in 0..FILTER_SIZE {
            // Step 2: Only add weighted input if input exists for this filter position
            if j < IN {
                outputs[i] = outputs[i] + inputs[j] * filters[i][j];
            }
        }
    }
    // Step 3: Return outputs
    outputs
}