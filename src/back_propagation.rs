use crate::loss_fn::*;
use crate::layers::Layer1D;
use crate::numbers::*;
use num_traits::FromPrimitive;

pub fn backward_pass_1d<T: Number + FromPrimitive, const OUT: usize, const IN: usize>(
    layers: &mut [Layer1D<T, OUT, IN>],
    loss_fn: Loss,
    predictions: &[T],
    targets: &[T],
    lr: T,
) {
    // compute per-output gradients (dL/dp) based on final predictions/targets
    let gradients = loss_fn.derivative(predictions, targets);

    // Expect one gradient value per output neuron
    assert_eq!(
        gradients.len(),
        OUT,
        "number of gradients must equal OUT (predictions length)"
    );

    // Build simple bias gradients = gradients and replicate to form weight gradients.
    // NOTE: without activations/inputs this is a placeholder update strategy;
    // for real backprop you must compute weight gradients from upstream gradients and layer inputs.
    let mut weight_grads = [[T::zero(); IN]; OUT];
    let mut bias_grads = [T::zero(); OUT];

    for i in 0..OUT {
        let g = gradients[i];
        bias_grads[i] = g;
        for j in 0..IN {
            weight_grads[i][j] = g;
        }
    }

    // Apply the same computed gradients to each layer (propagating/update order: last -> first)
    for layer in layers.iter_mut().rev() {
        layer.update_weights(&weight_grads, &bias_grads, lr);
    }
}