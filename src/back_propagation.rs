use crate::loss_fn::*;
use crate::layers::Layer1D;
use crate::numbers::*;
use num_traits::FromPrimitive;

pub fn backward_pass_1d<T: Number + FromPrimitive, const N: usize>(
    layers: &mut [Layer1D<T, N>],
    loss_fn: Loss,
    predictions: &[T],
    targets: &[T],
    lr: T,
) {
    // compute gradients once (derivative is based on final predictions/targets)
    let gradients = loss_fn.derivative(predictions, targets);

    // iterate mutably so we can update each layer in-place
    for layer in layers.iter_mut().rev() {
        // update_weights should take &mut self; adjust argument to match your implementation:
        // - if update_weights accepts a slice: layer.update_weights(&gradients, lr);
        // - if it accepts Vec<T>: layer.update_weights(gradients.clone(), lr);
        layer.update_weights(&gradients, lr);
    }
}