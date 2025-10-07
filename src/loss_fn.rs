//! Loss functions implemented generically over a numeric type `T`.
//!
//! This module provides three common loss functions used in machine learning:
//! - Mean Squared Error (MSE)
//! - Cross-Entropy (element-wise)
//! - Binary Cross-Entropy (scalar, single-prediction binary case)
//!
//! Each function is generic over `T` which is expected to implement the project's
//! `Number` trait (for arithmetic and numeric helpers) and `FromPrimitive` (to
//! construct constants like `2.0` from primitive floats). The implementations
//! assume `T` behaves like a floating-point numeric type for correct results.

use crate::numbers::Number;
use num_traits::FromPrimitive;

/// Compute the **mean squared error (MSE)** between `predictions` and `targets`.
///
/// # Mathematical definition
///
/// $$\mathrm{MSE} = \frac{1}{n} \sum_{i=0}^{n-1} (\text{pred}_i - \text{target}_i)^2$$
///
/// # Steps performed by the function
/// 1. Convert the length `n` of the slices into the generic numeric type `T` using
///    `T::to_number(predictions.len() as f64)`. This yields the denominator used
///    when computing the mean.
/// 2. Initialize an accumulator `sum` to `T::zero()`.
/// 3. Iterate over every index `i` from `0` to `predictions.len() - 1`:
///    - Compute the element-wise difference: `diff = predictions[i] - targets[i]`.
///    - Square it: `diff * diff`.
///    - Add the squared difference to the accumulator: `sum = sum + diff * diff`.
/// 4. Return the mean by dividing the accumulated `sum` by `n` (`sum / n`).
///
/// # Preconditions and notes
/// - `predictions.len()` must equal `targets.len()`; otherwise indexing will panic.
/// - If `predictions.len()` is `0`, `n` becomes zero and the final division `sum / n`
///   will generally be invalid (division by zero). The function does **not** check
///   for empty inputs — the caller should ensure non-empty slices.
/// - `T` is expected to behave like a floating-point numeric type: integer types
///   will produce integer arithmetic semantics which are usually not desired for
///   MSE. Prefer `f32`/`f64` or other floating-point-like `T` implementations.
///
pub fn mean_squared_error<T: Number + FromPrimitive>(predictions: &[T], targets: &[T]) -> T {
    let n = T::to_number(predictions.len() as f64);
    let mut sum = T::zero();
    for i in 0..predictions.len() {
        let diff = predictions[i] - targets[i];
        sum = sum + diff * diff;
    }
    sum / n
}

/// Compute the (element-wise) **cross-entropy loss** between `predictions` and `targets`.
///
/// This function implements the usual cross-entropy term applied element-wise
/// and then averages across the provided arrays:
///
/// $$L = -\frac{1}{n} \sum_{i=0}^{n-1} t_i \ln(p_i)$$
///
/// where `p_i` are predicted probabilities and `t_i` are target probabilities
/// (e.g. one-hot targets or soft-labels).
///
/// # Steps performed by the function
/// 1. Convert the length `n` to `T` with `T::to_number(predictions.len() as f64)`.
/// 2. Initialize an accumulator `sum` to `T::zero()`.
/// 3. For each element `i`:
///    - Clamp `predictions[i]` to a small positive lower bound (`eps = 1e-15`) to
///      avoid `ln(0)`. The code chooses `p = max(predictions[i], eps)` implemented
///      by comparing with `T::to_number(1e-15)`.
///    - Add `- targets[i] * p.ln()` to the accumulator.
/// 4. Return the average `sum / n`.
///
/// # Preconditions and notes
/// - `predictions` should contain values in `[0, 1]` representing probabilities.
/// - `targets` is typically one-hot encoded (0 or 1) or soft labels in `[0, 1]`.
/// - The function prevents `ln(0)` by clamping `p` to `1e-15` converted to `T`.
/// - The natural logarithm (`ln`) is used.
///
pub fn cross_entropy_loss<T: Number + FromPrimitive>(predictions: &[T], targets: &[T]) -> T {
    let n = T::to_number(predictions.len() as f64);
    let mut sum = T::zero();
    for i in 0..predictions.len() {
        // To avoid log(0), we clamp predictions to a small positive value
        let p = if predictions[i] < T::to_number(1e-15) {
            T::to_number(1e-15)
        } else {
            predictions[i]
        };
        sum = sum - targets[i] * p.ln();
    }
    sum / n
}

/// Compute the **binary cross-entropy** (BCE) for a *single* scalar prediction and target.
///
/// This implements the scalar binary cross-entropy term:
///
/// $$L = -\left( t \ln(p) + (1 - t) \ln(1 - p) \right)$$
///
/// where `p` is the predicted probability for the positive class and `t` is the
/// target (commonly `0` or `1`, but soft labels in `[0,1]` are supported).
///
/// # Steps performed by the function
/// 1. Convert `eps = 1e-15` to `T` using `T::to_number(1e-15)`.
/// 2. Clamp `prediction` to the interval `[eps, 1 - eps]` to avoid `ln(0)` and to
///    prevent numerical infinities. This yields the stable `p` used below.
/// 3. Compute `one_minus_p = 1 - p` and clamp it to `eps` as well to keep the
///    logarithm numerically stable.
/// 4. Return `- (target * ln(p) + (1 - target) * ln(1 - p))`.
///
/// # Preconditions and notes
/// - This function is scalar: it computes BCE for a single `prediction` and `target`.
/// - For batched binary BCE, call this per element and average (or implement a
///   batched wrapper).
/// - Clamping uses `1e-15` as a practical epsilon; depending on `T`'s range you
///   may choose a different epsilon.
///
pub fn binary_cross_entropy_loss<T: Number + FromPrimitive>(prediction: T, target: T) -> T {
    let eps = T::to_number(1e-15);
    let p = if prediction < eps { eps } else if prediction > T::one() - eps { T::one() - eps } else { prediction };
    let one_minus_p = if T::one() - p < eps { eps } else { T::one() - p };
    - (target * p.ln() + (T::one() - target) * one_minus_p.ln())
}

/// A small enum wrapper over the implemented loss functions with convenience
/// `forward` and `derivative` helpers.
///
/// - `forward` computes the loss value.
/// - `derivative` computes the derivative of the loss with respect to a single
///   `prediction` scalar (i.e. `dL/d(prediction)`). Important: `derivative`
///   returns the derivative **per sample** (it does not average over a batch).
pub enum Loss {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
}

impl Loss {
    /// Compute the forward loss value for the enum variant.
    ///
    /// # Behavior
    /// - For `MeanSquaredError` and `CrossEntropy` this expects `predictions` and
    ///   `targets` to be slices of the same length and computes the averaged loss.
    /// - For `BinaryCrossEntropy` the function **expects** `predictions.len() == 1`
    ///   and `targets.len() == 1`. If that is not the case it will `panic!` with a
    ///   message indicating the expectation.
    pub fn forward<T: Number + FromPrimitive>(&self, predictions: &[T], targets: &[T]) -> T {
        match self {
            Loss::MeanSquaredError => mean_squared_error(predictions, targets),
            Loss::CrossEntropy => cross_entropy_loss(predictions, targets),
            Loss::BinaryCrossEntropy => {
                if predictions.len() != 1 || targets.len() != 1 {
                    panic!("BinaryCrossEntropy loss expects single prediction and target values.");
                }
                binary_cross_entropy_loss(predictions[0], targets[0])
            }
        }
    }

    /// Compute the derivative of the loss with respect to a *single* prediction.
    ///
    /// # Return value
    /// The returned `T` is the scalar partial derivative `dL / d(prediction)` for
    /// the element passed in. Note that for batched losses you will typically call
    /// this per-element and then average or sum the derivatives as appropriate.
    ///
    /// # Implemented derivatives
    /// - `MeanSquaredError`: returns `2 * (prediction - target)` (no `1/n` factor).
    /// - `CrossEntropy`: returns `- target / prediction` (derivative of `-t ln p`
    ///   with respect to `p`). **This implementation does not clamp `prediction`**
    ///   — take care to avoid division by zero when using this function.
    /// - `BinaryCrossEntropy`: returns `- (target / p) + ((1 - target) / (1 - p))`
    ///   where `p` is clamped to `[eps, 1 - eps]` for numerical stability.
    ///
    /// # Notes on averaging
    /// If you computed `forward` as an average over `n` items, but want to obtain
    /// the gradient of that averaged loss, you must divide the per-sample
    /// derivatives by `n` yourself (i.e. the `derivative` method intentionally
    /// returns the derivative for a single element, not the batch mean's derivative).
    pub fn derivative<T: Number + FromPrimitive>(&self, prediction: T, target: T) -> T {
        match self {
            Loss::MeanSquaredError => T::from_f64(2.0).unwrap() * (prediction - target),
            Loss::CrossEntropy => - target / prediction,
            Loss::BinaryCrossEntropy => {
                let eps = T::to_number(1e-15);
                let p = if prediction < eps { eps } else if prediction > T::one() - eps { T::one() - eps } else { prediction };
                let one_minus_p = if T::one() - p < eps { eps } else { T::one() - p };
                - (target / p) + ((T::one() - target) / one_minus_p)
            }
        }
    }
}
