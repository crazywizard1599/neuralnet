use crate::numbers::Number;
use num_traits::FromPrimitive;

pub fn mean_squared_error<T: Number + FromPrimitive>(predictions: &[T], targets: &[T]) -> T {
    let n = T::to_number(predictions.len() as f64);
    let mut sum = T::zero();
    for i in 0..predictions.len() {
        let diff = predictions[i] - targets[i];
        sum = sum + diff * diff;
    }
    sum / n
}

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

pub fn binary_cross_entropy_loss<T: Number + FromPrimitive>(prediction: T, target: T) -> T {
    let eps = T::to_number(1e-15);
    let p = if prediction < eps { eps } else if prediction > T::one() - eps { T::one() - eps } else { prediction };
    let one_minus_p = if T::one() - p < eps { eps } else { T::one() - p };
    - (target * p.ln() + (T::one() - target) * one_minus_p.ln())
}

pub enum  Loss {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
}

impl Loss {
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