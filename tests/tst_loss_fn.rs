use neuralnet::loss_fn::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_squared_error_f32() {
        let predictions = [1.0f32, 2.0, 3.0];
        let targets = [1.0f32, 2.0, 4.0];
        let mse = mean_squared_error(&predictions, &targets);
        // (0^2 + 0^2 + 1^2) / 3 = 1/3
        assert!((mse - (1.0/3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mean_squared_error_i32() {
        let predictions = [1i32, 2, 3];
        let targets = [1i32, 2, 4];
        let mse = mean_squared_error(&predictions, &targets);
        // (0^2 + 0^2 + 1^2) / 3 = 1/3, but integer division so result is 0
        assert_eq!(mse, 0);
    }

    #[test]
    fn test_mean_squared_error_all_equal() {
        let predictions = [5.0f32, 5.0, 5.0];
        let targets = [5.0f32, 5.0, 5.0];
        let mse = mean_squared_error(&predictions, &targets);
        assert!((mse - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_f32() {
        let predictions = [0.9f32, 0.2f32];
        let targets = [1.0f32, 0.0f32];
        let ce = cross_entropy_loss(&predictions, &targets);
        // -[1*ln(0.9) + 0*ln(0.2)]/2 = -ln(0.9)/2
        let expected = -0.9f32.ln() / 2.0;
        assert!((ce - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_perfect_prediction() {
        let predictions = [1.0f32, 0.0f32];
        let targets = [1.0f32, 0.0f32];
        let ce = cross_entropy_loss(&predictions, &targets);
        // Should be zero (no loss)
        assert!((ce - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss_f32() {
        let prediction = 0.8f32;
        let target = 1.0f32;
        let bce = binary_cross_entropy_loss(prediction, target);
        // -[1*ln(0.8) + 0*ln(0.2)] = -ln(0.8)
        let expected = -prediction.ln();
        assert!((bce - expected).abs() < 1e-6);

        let prediction = 0.2f32;
        let target = 0.0f32;
        let bce = binary_cross_entropy_loss(prediction, target);
        // -[0*ln(0.2) + 1*ln(0.8)] = -ln(0.8)
        let expected = -(1.0 - prediction).ln();
        assert!((bce - expected).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss_perfect_prediction() {
        let prediction = 1.0f32;
        let target = 1.0f32;
        let bce = binary_cross_entropy_loss(prediction, target);
        // Should be zero (no loss)
        dbg!((bce - 0.0).abs());
        assert!((bce - 0.0).abs() < 1e-6);

        let prediction = 0.0f32;
        let target = 0.0f32;
        let bce = binary_cross_entropy_loss(prediction, target);
        // Should be zero (no loss)
        assert!((bce - 0.0).abs() < 1e-6);
    }
}