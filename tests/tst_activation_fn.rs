use neuralnet::activation_fn::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_layer_f32() {
        let input = [0.0f32, 1.0, -1.0];
        let output = sigmoid_layer::<f32, 3>(&input);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.7310586).abs() < 1e-6);
        assert!((output[2] - 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn test_relu_layer_f32() {
        let input = [-2.0f32, 0.0, 3.0];
        let output = relu_layer::<f32, 3>(&input);
        assert_eq!(output, [0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_tanh_layer_f32() {
        let input = [0.0f32, 1.0, -1.0];
        let output = tanh_layer::<f32, 3>(&input);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.7615942).abs() < 1e-6);
        assert!((output[2] + 0.7615942).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_layer_f64() {
        let input = [0.0f64, 2.0, -2.0];
        let output = sigmoid_layer::<f64, 3>(&input);
        assert!((output[0] - 0.5).abs() < 1e-12);
        assert!((output[1] - 0.8807971).abs() < 1e-7);
        assert!((output[2] - 0.1192029).abs() < 1e-7);
    }

    #[test]
    fn test_relu_layer_f64() {
        let input = [-2.0f64, 0.0, 3.0];
        let output = relu_layer::<f64, 3>(&input);
        assert_eq!(output, [0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_tanh_layer_f64() {
        let input = [0.0f64, 2.0, -2.0];
        let output = tanh_layer::<f64, 3>(&input);
        assert!((output[0] - 0.0).abs() < 1e-12);
        assert!((output[1] - 0.9640275).abs() < 1e-7);
        assert!((output[2] + 0.9640275).abs() < 1e-7);
    }

    #[test]
    fn test_activation_forward_sigmoid() {
        let inputs = [0.0f32, 1.0, -1.0];
        let act = Activation::Sigmoid;
        let output = act.forward(&inputs);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.7310586).abs() < 1e-6);
        assert!((output[2] - 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn test_activation_forward_relu() {
        let inputs = [-2.0f32, 0.0, 3.0];
        let act = Activation::ReLU;
        let output = act.forward(&inputs);
        assert_eq!(output, [0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_activation_forward_tanh() {
        let inputs = [0.0f32, 1.0, -1.0];
        let act = Activation::Tanh;
        let output = act.forward(&inputs);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.7615942).abs() < 1e-6);
        assert!((output[2] + 0.7615942).abs() < 1e-6);
    }

    #[test]
    fn test_activation_derivative_sigmoid() {
        let act = Activation::Sigmoid;
        let x = 0.0f32;
        let d = act.derivative(x);
        // sigmoid(0) = 0.5, derivative = 0.5 * (1 - 0.5) = 0.25
        assert!((d - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_activation_derivative_relu() {
        let act = Activation::ReLU;
        assert_eq!(act.derivative(2.0f32), 1.0);
        assert_eq!(act.derivative(0.0f32), 0.0);
        assert_eq!(act.derivative(-1.0f32), 0.0);
    }

    #[test]
    fn test_activation_derivative_tanh() {
        let act = Activation::Tanh;
        let x = 0.0f32;
        let d = act.derivative(x);
        // tanh(0) = 0, derivative = 1 - 0^2 = 1
        assert!((d - 1.0).abs() < 1e-6);

        let x = 1.0f32;
        let t = x.tanh();
        let expected = 1.0 - t * t;
        assert!((act.derivative(x) - expected).abs() < 1e-6);
    }
}