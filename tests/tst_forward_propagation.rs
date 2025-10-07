use neuralnet::forward_propagation::*;
use neuralnet::layers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_linear_f32() {
        // 2 inputs, 2 outputs
        let inputs = [1.0f32, 2.0];
        let weights = [0.5, -1.0];
        let biases = [0.1, -0.2];
        let outputs = dense_linear(&inputs, &Layer1D { weights, biases });
        // outputs[0] = 0.1 + (1.0 * 0.5) + (2.0 * 0.5) = 0.1 + 0.5 + 1.0 = 1.6
        // outputs[1] = -0.2 + (1.0 * -1.0) + (2.0 * -1.0) = -0.2 - 1.0 - 2.0 = -3.2
        dbg!(outputs);
        assert!((outputs[0] - 1.6).abs() < 1e-6);
        assert!((outputs[1] + 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_dense_linear_i32() {
        // 3 inputs, 2 outputs
        let inputs = [2i32, 1, 3];
        let weights = [1, 2, 3];
        let biases = [1, -1, 1];
        let outputs = dense_linear(&inputs, &Layer1D { weights, biases });
        // outputs[0] = 1 + (2*1) + (1*1) + (3*1) = 1 + 2 + 2 + 3 = 8
        // outputs[1] = -1 + (2*1) + (2*2) + (2*3) = -1 + 2 + 4 + 6 = 11
        // outputs[1] = 1 + (3*1) + (3*2) + (3*3) = 1 + 3 + 6 + 9 = 19
        dbg!(outputs);
        assert_eq!(outputs, [7, 11, 19]);
    }

    #[test]
    fn test_dense_conv2d_f32() {
        // 3 inputs, 2 filters of size 3
        let inputs = [1.0f32, 2.0, 3.0];
        let filters = [
            [0.5, -1.0, 0.2], // filter for output 0
            [1.5, 2.0, -0.3], // filter for output 1
        ];
        let biases = [0.1, -0.2];
        let outputs = dense_conv2d(&inputs, &Layer2D { filters, biases });
        // outputs[0] = 0.1 + (1.0*0.5) + (2.0*-1.0) + (3.0*0.2) = 0.1 + 0.5 -2.0 + 0.6 = -0.8
        // outputs[1] = -0.2 + (1.0*1.5) + (2.0*2.0) + (3.0*-0.3) = -0.2 + 1.5 + 4.0 -0.9 = 4.4
        assert!((outputs[0] + 0.8).abs() < 1e-6);
        assert!((outputs[1] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_dense_conv2d_i32() {
        // 2 inputs, 2 filters of size 3 (extra filter size, should ignore last weight)
        let inputs = [2i32, 3];
        let filters = [
            [1, 2, 3], // filter for output 0
            [4, 5, 6], // filter for output 1
        ];
        let biases = [0, 1];
        let outputs = dense_conv2d(&inputs, &Layer2D { filters, biases });
        // outputs[0] = 0 + (2*1) + (3*2) = 0 + 2 + 6 = 8
        // outputs[1] = 1 + (2*4) + (3*5) = 1 + 8 + 15 = 24
        assert_eq!(outputs, [8, 24]);
    }
}