use neuralnet::forward_propagation::*;
use neuralnet::layers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_linear_f32() {
        // 2 inputs, 2 outputs
        let inputs = [1.0f32, 2.0];
        let weights = [[0.5f32, 0.5], [-1.0, -1.0]]; // weights[0] for output0, weights[1] for output1
        let biases = [0.1f32, -0.2];
        let layer = Layer1D { weights, biases };
        let outputs = dense_linear::<f32, 2, 2>(&inputs, &layer);
        // outputs[0] = 0.1 + 1.0*0.5 + 2.0*0.5 = 1.6
        // outputs[1] = -0.2 + 1.0*(-1.0) + 2.0*(-1.0) = -3.2
        assert!((outputs[0] - 1.6).abs() < 1e-6);
        assert!((outputs[1] + 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_dense_linear_i32() {
        // 3 inputs, 3 outputs
        let inputs = [2i32, 1, 3];
        let weights = [
            [1i32, 2, 3], // output 0
            [4, 5, 6],    // output 1
            [7, 8, 9],    // output 2
        ];
        let biases = [1i32, -1, 1];
        let layer = Layer1D { weights, biases };
        let outputs = dense_linear::<i32, 3, 3>(&inputs, &layer);
        // outputs computed manually:
        // out0 = 1 + 2*1 + 1*2 + 3*3 = 14
        // out1 = -1 + 2*4 + 1*5 + 3*6 = 30
        // out2 = 1 + 2*7 + 1*8 + 3*9 = 50
        assert_eq!(outputs, [14, 30, 50]);
    }

    #[test]
    fn test_dense_conv2d_f32() {
        // 3 inputs, 2 filters of size 3
        let inputs = [1.0f32, 2.0, 3.0];
        let filters = [
            [0.5f32, -1.0, 0.2], // filter for output 0
            [1.5, 2.0, -0.3],    // filter for output 1
        ];
        let biases = [0.1f32, -0.2];
        let layer = Layer2D { filters, biases };
        let outputs = dense_conv2d::<f32, 3, 2, 3>(&inputs, &layer);
        // outputs[0] = 0.1 + (1.0*0.5) + (2.0*-1.0) + (3.0*0.2) = -0.8
        // outputs[1] = -0.2 + (1.0*1.5) + (2.0*2.0) + (3.0*-0.3) = 4.4
        assert!((outputs[0] + 0.8).abs() < 1e-6);
        assert!((outputs[1] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_dense_conv2d_i32() {
        // 2 inputs, 2 filters of size 3 (extra filter element ignored)
        let inputs = [2i32, 3];
        let filters = [
            [1i32, 2, 3], // filter for output 0
            [4, 5, 6],    // filter for output 1
        ];
        let biases = [0i32, 1];
        let layer = Layer2D { filters, biases };
        let outputs = dense_conv2d::<i32, 2, 2, 3>(&inputs, &layer);
        // outputs[0] = 0 + (2*1) + (3*2) = 8
        // outputs[1] = 1 + (2*4) + (3*5) = 24
        assert_eq!(outputs, [8, 24]);
    }
}