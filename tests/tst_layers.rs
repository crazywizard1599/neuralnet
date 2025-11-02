use neuralnet::layers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_i32_exact_size() {
        let input = vec![1i32, 2, 3, 4];
        // N = 4 outputs, IN = 1 input per output (weights shape [4][1])
        let layer = linear::<i32, 4, 1>(&input);
        assert_eq!(layer.weights, [[1i32], [2], [3], [4]]);
        assert_eq!(layer.biases, [0i32, 0, 0, 0]);
    }

    #[test]
    fn test_linear_i32_shorter_input() {
        let input = vec![1i32, 2];
        let layer = linear::<i32, 4, 1>(&input);
        assert_eq!(layer.weights, [[1i32], [2], [0], [0]]);
        assert_eq!(layer.biases, [0i32, 0, 0, 0]);
    }

    #[test]
    fn test_linear_f64_longer_input() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let layer = linear::<f64, 4, 1>(&input);
        assert_eq!(layer.weights, [[1.0f64], [2.0], [3.0], [4.0]]);
        assert_eq!(layer.biases, [0.0f64, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_conv2d_i32_exact_size() {
        let input = vec![
            1i32, 2, 3, // filter 1
            4, 5, 6     // filter 2
        ];
        let layer = conv2d::<i32, 2, 3>(&input);
        assert_eq!(layer.filters, [[1i32, 2, 3], [4, 5, 6]]);
        assert_eq!(layer.biases, [0i32, 0]);
    }

    #[test]
    fn test_conv2d_i32_shorter_input() {
        let input = vec![
            1i32, 2, 3
        ];
        let layer = conv2d::<i32, 2, 3>(&input);
        assert_eq!(layer.filters, [[1i32, 2, 3], [0, 0, 0]]);
        assert_eq!(layer.biases, [0i32, 0]);
    }

    #[test]
    fn test_conv2d_f64_longer_input() {
        let input = vec![
            1.0f64, 2.0, 3.0, // filter 1
            4.0, 5.0, 6.0,    // filter 2
            7.0, 8.0, 9.0     // excess
        ];
        let layer = conv2d::<f64, 2, 3>(&input);
        assert_eq!(layer.filters, [[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(layer.biases, [0.0f64, 0.0]);
    }
}