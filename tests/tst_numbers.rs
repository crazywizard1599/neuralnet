use neuralnet::numbers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_and_one() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
    }

    #[test]
    fn test_exp_and_tanh_float() {
        let x: f32 = 1.0;
        assert!((x.exp() - x.exp()).abs() < 1e-6);
        assert!((x.tanh() - x.tanh()).abs() < 1e-6);

        let y: f64 = 2.0;
        assert!((y.exp() - y.exp()).abs() < 1e-12);
        assert!((y.tanh() - y.tanh()).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_exp_int_should_panic() {
        let x: i32 = 2;
        let _ = x.exp();
    }

    #[test]
    #[should_panic]
    fn test_tanh_int_should_panic() {
        let x: i64 = 2;
        let _ = x.tanh();
    }

    #[test]
    fn test_logical_ops_f32() {
        assert_eq!(1.0f32.and(1.0), 1.0);
        assert_eq!(1.0f32.and(0.0), 0.0);
        assert_eq!(0.0f32.or(1.0), 1.0);
        assert_eq!(0.0f32.or(0.0), 0.0);
        assert_eq!(1.0f32.not(), 0.0);
        assert_eq!(0.0f32.not(), 1.0);
    }

    #[test]
    fn test_logical_ops_i32() {
        assert_eq!(1i32.and(1), 1);
        assert_eq!(1i32.and(0), 0);
        assert_eq!(0i32.or(1), 1);
        assert_eq!(0i32.or(0), 0);
        assert_eq!(1i32.not(), 0);
        assert_eq!(0i32.not(), 1);
    }

    #[test]
    fn test_comparisons() {
        assert!(1.0f32.eq(1.0));
        assert!(1.0f32.ne(0.0));
        assert!(2.0f32.gt(1.0));
        assert!(1.0f32.lt(2.0));
        assert!(2.0f32.ge(2.0));
        assert!(1.0f32.le(2.0));

        assert!(2i32.eq(2));
        assert!(2i32.ne(1));
        assert!(3i32.gt(2));
        assert!(2i32.lt(3));
        assert!(3i32.ge(3));
        assert!(2i32.le(3));
    }
}