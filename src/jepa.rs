//! Joint Embedding Predictive Architecture (JEPA) mapping.
//!
//! Each room learns an affine mapping from its Z_in perception space to its
//! Z_out prediction space using local Hebbian-style updates with momentum.

use crate::simd;

/// Learnable affine transform: `y = W * x + b`.
///
/// `DIM_IN` and `DIM_OUT` are const generics so the weight matrix size is
/// known at compile time and lives inline in the struct.
#[derive(Debug, Clone, PartialEq)]
pub struct JEPAMapper<const DIM_IN: usize, const DIM_OUT: usize> {
    /// Row-major weight matrix: `W[i][j]` connects input j to output i.
    weights: [[f32; DIM_IN]; DIM_OUT],
    bias: [f32; DIM_OUT],
    /// Momentum accumulator for weights.
    momentum_w: [[f32; DIM_IN]; DIM_OUT],
    /// Momentum accumulator for bias.
    momentum_b: [f32; DIM_OUT],
    /// Learning rate.
    lr: f32,
    /// Momentum coefficient.
    beta: f32,
}

impl<const DIM_IN: usize, const DIM_OUT: usize> JEPAMapper<DIM_IN, DIM_OUT> {
    pub fn new(lr: f32, beta: f32) -> Self {
        Self {
            weights: [[0.0f32; DIM_IN]; DIM_OUT],
            bias: [0.0f32; DIM_OUT],
            momentum_w: [[0.0f32; DIM_IN]; DIM_OUT],
            momentum_b: [0.0f32; DIM_OUT],
            lr,
            beta,
        }
    }

    /// Xavier-like initialization scaled by golden ratio for deterministic variance.
    pub fn init_golden(&mut self) {
        const PHI: f32 = 1.618_034;
        let scale = (PHI / ((DIM_IN + DIM_OUT) as f32)).sqrt();
        for i in 0..DIM_OUT {
            for j in 0..DIM_IN {
                // Deterministic pseudo-random from coordinates.
                let seed = ((i * 7919 + j * 104729) as f32).sin();
                self.weights[i][j] = seed * scale;
            }
        }
    }

    /// Forward pass: predict output from input.
    pub fn predict(&self, input: &[f32; DIM_IN]) -> [f32; DIM_OUT] {
        let mut out = self.bias;
        for i in 0..DIM_OUT {
            out[i] += simd::simd_dot(&self.weights[i], input);
        }
        out
    }

    /// Compute prediction error (MSE) against a target.
    pub fn error(&self, input: &[f32; DIM_IN], target: &[f32; DIM_OUT]) -> f32 {
        let pred = self.predict(input);
        let diff = simd::simd_sub(target, &pred);
        simd::simd_norm2(&diff) / (DIM_OUT as f32)
    }

    /// Online update using the most recent input and a target (e.g., next actual Z_in).
    pub fn learn(&mut self, input: &[f32; DIM_IN], target: &[f32; DIM_OUT]) {
        let pred = self.predict(input);
        let diff = simd::simd_sub(target, &pred); // target - pred

        for i in 0..DIM_OUT {
            let grad_b = -2.0 * diff[i] / (DIM_OUT as f32);
            self.momentum_b[i] = self.beta * self.momentum_b[i] + grad_b;
            self.bias[i] -= self.lr * self.momentum_b[i];

            for j in 0..DIM_IN {
                let grad_w = grad_b * input[j];
                self.momentum_w[i][j] = self.beta * self.momentum_w[i][j] + grad_w;
                self.weights[i][j] -= self.lr * self.momentum_w[i][j];
            }
        }
    }

    /// Reset all parameters.
    pub fn reset(&mut self) {
        self.weights = [[0.0f32; DIM_IN]; DIM_OUT];
        self.bias = [0.0f32; DIM_OUT];
        self.momentum_w = [[0.0f32; DIM_IN]; DIM_OUT];
        self.momentum_b = [0.0f32; DIM_OUT];
    }

    pub fn weights(&self) -> &[[f32; DIM_IN]; DIM_OUT] {
        &self.weights
    }

    pub fn bias(&self) -> &[f32; DIM_OUT] {
        &self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jepa_predict() {
        let mut j = JEPAMapper::<2, 2>::new(0.1, 0.9);
        j.init_golden();
        let out = j.predict(&[1.0, 0.0]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_jepa_learn_converges() {
        let mut j = JEPAMapper::<2, 2>::new(0.05, 0.9);
        j.init_golden();
        let input = [1.0f32, 0.5];
        let target = [0.2f32, -0.3];

        let initial_err = j.error(&input, &target);
        for _ in 0..200 {
            j.learn(&input, &target);
        }
        let final_err = j.error(&input, &target);
        assert!(final_err < initial_err);
        assert!(final_err < 1e-3);
    }

    #[test]
    fn test_jepa_reset() {
        let mut j = JEPAMapper::<2, 2>::new(0.1, 0.9);
        j.init_golden();
        j.reset();
        assert_eq!(j.weights, [[0.0; 2]; 2]);
    }
}
