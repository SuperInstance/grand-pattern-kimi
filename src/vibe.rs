//! Spectral vibe computation.
//!
//! Vibe is the "energy" of a room, computed from the exponential moving average
//! of L2 norms across its dual vector spaces, modulated by ledger volume and
//! subject to Fibonacci-decay over time.

/// Vibe state for a single room.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VibeState {
    /// Current vibe level.
    pub level: f32,
    /// EMA smoothing factor.
    alpha: f32,
    /// Decay rate (multiplied by inverse Fibonacci ratio each tick).
    pub decay: f32,
    /// Minimum floor.
    floor: f32,
}

impl VibeState {
    pub fn new(alpha: f32, decay: f32, floor: f32) -> Self {
        Self {
            level: 0.0,
            alpha,
            decay,
            floor,
        }
    }

    /// Update vibe given an observed signal (e.g., vector norm + ledger volume).
    pub fn update(&mut self, signal: f32) {
        self.level = self.alpha * signal + (1.0 - self.alpha) * self.level;
        self.level = self.level.max(self.floor);
    }

    /// Apply Fibonacci-style decay: multiply by the inverse golden ratio (~0.618).
    pub fn tick_decay(&mut self) {
        const INV_PHI: f32 = 0.618_034;
        self.level *= self.decay * INV_PHI;
        self.level = self.level.max(self.floor);
    }

    /// Check if vibe is above a threshold.
    pub fn is_hot(&self, threshold: f32) -> bool {
        self.level >= threshold
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.level = 0.0;
    }
}

impl Default for VibeState {
    fn default() -> Self {
        Self::new(0.3, 1.0, 0.0)
    }
}

/// Compute a raw signal from vector norm and ledger volume.
pub fn compute_signal(vector_norm: f32, ledger_volume: f32, norm_weight: f32) -> f32 {
    norm_weight * vector_norm + (1.0 - norm_weight) * ledger_volume
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vibe_update() {
        let mut v = VibeState::new(0.5, 1.0, 0.0);
        v.update(10.0);
        assert!((v.level - 5.0).abs() < 1e-6);
        v.update(0.0);
        assert!((v.level - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_vibe_decay() {
        let mut v = VibeState::new(1.0, 1.0, 0.0);
        v.update(10.0);
        v.tick_decay();
        assert!(v.level < 10.0);
        assert!(v.level > 0.0);
    }

    #[test]
    fn test_vibe_floor() {
        let mut v = VibeState::new(0.5, 1.0, 2.0);
        v.update(1.0);
        assert!(v.level >= 2.0);
    }

    #[test]
    fn test_compute_signal() {
        let s = compute_signal(10.0, 0.0, 0.7);
        assert!((s - 7.0).abs() < 1e-6);
    }
}
