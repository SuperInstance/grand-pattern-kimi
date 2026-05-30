//! Dual vector databases: Z_in (perception) and Z_out (prediction).
//!
//! Each `ZSpace` is a fixed-capacity ring buffer of `DIM`-dimensional vectors.
//! When full, the oldest vector is overwritten. This gives bounded memory and
//! models short-term memory decay.

use crate::simd;

/// A ring-buffer vector database with compile-time dimensionality and capacity.
#[derive(Debug, Clone, PartialEq)]
pub struct ZSpace<const DIM: usize, const CAP: usize> {
    /// Flat storage: `buffer[i * DIM + j]` is the j-th component of the i-th vector.
    buffer: Vec<f32>,
    /// Write position; wraps modulo CAP.
    head: usize,
    /// Current number of stored vectors (saturates at CAP).
    len: usize,
}

impl<const DIM: usize, const CAP: usize> ZSpace<DIM, CAP> {
    pub fn new() -> Self {
        assert!(CAP > 0, "CAP must be > 0");
        assert!(DIM > 0, "DIM must be > 0");
        Self {
            buffer: vec![0.0f32; DIM * CAP],
            head: 0,
            len: 0,
        }
    }

    /// Push a new vector. Overwrites oldest if at capacity.
    pub fn push(&mut self, vec: &[f32; DIM]) {
        let offset = self.head * DIM;
        self.buffer[offset..offset + DIM].copy_from_slice(&vec[..]);
        self.head = (self.head + 1) % CAP;
        if self.len < CAP {
            self.len += 1;
        }
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access the i-th most recent vector (0 = newest).
    pub fn get_recent(&self, i: usize) -> Option<&[f32; DIM]> {
        if i >= self.len {
            return None;
        }
        // head points to next write slot, so newest is at (head + CAP - 1) % CAP
        let idx = (self.head + CAP - 1 - i) % CAP;
        let offset = idx * DIM;
        // Safety: offset is always aligned and within bounds.
        Some(
            self.buffer[offset..offset + DIM]
                .try_into()
                .expect("slice length matches DIM"),
        )
    }

    /// Compute the mean vector of all stored vectors.
    pub fn mean(&self) -> Option<[f32; DIM]> {
        if self.is_empty() {
            return None;
        }
        let mut sum = [0.0f32; DIM];
        for i in 0..self.len {
            let v = self.get_recent(i).unwrap();
            for j in 0..DIM {
                sum[j] += v[j];
            }
        }
        let inv = 1.0 / (self.len as f32);
        for j in 0..DIM {
            sum[j] *= inv;
        }
        Some(sum)
    }

    /// Compute cosine similarity between the most recent vector and `other`.
    pub fn cosine_recent(&self, other: &[f32; DIM]) -> Option<f32> {
        let recent = self.get_recent(0)?;
        let dot = simd::simd_dot(recent, other);
        let n1 = simd::simd_norm(recent);
        let n2 = simd::simd_norm(other);
        if n1 == 0.0 || n2 == 0.0 {
            return Some(0.0);
        }
        Some(dot / (n1 * n2))
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.len = 0;
    }

    /// Return the newest vector if any.
    pub fn newest(&self) -> Option<&[f32; DIM]> {
        self.get_recent(0)
    }
}

impl<const DIM: usize, const CAP: usize> Default for ZSpace<DIM, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

/// Dual ZSpace holding both perception (Z_in) and prediction (Z_out).
#[derive(Debug, Clone, PartialEq)]
pub struct DualZSpace<const DIM: usize, const CAP: usize> {
    pub z_in: ZSpace<DIM, CAP>,
    pub z_out: ZSpace<DIM, CAP>,
}

impl<const DIM: usize, const CAP: usize> DualZSpace<DIM, CAP> {
    pub fn new() -> Self {
        Self {
            z_in: ZSpace::new(),
            z_out: ZSpace::new(),
        }
    }

    pub fn clear(&mut self) {
        self.z_in.clear();
        self.z_out.clear();
    }
}

impl<const DIM: usize, const CAP: usize> Default for DualZSpace<DIM, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zspace_push_and_get() {
        let mut z = ZSpace::<3, 4>::new();
        z.push(&[1.0, 2.0, 3.0]);
        z.push(&[4.0, 5.0, 6.0]);
        assert_eq!(z.len(), 2);
        assert_eq!(z.get_recent(0).unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(z.get_recent(1).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zspace_wrap() {
        let mut z = ZSpace::<2, 3>::new();
        z.push(&[1.0, 1.0]);
        z.push(&[2.0, 2.0]);
        z.push(&[3.0, 3.0]);
        z.push(&[4.0, 4.0]); // overwrites oldest
        assert_eq!(z.len(), 3);
        assert_eq!(z.get_recent(0).unwrap(), &[4.0, 4.0]);
        assert_eq!(z.get_recent(1).unwrap(), &[3.0, 3.0]);
        assert_eq!(z.get_recent(2).unwrap(), &[2.0, 2.0]);
        assert!(z.get_recent(3).is_none());
    }

    #[test]
    fn test_zspace_mean() {
        let mut z = ZSpace::<2, 4>::new();
        z.push(&[1.0, 2.0]);
        z.push(&[3.0, 4.0]);
        let m = z.mean().unwrap();
        assert!((m[0] - 2.0).abs() < 1e-6);
        assert!((m[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_zspace_cosine() {
        let mut z = ZSpace::<2, 2>::new();
        z.push(&[1.0, 0.0]);
        let c = z.cosine_recent(&[1.0, 0.0]).unwrap();
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dual_zspace() {
        let mut dz = DualZSpace::<2, 2>::new();
        dz.z_in.push(&[1.0, 0.0]);
        dz.z_out.push(&[0.0, 1.0]);
        assert_eq!(dz.z_in.len(), 1);
        assert_eq!(dz.z_out.len(), 1);
        dz.clear();
        assert!(dz.z_in.is_empty());
        assert!(dz.z_out.is_empty());
    }
}
