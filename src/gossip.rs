//! Murmur gossip protocol with Fibonacci backoff.
//!
//! Rooms gossip their predictions and vibe levels to neighbors. The inter-gossip
//! interval follows the Fibonacci sequence, creating a natural rhythm that
//! reduces contention while preserving eventual consistency.

use crate::ledger::RoomId;
use crate::simd;

/// A gossip payload exchanged between rooms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GossipPayload<const DIM: usize> {
    pub source: RoomId,
    pub z_out: [f32; DIM],
    pub vibe: f32,
    pub epoch: u64,
}

/// Gossip schedule based on Fibonacci sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FibonacciBackoff {
    a: u64,
    b: u64,
    max: u64,
}

impl FibonacciBackoff {
    pub fn new(max: u64) -> Self {
        Self { a: 1, b: 1, max }
    }

    /// Advance and return the next interval.
    pub fn next(&mut self) -> u64 {
        let interval = self.b;
        let next = self.a.saturating_add(self.b);
        self.a = self.b;
        self.b = next.min(self.max);
        interval
    }

    pub fn reset(&mut self) {
        self.a = 1;
        self.b = 1;
    }
}

/// Murmur gossip state for a single room.
#[derive(Debug, Clone, PartialEq)]
pub struct MurmurGossip<const DIM: usize> {
    pub backoff: FibonacciBackoff,
    pub inbox: Vec<GossipPayload<DIM>>,
    pub outbox: Vec<GossipPayload<DIM>>,
    pub tick_counter: u64,
    pub max_inbox: usize,
}

impl<const DIM: usize> MurmurGossip<DIM> {
    pub fn new(backoff_max: u64, max_inbox: usize) -> Self {
        Self {
            backoff: FibonacciBackoff::new(backoff_max),
            inbox: Vec::new(),
            outbox: Vec::new(),
            tick_counter: 0,
            max_inbox,
        }
    }

    /// Enqueue a payload for outgoing gossip.
    pub fn enqueue(&mut self, payload: GossipPayload<DIM>) {
        self.outbox.push(payload);
    }

    /// Receive a payload from a neighbor.
    pub fn receive(&mut self, payload: GossipPayload<DIM>) {
        if self.inbox.len() >= self.max_inbox {
            // Drop oldest (murmur is lossy).
            self.inbox.remove(0);
        }
        self.inbox.push(payload);
    }

    /// Drain outbox for broadcast.
    pub fn drain_outbox(&mut self) -> Vec<GossipPayload<DIM>> {
        core::mem::take(&mut self.outbox)
    }

    /// Compute average received vibe.
    pub fn avg_inbox_vibe(&self) -> Option<f32> {
        if self.inbox.is_empty() {
            return None;
        }
        let sum: f32 = self.inbox.iter().map(|p| p.vibe).sum();
        Some(sum / (self.inbox.len() as f32))
    }

    /// Compute average received Z_out vector.
    pub fn avg_inbox_z_out(&self) -> Option<[f32; DIM]> {
        if self.inbox.is_empty() {
            return None;
        }
        let mut sum = [0.0f32; DIM];
        for p in &self.inbox {
            for i in 0..DIM {
                sum[i] += p.z_out[i];
            }
        }
        let inv = 1.0 / (self.inbox.len() as f32);
        for i in 0..DIM {
            sum[i] *= inv;
        }
        Some(sum)
    }

    /// Tick the gossip timer; returns `true` if it's time to gossip.
    pub fn tick(&mut self) -> bool {
        self.tick_counter += 1;
        let interval = self.backoff.next();
        if self.tick_counter >= interval {
            self.tick_counter = 0;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.backoff.reset();
        self.inbox.clear();
        self.outbox.clear();
        self.tick_counter = 0;
    }
}

/// Compute similarity between two gossip payloads using cosine similarity.
pub fn gossip_similarity<const DIM: usize>(a: &GossipPayload<DIM>, b: &GossipPayload<DIM>) -> f32 {
    let dot = simd::simd_dot(&a.z_out, &b.z_out);
    let na = simd::simd_norm(&a.z_out);
    let nb = simd::simd_norm(&b.z_out);
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_backoff() {
        let mut fb = FibonacciBackoff::new(100);
        assert_eq!(fb.next(), 1);
        assert_eq!(fb.next(), 2);
        assert_eq!(fb.next(), 3);
        assert_eq!(fb.next(), 5);
        assert_eq!(fb.next(), 8);
    }

    #[test]
    fn test_fibonacci_saturates() {
        let mut fb = FibonacciBackoff::new(5);
        assert_eq!(fb.next(), 1);
        assert_eq!(fb.next(), 2);
        assert_eq!(fb.next(), 3);
        assert_eq!(fb.next(), 5);
        assert_eq!(fb.next(), 5);
    }

    #[test]
    fn test_murmur_gossip() {
        let mut g = MurmurGossip::<2>::new(10, 4);
        let p = GossipPayload {
            source: RoomId(0),
            z_out: [1.0, 0.0],
            vibe: 0.5,
            epoch: 0,
        };
        g.enqueue(p);
        assert_eq!(g.outbox.len(), 1);
        let drained = g.drain_outbox();
        assert_eq!(drained.len(), 1);
        assert!(g.outbox.is_empty());
    }

    #[test]
    fn test_gossip_similarity() {
        let a = GossipPayload {
            source: RoomId(0),
            z_out: [1.0, 0.0],
            vibe: 0.5,
            epoch: 0,
        };
        let b = GossipPayload {
            source: RoomId(1),
            z_out: [1.0, 0.0],
            vibe: 0.5,
            epoch: 0,
        };
        assert!((gossip_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }
}
