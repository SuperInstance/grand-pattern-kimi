//! Cellular room: the fundamental node of the graph.
//!
//! Each room encapsulates dual vector DBs, a JEPA mapper, a ledger,
//! a vibe state, a gossip endpoint, and a mark flag for GC.

use crate::gc::Mark;
use crate::gossip::{GossipPayload, MurmurGossip};
use crate::jepa::JEPAMapper;
use crate::ledger::{Ledger, RoomId, Transaction};
use crate::simd;
use crate::vector_db::DualZSpace;
use crate::vibe::{compute_signal, VibeState};

/// A cellular room in the grand pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Room<const DIM: usize, const CAP: usize> {
    pub id: RoomId,
    pub dual: DualZSpace<DIM, CAP>,
    pub jepa: JEPAMapper<DIM, DIM>,
    pub ledger: Ledger,
    pub vibe: VibeState,
    pub gossip: MurmurGossip<DIM>,
    pub mark: Mark,
    pub active: bool,
}

impl<const DIM: usize, const CAP: usize> Room<DIM, CAP> {
    pub fn new(id: RoomId, lr: f32, beta: f32, vibe_alpha: f32, vibe_decay: f32) -> Self {
        let mut jepa = JEPAMapper::new(lr, beta);
        jepa.init_golden();
        Self {
            id,
            dual: DualZSpace::new(),
            jepa,
            ledger: Ledger::new(),
            vibe: VibeState::new(vibe_alpha, vibe_decay, 0.0),
            gossip: MurmurGossip::new(21, 8),
            mark: Mark::White,
            active: true,
        }
    }

    /// Perceive a vector: push to Z_in and update vibe.
    pub fn perceive(&mut self, vec: &[f32; DIM], epoch: u64) {
        self.dual.z_in.push(vec);
        self.update_vibe(epoch);
    }

    /// Predict next state into Z_out using JEPA.
    pub fn predict(&mut self) {
        if let Some(last_in) = self.dual.z_in.newest().copied() {
            let pred = self.jepa.predict(&last_in);
            self.dual.z_out.push(&pred);
        }
    }

    /// Learn JEPA mapping from most recent input to a target.
    pub fn learn(&mut self, target: &[f32; DIM]) {
        if let Some(last_in) = self.dual.z_in.newest().copied() {
            self.jepa.learn(&last_in, target);
        }
    }

    /// Record a transaction and update vibe.
    pub fn transact(&mut self, txn: Transaction, epoch: u64) {
        self.ledger.append(txn);
        self.update_vibe(epoch);
    }

    /// Update vibe from current state.
    pub fn update_vibe(&mut self, _epoch: u64) {
        let norm_in = self
            .dual
            .z_in
            .newest()
            .map_or(0.0, |v| simd::simd_norm(v));
        let signal = compute_signal(norm_in, self.ledger.volume(), 0.7);
        self.vibe.update(signal);
    }

    /// Decay vibe by one tick.
    pub fn tick_vibe(&mut self) {
        self.vibe.tick_decay();
    }

    /// Enqueue gossip payload.
    pub fn enqueue_gossip(&mut self, epoch: u64) {
        if let Some(z_out) = self.dual.z_out.newest().copied() {
            let payload = GossipPayload {
                source: self.id,
                z_out,
                vibe: self.vibe.level,
                epoch,
            };
            self.gossip.enqueue(payload);
        }
    }

    /// Reset the room (used by GC sweep).
    pub fn reset(&mut self) {
        self.dual.clear();
        self.jepa.reset();
        self.ledger.clear();
        self.vibe.reset();
        self.gossip.reset();
        self.mark = Mark::White;
        self.active = false;
    }

    /// Reactivate a recycled room.
    pub fn reactivate(&mut self, id: RoomId) {
        self.id = id;
        self.active = true;
        self.jepa.init_golden();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_perceive() {
        let mut r = Room::<3, 4>::new(RoomId(0), 0.1, 0.9, 0.3, 1.0);
        r.perceive(&[1.0, 2.0, 3.0], 0);
        assert_eq!(r.dual.z_in.len(), 1);
        assert!(r.vibe.level > 0.0);
    }

    #[test]
    fn test_room_predict() {
        let mut r = Room::<2, 4>::new(RoomId(0), 0.1, 0.9, 0.3, 1.0);
        r.perceive(&[1.0, 0.0], 0);
        r.predict();
        assert_eq!(r.dual.z_out.len(), 1);
    }

    #[test]
    fn test_room_learn() {
        let mut r = Room::<2, 4>::new(RoomId(0), 0.05, 0.9, 0.3, 1.0);
        r.perceive(&[1.0, 0.5], 0);
        r.learn(&[0.2, -0.3]);
        assert!(r.dual.z_in.len() > 0);
    }

    #[test]
    fn test_room_reset() {
        let mut r = Room::<2, 4>::new(RoomId(0), 0.1, 0.9, 0.3, 1.0);
        r.perceive(&[1.0, 0.0], 0);
        r.reset();
        assert!(!r.active);
        assert!(r.dual.z_in.is_empty());
    }

    #[test]
    fn test_room_gossip() {
        let mut r = Room::<2, 4>::new(RoomId(0), 0.1, 0.9, 0.3, 1.0);
        r.perceive(&[1.0, 0.0], 0);
        r.predict();
        r.enqueue_gossip(0);
        assert_eq!(r.gossip.outbox.len(), 1);
    }
}
