//! Cellular graph: arena of rooms with Fibonacci topology.
//!
//! New rooms connect to the two previous rooms in creation order, yielding
//! a sparse, scale-free-like structure. The graph orchestrates simulation
//! ticks, gossip dissemination, and 3-phase GC.

use crate::gc::{compact_edges, mark_reachable, GarbageCollector, GCPhase, Mark};
use crate::gossip::GossipPayload;
use crate::ledger::{Ledger, RoomId, Transaction};
use crate::room::Room;

/// The cellular graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph<const DIM: usize, const CAP: usize> {
    pub rooms: Vec<Room<DIM, CAP>>,
    pub edges: Vec<Vec<RoomId>>,
    pub free_list: Vec<RoomId>,
    pub gc: GarbageCollector,
    pub epoch: u64,
    pub next_id: usize,
}

impl<const DIM: usize, const CAP: usize> Graph<DIM, CAP> {
    pub fn new(gc_threshold: f32) -> Self {
        Self {
            rooms: Vec::new(),
            edges: Vec::new(),
            free_list: Vec::new(),
            gc: GarbageCollector::new(gc_threshold),
            epoch: 0,
            next_id: 0,
        }
    }

    /// Add a new room, connecting via Fibonacci adjacency.
    pub fn add_room(
        &mut self,
        lr: f32,
        beta: f32,
        vibe_alpha: f32,
        vibe_decay: f32,
    ) -> RoomId {
        // Reuse from free list if available.
        let id = if let Some(recycled) = self.free_list.pop() {
            let idx = recycled.0;
            self.rooms[idx].reactivate(recycled);
            recycled
        } else {
            let id = RoomId(self.next_id);
            self.next_id += 1;
            let room = Room::new(id, lr, beta, vibe_alpha, vibe_decay);
            self.rooms.push(room);
            self.edges.push(Vec::new());
            id
        };

        // Fibonacci adjacency: connect to previous two rooms.
        let idx = id.0;
        if idx >= 2 {
            let prev1 = RoomId(idx - 1);
            let prev2 = RoomId(idx - 2);
            if prev1.0 < self.rooms.len() && self.rooms[prev1.0].active {
                self.edges[idx].push(prev1);
                self.edges[prev1.0].push(id);
            }
            if prev2.0 < self.rooms.len() && self.rooms[prev2.0].active {
                self.edges[idx].push(prev2);
                self.edges[prev2.0].push(id);
            }
        }

        id
    }

    /// Run one simulation tick: predict, gossip, perceive received gossip, decay.
    pub fn tick(&mut self) {
        self.epoch += 1;
        let epoch = self.epoch;
        let room_count = self.rooms.len();

        // Phase 1: Predict and enqueue gossip.
        for i in 0..room_count {
            if !self.rooms[i].active {
                continue;
            }
            self.rooms[i].predict();
            if self.rooms[i].gossip.tick() {
                self.rooms[i].enqueue_gossip(epoch);
            }
        }

        // Phase 2: Disseminate gossip along edges.
        let mut deliveries: Vec<Vec<GossipPayload<DIM>>> = vec![Vec::new(); room_count];
        for i in 0..room_count {
            if !self.rooms[i].active {
                continue;
            }
            let payloads = self.rooms[i].gossip.drain_outbox();
            for payload in payloads {
                for &neighbor in &self.edges[i] {
                    if neighbor.0 < room_count && self.rooms[neighbor.0].active {
                        deliveries[neighbor.0].push(payload);
                    }
                }
            }
        }
        for i in 0..room_count {
            if !self.rooms[i].active {
                continue;
            }
            for payload in &deliveries[i] {
                self.rooms[i].gossip.receive(*payload);
            }
            // Incorporate average gossip into Z_in as synthetic perception.
            if let Some(avg_z) = self.rooms[i].gossip.avg_inbox_z_out() {
                self.rooms[i].perceive(&avg_z, epoch);
            }
        }

        // Phase 3: Decay vibes.
        for i in 0..room_count {
            if self.rooms[i].active {
                self.rooms[i].tick_vibe();
            }
        }

        // Phase 4: Run GC step if needed.
        self.gc_step();
    }

    /// Inject a sensor reading into a room.
    pub fn inject(&mut self, room: RoomId, vec: &[f32; DIM]) {
        if room.0 < self.rooms.len() && self.rooms[room.0].active {
            self.rooms[room.0].perceive(vec, self.epoch);
        }
    }

    /// Create a transaction between two rooms.
    pub fn transact(&mut self, from: RoomId, to: RoomId, amount: f32) {
        let txn = Transaction::new(from, to, amount, self.epoch);
        if from.0 < self.rooms.len() && self.rooms[from.0].active {
            self.rooms[from.0].transact(txn.clone(), self.epoch);
        }
        if to.0 < self.rooms.len() && self.rooms[to.0].active {
            self.rooms[to.0].transact(txn, self.epoch);
        }
    }

    /// Compute mean vibe across all active rooms.
    pub fn mean_vibe(&self) -> f32 {
        let (sum, count) = self
            .rooms
            .iter()
            .filter(|r| r.active)
            .fold((0.0f32, 0usize), |(s, c), r| (s + r.vibe.level, c + 1));
        if count == 0 {
            0.0
        } else {
            sum / (count as f32)
        }
    }

    /// Compute global ledger.
    pub fn global_ledger(&self) -> Ledger {
        let mut ledger = Ledger::new();
        for room in &self.rooms {
            if room.active {
                for txn in room.ledger.transactions() {
                    ledger.append(txn.clone());
                }
            }
        }
        ledger
    }

    /// Step the GC by one phase.
    fn gc_step(&mut self) {
        match self.gc.phase {
            GCPhase::Idle => {
                // Trigger GC every 8 epochs.
                if self.epoch % 8 == 0 {
                    self.gc.start_cycle();
                    self.run_mark();
                    self.gc.advance();
                }
            }
            GCPhase::Mark => {
                self.run_mark();
                self.gc.advance();
            }
            GCPhase::Sweep => {
                self.run_sweep();
                self.gc.advance();
            }
            GCPhase::Compact => {
                self.run_compact();
                self.gc.advance();
            }
        }
    }

    fn run_mark(&mut self) {
        let n = self.rooms.len();
        let mut marks = vec![Mark::White; n];
        let mut roots = Vec::new();

        for i in 0..n {
            if self.rooms[i].active && self.gc.is_root(&self.rooms[i].vibe) {
                roots.push(RoomId(i));
            }
        }

        // If no hot roots, keep all active rooms as roots to avoid total wipe.
        if roots.is_empty() {
            for i in 0..n {
                if self.rooms[i].active {
                    roots.push(RoomId(i));
                }
            }
        }

        mark_reachable(&mut marks, &self.edges, &roots);

        for i in 0..n {
            self.rooms[i].mark = marks[i];
        }
    }

    fn run_sweep(&mut self) {
        for i in 0..self.rooms.len() {
            if self.rooms[i].active && self.rooms[i].mark == Mark::White {
                self.rooms[i].reset();
                self.free_list.push(RoomId(i));
            }
        }
    }

    fn run_compact(&mut self) {
        for i in 0..self.edges.len() {
            let marks: Vec<Mark> = self.rooms.iter().map(|r| r.mark).collect();
            compact_edges(&mut self.edges[i], &marks);
        }
    }

    /// Number of active rooms.
    pub fn active_count(&self) -> usize {
        self.rooms.iter().filter(|r| r.active).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_add_room() {
        let mut g = Graph::<2, 4>::new(1.0);
        let r0 = g.add_room(0.1, 0.9, 0.3, 1.0);
        let r1 = g.add_room(0.1, 0.9, 0.3, 1.0);
        assert_eq!(r0, RoomId(0));
        assert_eq!(r1, RoomId(1));
        assert_eq!(g.active_count(), 2);
    }

    #[test]
    fn test_fibonacci_edges() {
        let mut g = Graph::<2, 4>::new(1.0);
        for _ in 0..5 {
            g.add_room(0.1, 0.9, 0.3, 1.0);
        }
        // Room 2 connects to 0 and 1; Room 3 to 1 and 2; Room 4 to 2 and 3.
        assert!(g.edges[2].contains(&RoomId(0)));
        assert!(g.edges[2].contains(&RoomId(1)));
        assert!(g.edges[3].contains(&RoomId(1)));
        assert!(g.edges[3].contains(&RoomId(2)));
        assert!(g.edges[4].contains(&RoomId(2)));
        assert!(g.edges[4].contains(&RoomId(3)));
    }

    #[test]
    fn test_graph_tick() {
        let mut g = Graph::<2, 4>::new(1.0);
        for _ in 0..3 {
            g.add_room(0.1, 0.9, 0.3, 1.0);
        }
        g.inject(RoomId(0), &[1.0, 0.0]);
        g.tick();
        assert!(g.rooms[0].dual.z_out.len() > 0 || g.rooms[0].dual.z_in.len() > 0);
    }

    #[test]
    fn test_graph_transact() {
        let mut g = Graph::<2, 4>::new(1.0);
        g.add_room(0.1, 0.9, 0.3, 1.0);
        g.add_room(0.1, 0.9, 0.3, 1.0);
        g.transact(RoomId(0), RoomId(1), 10.0);
        assert_eq!(g.rooms[0].ledger.len(), 1);
        assert_eq!(g.rooms[1].ledger.len(), 1);
    }

    #[test]
    fn test_gc_cycle() {
        let mut g = Graph::<2, 4>::new(100.0); // high threshold so no roots
        for _ in 0..4 {
            g.add_room(0.1, 0.9, 0.3, 1.0);
        }
        // Inject high vibe into room 0 to make it a root.
        g.inject(RoomId(0), &[100.0, 100.0]);
        g.tick();
        g.tick();
        g.tick();
        g.tick();
        g.tick();
        g.tick();
        g.tick();
        g.tick(); // triggers GC mark
        // After GC, room 0 should still be active (it's a root).
        assert!(g.rooms[0].active);
    }
}
