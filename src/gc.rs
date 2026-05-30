//! Three-phase garbage collector: Mark, Sweep, Compact.
//!
//! Operates on the cellular graph. Roots are rooms with vibe above a threshold.
//! Reachability is determined by graph edges. Unreachable rooms are zeroed and
//! recycled; edge lists are compacted.

use crate::ledger::RoomId;
use crate::vibe::VibeState;

/// GC phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GCPhase {
    Idle,
    Mark,
    Sweep,
    Compact,
}

/// Per-room mark state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    White, // Unvisited
    Grey,  // Discovered, not processed
    Black, // Processed, reachable
}

/// GC controller.
#[derive(Debug, Clone, PartialEq)]
pub struct GarbageCollector {
    pub phase: GCPhase,
    pub threshold: f32,
    pub epoch: u64,
}

impl GarbageCollector {
    pub fn new(threshold: f32) -> Self {
        Self {
            phase: GCPhase::Idle,
            threshold,
            epoch: 0,
        }
    }

    /// Start a new GC cycle.
    pub fn start_cycle(&mut self) {
        self.phase = GCPhase::Mark;
        self.epoch += 1;
    }

    /// Advance phase.
    pub fn advance(&mut self) {
        self.phase = match self.phase {
            GCPhase::Idle => GCPhase::Idle,
            GCPhase::Mark => GCPhase::Sweep,
            GCPhase::Sweep => GCPhase::Compact,
            GCPhase::Compact => GCPhase::Idle,
        };
    }

    /// Determine if a room is a GC root based on vibe.
    pub fn is_root(&self, vibe: &VibeState) -> bool {
        vibe.is_hot(self.threshold)
    }
}

/// Mark reachable rooms from a set of roots using BFS.
pub fn mark_reachable(
    marks: &mut [Mark],
    adjacency: &[Vec<RoomId>],
    roots: &[RoomId],
) -> usize {
    let mut queue: Vec<RoomId> = roots.to_vec();
    let mut marked = 0usize;

    for r in roots {
        if r.0 < marks.len() {
            marks[r.0] = Mark::Grey;
        }
    }

    while let Some(current) = queue.pop() {
        let idx = current.0;
        if idx >= marks.len() || marks[idx] == Mark::Black {
            continue;
        }
        marks[idx] = Mark::Black;
        marked += 1;

        for neighbor in &adjacency[idx] {
            let nidx = neighbor.0;
            if nidx < marks.len() && marks[nidx] == Mark::White {
                marks[nidx] = Mark::Grey;
                queue.push(*neighbor);
            }
        }
    }

    marked
}

/// Compact an edge list by removing references to swept (White) rooms.
pub fn compact_edges(edges: &mut Vec<RoomId>, marks: &[Mark]) {
    edges.retain(|r| r.0 < marks.len() && marks[r.0] == Mark::Black);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_phase_cycle() {
        let mut gc = GarbageCollector::new(1.0);
        assert_eq!(gc.phase, GCPhase::Idle);
        gc.start_cycle();
        assert_eq!(gc.phase, GCPhase::Mark);
        gc.advance();
        assert_eq!(gc.phase, GCPhase::Sweep);
        gc.advance();
        assert_eq!(gc.phase, GCPhase::Compact);
        gc.advance();
        assert_eq!(gc.phase, GCPhase::Idle);
    }

    #[test]
    fn test_mark_reachable() {
        let mut marks = [Mark::White; 4];
        let adj = vec![
            vec![RoomId(1)],
            vec![RoomId(2)],
            vec![RoomId(3)],
            vec![],
        ];
        let roots = vec![RoomId(0)];
        let count = mark_reachable(&mut marks, &adj, &roots);
        assert_eq!(count, 4);
        assert!(marks.iter().all(|&m| m == Mark::Black));
    }

    #[test]
    fn test_mark_with_island() {
        let mut marks = [Mark::White; 4];
        let adj = vec![
            vec![RoomId(1)],
            vec![RoomId(0)],
            vec![RoomId(3)],
            vec![RoomId(2)],
        ];
        let roots = vec![RoomId(0)];
        let count = mark_reachable(&mut marks, &adj, &roots);
        assert_eq!(count, 2);
        assert_eq!(marks[0], Mark::Black);
        assert_eq!(marks[1], Mark::Black);
        assert_eq!(marks[2], Mark::White);
        assert_eq!(marks[3], Mark::White);
    }

    #[test]
    fn test_compact_edges() {
        let marks = [Mark::Black, Mark::White, Mark::Black, Mark::White];
        let mut edges = vec![RoomId(0), RoomId(1), RoomId(2), RoomId(3)];
        compact_edges(&mut edges, &marks);
        assert_eq!(edges, vec![RoomId(0), RoomId(2)]);
    }
}
