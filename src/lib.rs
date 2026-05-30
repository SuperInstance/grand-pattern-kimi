//! # Grand Pattern Fibonacci Dual-Direction Architecture
//!
//! A cellular graph of rooms with dual vector databases, JEPA predictive mapping,
//! double-entry bookkeeping, spectral vibe computation, three-phase garbage
//! collection, and murmur gossip dissemination.
//!
//! ## Design Principles
//! - **Zero dependencies**: Pure Rust, `std` only.
//! - **Const generics**: Compile-time dimensionality and capacity.
//! - **SIMD**: SSE/AVX acceleration on x86_64 with scalar fallback.
//! - **Fibonacci topology**: Sparse, small-world graph generation.

pub mod gc;
pub mod gossip;
pub mod graph;
pub mod jepa;
pub mod ledger;
pub mod room;
pub mod simd;
pub mod vector_db;
pub mod vibe;

pub use gc::{GarbageCollector, GCPhase, Mark};
pub use gossip::{FibonacciBackoff, GossipPayload, MurmurGossip};
pub use graph::Graph;
pub use jepa::JEPAMapper;
pub use ledger::{Ledger, RoomId, Transaction};
pub use room::Room;
pub use simd::{simd_add, simd_dot, simd_mul, simd_norm, simd_norm2, simd_scale};
pub use vector_db::{DualZSpace, ZSpace};
pub use vibe::{compute_signal, VibeState};

/// Convenience type alias for a common graph configuration.
pub type StandardGraph<const DIM: usize> = Graph<DIM, 16>;

/// Run a small demonstration harness.
pub fn demo() {
    const DIM: usize = 4;
    const CAP: usize = 8;
    let mut graph = Graph::<DIM, CAP>::new(0.5);

    // Seed rooms.
    for _ in 0..8 {
        graph.add_room(0.05, 0.9, 0.3, 1.0);
    }

    // Inject sensor data into room 0.
    graph.inject(RoomId(0), &[1.0, 0.0, 0.0, 0.0]);
    graph.inject(RoomId(1), &[0.0, 1.0, 0.0, 0.0]);

    // Simulate 20 ticks.
    for _ in 0..20 {
        graph.tick();
    }

    // Create some transactions.
    graph.transact(RoomId(0), RoomId(1), 5.0);
    graph.transact(RoomId(2), RoomId(3), 3.0);

    println!("Active rooms: {}", graph.active_count());
    println!("Mean vibe: {:.4}", graph.mean_vibe());
    println!("Global ledger volume: {:.4}", graph.global_ledger().volume());
    println!("GC phase: {:?}", graph.gc.phase);
}
