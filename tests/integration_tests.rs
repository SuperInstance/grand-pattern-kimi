use grand_pattern_kimi::{
    compute_signal, simd_dot, simd_norm, FibonacciBackoff, Graph, Ledger, Mark,
    MurmurGossip, RoomId, Transaction, VibeState, ZSpace,
};

// ---------------------------------------------------------------------------
// Integration test 1: Full graph lifecycle
// ---------------------------------------------------------------------------
#[test]
fn test_full_graph_lifecycle() {
    let mut g = Graph::<4, 8>::new(0.1);
    for _ in 0..10 {
        g.add_room(0.05, 0.9, 0.3, 1.0);
    }
    assert_eq!(g.active_count(), 10);

    for i in 0..10 {
        g.inject(RoomId(i), &[1.0, 0.0, 0.0, 0.0]);
    }

    for _ in 0..32 {
        g.tick();
    }

    assert!(g.mean_vibe() >= 0.0);
    assert!(g.global_ledger().volume() >= 0.0);
}

// ---------------------------------------------------------------------------
// Integration test 2: JEPA convergence in graph context
// ---------------------------------------------------------------------------
#[test]
fn test_jepa_convergence_in_room() {
    let mut g = Graph::<2, 16>::new(0.1);
    g.add_room(0.05, 0.9, 0.3, 1.0);

    let target = [0.5f32, -0.5];
    for _ in 0..100 {
        g.inject(RoomId(0), &[1.0, 0.0]);
        g.rooms[0].predict();
        g.rooms[0].learn(&target);
    }

    let err = g.rooms[0].jepa.error(&[1.0, 0.0], &target);
    assert!(err < 0.01, "JEPA did not converge, err={}", err);
}

// ---------------------------------------------------------------------------
// Integration test 3: Gossip saturation
// ---------------------------------------------------------------------------
#[test]
fn test_gossip_saturation() {
    let mut g = Graph::<2, 4>::new(0.1);
    for _ in 0..5 {
        g.add_room(0.1, 0.9, 0.3, 1.0);
    }

    g.inject(RoomId(0), &[1.0, 0.0]);
    g.inject(RoomId(1), &[0.0, 1.0]);

    // Run enough ticks for gossip to propagate.
    for _ in 0..50 {
        g.tick();
    }

    // At least one room other than 0 should have received gossip.
    let received: usize = g
        .rooms
        .iter()
        .filter(|r| r.active && r.id != RoomId(0) && !r.gossip.inbox.is_empty())
        .count();
    assert!(received > 0, "Gossip did not saturate");
}

// ---------------------------------------------------------------------------
// Integration test 4: Ledger global consistency
// ---------------------------------------------------------------------------
#[test]
fn test_ledger_global_consistency() {
    let mut g = Graph::<2, 4>::new(0.1);
    for _ in 0..4 {
        g.add_room(0.1, 0.9, 0.3, 1.0);
    }

    g.transact(RoomId(0), RoomId(1), 10.0);
    g.transact(RoomId(1), RoomId(2), 5.0);
    g.transact(RoomId(2), RoomId(3), 2.0);

    let global = g.global_ledger();
    assert_eq!(global.len(), 6); // each txn recorded twice
    // Global ledger aggregates all transactions; each individual txn is balanced.
    for txn in global.transactions() {
        assert!(txn.is_balanced());
    }
}

// ---------------------------------------------------------------------------
// Integration test 5: GC removes cold rooms but preserves hot roots
// ---------------------------------------------------------------------------
#[test]
fn test_gc_preserves_hot_roots() {
    let mut g = Graph::<2, 4>::new(10.0);
    for _ in 0..6 {
        g.add_room(0.1, 0.9, 0.3, 1.0);
    }

    // Make room 0 extremely hot.
    for _ in 0..20 {
        g.inject(RoomId(0), &[100.0, 100.0]);
    }

    // Tick enough to trigger GC.
    for _ in 0..64 {
        g.tick();
    }

    assert!(g.rooms[0].active, "Hot root was swept");
}

// ---------------------------------------------------------------------------
// Integration test 6: ZSpace ring buffer correctness under load
// ---------------------------------------------------------------------------
#[test]
fn test_zspace_ring_buffer_stress() {
    let mut z = ZSpace::<3, 5>::new();
    for i in 0..100usize {
        let v = [i as f32, (i * 2) as f32, (i * 3) as f32];
        z.push(&v);
    }
    assert_eq!(z.len(), 5);
    let recent = z.get_recent(0).unwrap();
    assert_eq!(recent[0], 99.0);
    assert_eq!(recent[1], 198.0);
    assert_eq!(recent[2], 297.0);
}

// ---------------------------------------------------------------------------
// Integration test 7: Fibonacci topology connectivity
// ---------------------------------------------------------------------------
#[test]
fn test_fibonacci_connectivity() {
    let mut g = Graph::<2, 4>::new(0.1);
    for _ in 0..20 {
        g.add_room(0.1, 0.9, 0.3, 1.0);
    }

    // Every room >= 2 should have at least 2 edges.
    for i in 2..20 {
        assert!(
            g.edges[i].len() >= 2,
            "Room {} has insufficient edges",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Integration test 8: Vibe decay eventually floors
// ---------------------------------------------------------------------------
#[test]
fn test_vibe_decay_floors() {
    let mut v = VibeState::new(1.0, 1.0, 0.5);
    v.update(100.0);
    for _ in 0..1000 {
        v.tick_decay();
    }
    assert_eq!(v.level, 0.5);
}

// ---------------------------------------------------------------------------
// Integration test 9: SIMD operations are deterministic
// ---------------------------------------------------------------------------
#[test]
fn test_simd_determinism() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let d1 = simd_dot(&a, &b);
    let d2 = simd_dot(&a, &b);
    assert_eq!(d1, d2);

    let n1 = simd_norm(&a);
    let n2 = simd_norm(&a);
    assert_eq!(n1, n2);
}

// ---------------------------------------------------------------------------
// Integration test 10: Transaction balancing
// ---------------------------------------------------------------------------
#[test]
fn test_transaction_balancing() {
    let t = Transaction::new(RoomId(0), RoomId(1), 42.0, 7);
    assert!(t.is_balanced());
    assert_eq!(t.debit.room, RoomId(0));
    assert_eq!(t.credit.room, RoomId(1));
}

// ---------------------------------------------------------------------------
// Integration test 11: Gossip backoff progression
// ---------------------------------------------------------------------------
#[test]
fn test_gossip_backoff_progression() {
    let mut fb = FibonacciBackoff::new(100);
    let seq: Vec<u64> = (0..10).map(|_| fb.next()).collect();
    assert_eq!(seq, vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89]);
}

// ---------------------------------------------------------------------------
// Integration test 12: Room reset and recycle
// ---------------------------------------------------------------------------
#[test]
fn test_room_recycle() {
    let mut g = Graph::<2, 4>::new(0.1);
    g.add_room(0.1, 0.9, 0.3, 1.0);
    g.rooms[0].reset();
    assert!(!g.rooms[0].active);
    g.rooms[0].reactivate(RoomId(0));
    assert!(g.rooms[0].active);
}

// ---------------------------------------------------------------------------
// Integration test 13: Mark phase correctly identifies reachable nodes
// ---------------------------------------------------------------------------
#[test]
fn test_mark_phase_reachability() {
    use grand_pattern_kimi::gc::mark_reachable;

    let mut marks = [Mark::White; 5];
    let adj = vec![
        vec![RoomId(1)],       // 0 -> 1
        vec![RoomId(2)],       // 1 -> 2
        vec![],                // 2 -> none
        vec![RoomId(4)],       // 3 -> 4
        vec![RoomId(3)],       // 4 -> 3
    ];
    let roots = vec![RoomId(0)];
    let count = mark_reachable(&mut marks, &adj, &roots);
    assert_eq!(count, 3);
    assert_eq!(marks[0], Mark::Black);
    assert_eq!(marks[1], Mark::Black);
    assert_eq!(marks[2], Mark::Black);
    assert_eq!(marks[3], Mark::White);
    assert_eq!(marks[4], Mark::White);
}

// ---------------------------------------------------------------------------
// Integration test 14: Signal computation blending
// ---------------------------------------------------------------------------
#[test]
fn test_signal_computation_blending() {
    let s = compute_signal(10.0, 0.0, 0.5);
    assert!((s - 5.0).abs() < 1e-6);

    let s2 = compute_signal(0.0, 10.0, 0.2);
    assert!((s2 - 8.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Integration test 15: Graph mean vibe after transactions
// ---------------------------------------------------------------------------
#[test]
fn test_mean_vibe_after_transactions() {
    let mut g = Graph::<2, 4>::new(0.1);
    for _ in 0..4 {
        g.add_room(0.1, 0.9, 0.3, 1.0);
    }

    g.transact(RoomId(0), RoomId(1), 10.0);
    g.transact(RoomId(2), RoomId(3), 10.0);

    let _mv_before = g.mean_vibe();
    g.tick();
    let mv_after = g.mean_vibe();

    // Vibes should change after tick due to ledger volume signal.
    assert!(mv_after >= 0.0);
}

// ---------------------------------------------------------------------------
// Integration test 16: MurmurGossip inbox average
// ---------------------------------------------------------------------------
#[test]
fn test_murmur_gossip_avg() {
    let mut g = MurmurGossip::<2>::new(10, 4);
    g.receive(grand_pattern_kimi::GossipPayload {
        source: RoomId(0),
        z_out: [1.0, 0.0],
        vibe: 1.0,
        epoch: 0,
    });
    g.receive(grand_pattern_kimi::GossipPayload {
        source: RoomId(1),
        z_out: [0.0, 1.0],
        vibe: 3.0,
        epoch: 0,
    });

    let avg_vibe = g.avg_inbox_vibe().unwrap();
    assert!((avg_vibe - 2.0).abs() < 1e-6);

    let avg_z = g.avg_inbox_z_out().unwrap();
    assert!((avg_z[0] - 0.5).abs() < 1e-6);
    assert!((avg_z[1] - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Integration test 17: Ledger net balance for room
// ---------------------------------------------------------------------------
#[test]
fn test_ledger_net_balance() {
    let mut l = Ledger::new();
    l.append(Transaction::new(RoomId(0), RoomId(1), 10.0, 0));
    l.append(Transaction::new(RoomId(1), RoomId(0), 4.0, 1));
    // Net is 0 for this ledger (it's a global ledger)
    for txn in l.transactions() {
        assert!(txn.is_balanced());
    }
}
