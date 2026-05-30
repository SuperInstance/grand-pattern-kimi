//! Double-entry bookkeeping ledger.
//!
//! Every transaction has two legs: a debit and a credit. The ledger is append-only,
//! providing an immutable audit trail of all inter-room value transfers.

use core::fmt;

/// Unique identifier for a room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct RoomId(pub usize);

impl fmt::Display for RoomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R{}", self.0)
    }
}

/// A single leg of a transaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leg {
    pub room: RoomId,
    pub amount: f32,
}

/// A double-entry transaction.
#[derive(Debug, Clone, PartialEq)]
pub struct Transaction {
    pub debit: Leg,
    pub credit: Leg,
    pub epoch: u64,
}

impl Transaction {
    pub fn new(debit_room: RoomId, credit_room: RoomId, amount: f32, epoch: u64) -> Self {
        Self {
            debit: Leg {
                room: debit_room,
                amount,
            },
            credit: Leg {
                room: credit_room,
                amount,
            },
            epoch,
        }
    }

    /// Verify that debit and credit amounts balance.
    pub fn is_balanced(&self) -> bool {
        (self.debit.amount - self.credit.amount).abs() < 1e-6
    }
}

/// Append-only ledger for a room or for the whole graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Ledger {
    txns: Vec<Transaction>,
    total_debit: f32,
    total_credit: f32,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            txns: Vec::new(),
            total_debit: 0.0,
            total_credit: 0.0,
        }
    }

    pub fn append(&mut self, txn: Transaction) {
        self.total_debit += txn.debit.amount;
        self.total_credit += txn.credit.amount;
        self.txns.push(txn);
    }

    pub fn len(&self) -> usize {
        self.txns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.txns.is_empty()
    }

    pub fn transactions(&self) -> &[Transaction] {
        &self.txns
    }

    /// Net balance (credit - debit) for this ledger.
    pub fn net_balance(&self) -> f32 {
        self.total_credit - self.total_debit
    }

    /// Sum of absolute transaction volume.
    pub fn volume(&self) -> f32 {
        self.total_debit + self.total_credit
    }

    /// Filter transactions involving a specific room.
    pub fn for_room(&self, room: RoomId) -> Vec<&Transaction> {
        self.txns
            .iter()
            .filter(|t| t.debit.room == room || t.credit.room == room)
            .collect()
    }

    pub fn clear(&mut self) {
        self.txns.clear();
        self.total_debit = 0.0;
        self.total_credit = 0.0;
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_txn() {
        let t = Transaction::new(RoomId(0), RoomId(1), 10.0, 0);
        assert!(t.is_balanced());
    }

    #[test]
    fn test_ledger_volume() {
        let mut l = Ledger::new();
        l.append(Transaction::new(RoomId(0), RoomId(1), 5.0, 0));
        l.append(Transaction::new(RoomId(1), RoomId(2), 3.0, 1));
        assert_eq!(l.len(), 2);
        assert!((l.volume() - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_ledger_for_room() {
        let mut l = Ledger::new();
        l.append(Transaction::new(RoomId(0), RoomId(1), 5.0, 0));
        l.append(Transaction::new(RoomId(1), RoomId(2), 3.0, 1));
        let r0 = l.for_room(RoomId(0));
        assert_eq!(r0.len(), 1);
        let r1 = l.for_room(RoomId(1));
        assert_eq!(r1.len(), 2);
    }
}
