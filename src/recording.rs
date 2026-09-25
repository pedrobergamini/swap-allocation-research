//! Reading a market recording as the two-pool market at one checkpoint.
//!
//! A recording is a snapshot followed by one update per block. Replaying a prefix of it rebuilds
//! the market as of the prefix's last block, so one recording yields many checkpoints without any
//! change to Fynd's replay. Every method sees the same restricted prefix.

use std::path::Path;

use anyhow::{bail, Context};
use fynd_test_fixtures::{read_recording, MarketRecording};
use tycho_simulation::protocol::models::Update;

use crate::pools::pool_index;

pub fn load(path: &Path) -> anyhow::Result<MarketRecording> {
    read_recording(path).with_context(|| format!("reading recording {}", path.display()))
}

/// Blocks at which the market can be rebuilt: every update's block, first to last.
pub fn blocks(recording: &MarketRecording) -> Vec<u64> {
    recording.updates.iter().map(|update| update.block_number_or_timestamp).collect()
}

/// The updates up to and including `block`, keeping only the two experiment pools.
///
/// Fails when `block` is not in the recording or either pool has no state by then, rather than
/// solving a market that silently lost a pool.
pub fn two_pool_prefix(recording: &MarketRecording, block: u64) -> anyhow::Result<Vec<Update>> {
    let Some(end) =
        recording.updates.iter().position(|update| update.block_number_or_timestamp == block)
    else {
        bail!("block {block} is not in the recording");
    };
    let updates: Vec<Update> = recording.updates[..=end].iter().map(restrict).collect();

    let mut has_state = [false; 2];
    for update in &updates {
        for id in update.states.keys() {
            if let Some(index) = pool_index(id) {
                has_state[index] = true;
            }
        }
    }
    if has_state != [true, true] {
        bail!("by block {block} the recording holds state for pools {has_state:?} only");
    }
    Ok(updates)
}

fn restrict(update: &Update) -> Update {
    let mut restricted = update.clone();
    restricted.states.retain(|id, _| pool_index(id).is_some());
    restricted.new_pairs.retain(|id, _| pool_index(id).is_some());
    restricted.removed_pairs.retain(|id, _| pool_index(id).is_some());
    restricted
}
