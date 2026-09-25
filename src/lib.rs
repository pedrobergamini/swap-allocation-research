//! Learned warm starts for splitting an exact-input WETH/USDC swap between two Base pools.
//!
//! A small trained model proposes pool 1's share of the order; Fynd's Water-fill refines that
//! proposal with its unchanged exchange routine. The runner measures complete quote latency and
//! gas-adjusted output against unmodified Fynd and simpler starting points.

pub mod allocation;
pub mod features;
pub mod model;
pub mod pools;
pub mod recording;
#[cfg(feature = "learned")]
pub mod seeds;
