//! The two pools and tokens this experiment is restricted to, in their fixed order.
//!
//! Every allocation, probe and feature is laid out pool 1 first. The model never sees which pool
//! Fynd happened to rank first, so the stable order is what keeps its output meaningful.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use tycho_simulation::tycho_common::models::Address;

/// Uniswap V3 WETH/USDC, 0.05% fee, on Base.
pub const POOL1: &str = "0xd0b53d9277642d899df5c87a3966a349a798f224";
/// Uniswap V3 WETH/USDC, 0.01% fee, on Base.
pub const POOL2: &str = "0xb4cb800910b228ed3d0834cf79d697127bbb00e5";
/// The two pools in allocation order.
pub const POOLS: [&str; 2] = [POOL1, POOL2];

pub const WETH: &str = "0x4200000000000000000000000000000000000006";
pub const USDC: &str = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913";

/// Which token an exact-input order sells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    /// Sell WETH for USDC.
    WethUsdc,
    /// Sell USDC for WETH.
    UsdcWeth,
}

impl Direction {
    pub fn token_in(self) -> Address {
        address(match self {
            Direction::WethUsdc => WETH,
            Direction::UsdcWeth => USDC,
        })
    }

    pub fn token_out(self) -> Address {
        address(match self {
            Direction::WethUsdc => USDC,
            Direction::UsdcWeth => WETH,
        })
    }

    /// Decimals of the input and output token.
    pub fn decimals(self) -> (u32, u32) {
        match self {
            Direction::WethUsdc => (18, 6),
            Direction::UsdcWeth => (6, 18),
        }
    }

    /// The model's direction feature: 0 sells WETH, 1 sells USDC.
    pub fn index(self) -> u8 {
        match self {
            Direction::WethUsdc => 0,
            Direction::UsdcWeth => 1,
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Direction::WethUsdc => "weth-usdc",
            Direction::UsdcWeth => "usdc-weth",
        })
    }
}

impl FromStr for Direction {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> anyhow::Result<Self> {
        match value {
            "weth-usdc" => Ok(Direction::WethUsdc),
            "usdc-weth" => Ok(Direction::UsdcWeth),
            other => anyhow::bail!("unknown direction `{other}`"),
        }
    }
}

/// Position of `component_id` in [`POOLS`], matching case-insensitively.
pub fn pool_index(component_id: &str) -> Option<usize> {
    POOLS.iter().position(|pool| pool.eq_ignore_ascii_case(component_id))
}

pub fn address(hex: &str) -> Address {
    Address::from_str(hex).expect("pool and token constants are valid addresses")
}
