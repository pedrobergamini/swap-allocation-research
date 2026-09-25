//! Quotes each experiment pool directly from a recording, one JSON line per quote.
//!
//! Supports comparison of supplied quote tables with Fynd's simulation of the same state.
//! A pool's state at a block is the last state the recording carries for it at or
//! before that block, which is what Fynd's replay holds after applying the same prefix.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use num_bigint::BigUint;
use serde::Serialize;
use swap_allocation_research::{
    pools::{Direction, POOLS},
    recording,
};

#[derive(Parser)]
#[command(about = "Quote the two experiment pools from a recording")]
struct Cli {
    #[arg(long)]
    recording: PathBuf,
    #[arg(long, value_delimiter = ',', required = true)]
    blocks: Vec<u64>,
    #[arg(long)]
    direction: Direction,
    /// Input amounts in base units, comma separated.
    #[arg(long, value_delimiter = ',', required = true)]
    amounts: Vec<BigUint>,
}

#[derive(Serialize)]
struct QuoteRow {
    block: u64,
    pool: String,
    direction: Direction,
    amount_in: String,
    amount_out: Option<String>,
    gas: Option<String>,
    error: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let market = recording::load(&cli.recording)?;
    for &block in &cli.blocks {
        let updates = recording::two_pool_prefix(&market, block)?;
        for pool in POOLS {
            let id = pool.to_string();
            let state = updates
                .iter()
                .rev()
                .find_map(|update| update.states.get(&id))
                .with_context(|| format!("no state for {id} by block {block}"))?;
            let component = updates
                .iter()
                .rev()
                .find_map(|update| update.new_pairs.get(&id))
                .with_context(|| format!("no component for {id}"))?;
            let token = |address| {
                component
                    .tokens
                    .iter()
                    .find(|token| token.address == address)
                    .with_context(|| format!("{id} has no token {address}"))
            };
            let token_in = token(cli.direction.token_in())?;
            let token_out = token(cli.direction.token_out())?;
            for amount in &cli.amounts {
                let quoted = state.get_amount_out(amount.clone(), token_in, token_out);
                let row = QuoteRow {
                    block,
                    pool: id.clone(),
                    direction: cli.direction,
                    amount_in: amount.to_string(),
                    amount_out: quoted.as_ref().ok().map(|q| q.amount.to_string()),
                    gas: quoted.as_ref().ok().map(|q| q.gas.to_string()),
                    error: quoted.as_ref().err().map(|e| e.to_string()),
                };
                println!("{}", serde_json::to_string(&row)?);
            }
        }
    }
    Ok(())
}
