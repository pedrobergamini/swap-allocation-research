//! Solves two-pool orders at recorded checkpoints and writes one JSON line per solve.
//!
//! Every method solves the same orders against the same restricted market, gas price, thread
//! count and deadline. Methods take turns in a rotating order so none always runs first, and each
//! solve builds its own swap cache, so no method warms another's. The timer covers the whole
//! `Solver::quote` call: discovery, the coarse pass, probes, inference, refinement and assembly.
//! Replay and derived-data computation happen before any timing.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{bail, Context};
use clap::Parser;
use fynd_core::{Order, OrderSide, PoolConfig, QuoteOptions, QuoteRequest, QuoteStatus, Solver};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use swap_allocation_research::{
    pools::{pool_index, Direction},
    recording,
};
use tycho_simulation::tycho_common::models::{Address, Chain};

#[derive(Parser)]
#[command(about = "Benchmark two-pool allocation methods on recorded Base checkpoints")]
struct Cli {
    #[arg(long)]
    recording: PathBuf,
    /// Checkpoint blocks, comma separated. Each must be a block in the recording.
    #[arg(long, value_delimiter = ',', required = true)]
    blocks: Vec<u64>,
    /// JSON file: `{"cases": [{"id", "direction", "amount_in"}]}`.
    #[arg(long)]
    cases: PathBuf,
    /// Methods, comma separated: water_fill, coarse, interpolation, learned, offset,
    /// offset_confident, offset_certified, quadratic, reference.
    #[arg(long, value_delimiter = ',', default_value = "water_fill")]
    methods: Vec<String>,
    /// Gas price every solve uses. Explicit, so replay never falls back to Fynd's 10 gwei default.
    #[arg(long)]
    gas_price_wei: u128,
    /// Model artifact for the `learned` method (probe-impact features).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Model artifact for the `offset`, `offset_confident` and `offset_certified` methods.
    #[arg(long)]
    offset_model: Option<PathBuf>,
    /// Confidence artifact (`regret_estimate` output) for `offset_confident` and `offset_certified`.
    #[arg(long)]
    confidence_model: Option<PathBuf>,
    /// For `offset_certified`: the certificate's step is `amount_in / this`, chosen on validation
    /// by `warm_starts.confidence`.
    #[arg(long)]
    certificate_step_divisor: Option<u32>,
    /// Timed passes over every case; the first is preceded by one untimed warm-up solve per method.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 5_000)]
    timeout_ms: u64,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CaseFile {
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct Case {
    id: String,
    direction: Direction,
    /// Input token base units, as a decimal string.
    amount_in: String,
}

#[derive(Debug, Serialize)]
struct SolveRow<'a> {
    build: &'static str,
    block: u64,
    gas_price_wei: String,
    case_id: &'a str,
    direction: Direction,
    amount_in: &'a str,
    method: &'a str,
    repeat: usize,
    status: String,
    amount_out: Option<String>,
    net_out: Option<String>,
    gas: Option<String>,
    /// Input routed to pool 1 and pool 2 by the returned route.
    pool_amounts: Option<[String; 2]>,
    elapsed_us: u128,
    #[cfg(feature = "learned")]
    seeds: Vec<swap_allocation_research::seeds::SeedRecord>,
}

const BUILD: &str = if cfg!(feature = "learned") { "patched" } else { "clean" };
const READY_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let cases: CaseFile = serde_json::from_str(
        &std::fs::read_to_string(&cli.cases)
            .with_context(|| format!("reading {}", cli.cases.display()))?,
    )?;
    for case in &cases.cases {
        let amount: BigUint = case.amount_in.parse().with_context(|| case.id.clone())?;
        if amount == BigUint::from(0u8) {
            bail!("case {} has a zero amount", case.id);
        }
    }
    let methods = Methods::new(&cli)?;
    let market = recording::load(&cli.recording)?;
    let gas_price = BigUint::from(cli.gas_price_wei);
    let mut out = BufWriter::new(File::create(&cli.output)?);

    for &block in &cli.blocks {
        let updates = recording::two_pool_prefix(&market, block)?;
        let solver = Solver::from_recording_with(
            Chain::Base,
            updates,
            methods.pool_configs(cli.timeout_ms)?,
            Some(gas_price.clone()),
            None,
            &methods.registry()?,
        )
        .await?;
        solver.wait_until_ready(READY_TIMEOUT).await?;

        for method in &methods.names {
            let case = &cases.cases[0];
            methods.solve(&solver, method, case, block, None).await;
        }
        for repeat in 0..cli.repeats {
            for (case_index, case) in cases.cases.iter().enumerate() {
                let count = methods.names.len();
                for turn in 0..count {
                    let method = &methods.names[(turn + case_index + repeat) % count];
                    let solved = methods.solve(&solver, method, case, block, Some(repeat)).await;
                    let row = SolveRow {
                        build: BUILD,
                        block,
                        gas_price_wei: gas_price.to_string(),
                        case_id: &case.id,
                        direction: case.direction,
                        amount_in: &case.amount_in,
                        method,
                        repeat,
                        status: solved.status,
                        amount_out: solved.amount_out,
                        net_out: solved.net_out,
                        gas: solved.gas,
                        pool_amounts: solved.pool_amounts,
                        elapsed_us: solved.elapsed_us,
                        #[cfg(feature = "learned")]
                        seeds: solved.seeds,
                    };
                    serde_json::to_writer(&mut out, &row)?;
                    out.write_all(b"\n")?;
                }
            }
        }
        out.flush()?;
        eprintln!(
            "block {block}: {} cases x {} methods done",
            cases.cases.len(),
            methods.names.len()
        );
    }
    Ok(())
}

struct Solved {
    status: String,
    amount_out: Option<String>,
    net_out: Option<String>,
    gas: Option<String>,
    pool_amounts: Option<[String; 2]>,
    elapsed_us: u128,
    #[cfg(feature = "learned")]
    seeds: Vec<swap_allocation_research::seeds::SeedRecord>,
}

/// The requested methods, each served by its own single-worker pool of the same name.
struct Methods {
    names: Vec<String>,
    #[cfg(feature = "learned")]
    log: swap_allocation_research::seeds::SeedLog,
    #[cfg(feature = "learned")]
    model: Option<std::sync::Arc<swap_allocation_research::model::AllocationModel>>,
    #[cfg(feature = "learned")]
    offset_model: Option<std::sync::Arc<swap_allocation_research::model::AllocationModel>>,
    #[cfg(feature = "learned")]
    confidence_model: Option<std::sync::Arc<swap_allocation_research::model::AllocationModel>>,
    #[cfg(feature = "learned")]
    certificate_step_divisor: Option<u32>,
}

impl Methods {
    fn new(cli: &Cli) -> anyhow::Result<Self> {
        for name in &cli.methods {
            let known = matches!(
                name.as_str(),
                "water_fill"
                    | "coarse"
                    | "interpolation"
                    | "learned"
                    | "offset"
                    | "offset_confident"
                    | "offset_certified"
                    | "offset_features"
                    | "quadratic"
                    | "reference"
            );
            if !known {
                bail!("unknown method `{name}`");
            }
            if name != "water_fill" && !cfg!(feature = "learned") {
                bail!("method `{name}` needs the patched build");
            }
        }
        #[cfg(feature = "learned")]
        use swap_allocation_research::model::{FeatureSpec, OutputKind};
        // Loads a model when any of `methods` is requested, checking it has the features and the
        // kind of output they read.
        #[cfg(feature = "learned")]
        let load = |methods: &[&str],
                    path: &Option<PathBuf>,
                    spec,
                    estimates_regret: bool|
         -> anyhow::Result<_> {
            let Some(method) =
                methods.iter().find(|method| cli.methods.iter().any(|m| m == *method))
            else {
                return Ok(None);
            };
            let path =
                path.as_ref().with_context(|| format!("method `{method}` needs its model"))?;
            let model = swap_allocation_research::model::AllocationModel::load(path)?;
            if model.spec() != spec {
                bail!(
                    "{} is a {:?} model; `{method}` needs {spec:?}",
                    path.display(),
                    model.spec()
                );
            }
            let kind = model.output_kind();
            if (kind == OutputKind::RegretEstimate) != estimates_regret {
                bail!("{} has a {kind:?} output, which `{method}` cannot use", path.display());
            }
            Ok(Some(std::sync::Arc::new(model)))
        };
        #[cfg(feature = "learned")]
        let (model, offset_model, confidence_model) = (
            load(&["learned"], &cli.model, FeatureSpec::ProbeImpact, false)?,
            load(
                &["offset", "offset_confident", "offset_certified"],
                &cli.offset_model,
                FeatureSpec::CoarseOffset,
                false,
            )?,
            load(
                &["offset_confident", "offset_certified"],
                &cli.confidence_model,
                FeatureSpec::CoarseOffset,
                true,
            )?,
        );
        Ok(Self {
            names: cli.methods.clone(),
            #[cfg(feature = "learned")]
            log: Default::default(),
            #[cfg(feature = "learned")]
            model,
            #[cfg(feature = "learned")]
            offset_model,
            #[cfg(feature = "learned")]
            confidence_model,
            #[cfg(feature = "learned")]
            certificate_step_divisor: cli.certificate_step_divisor,
        })
    }

    fn algorithm(name: &str) -> String {
        match name {
            "water_fill" => "water_fill".to_string(),
            seeded => format!("water_fill_{seeded}_seed"),
        }
    }

    fn pool_configs(&self, timeout_ms: u64) -> anyhow::Result<HashMap<String, PoolConfig>> {
        self.names
            .iter()
            .map(|name| {
                let table = toml::toml! {
                    algorithm = (Self::algorithm(name))
                    num_workers = 1
                    task_queue_capacity = 1000
                    max_hops = 2
                    timeout_ms = (timeout_ms as i64)
                };
                Ok((name.clone(), toml::Value::Table(table).try_into()?))
            })
            .collect()
    }

    #[cfg(not(feature = "learned"))]
    fn registry(&self) -> anyhow::Result<fynd_core::AlgorithmRegistry> {
        Ok(fynd_core::AlgorithmRegistry::new())
    }

    #[cfg(feature = "learned")]
    fn registry(&self) -> anyhow::Result<fynd_core::AlgorithmRegistry> {
        use std::sync::Arc;

        use fynd_core::algorithm::water_fill::{
            initializer::FineAllocationInitializer, WaterFillAlgorithm,
        };
        use swap_allocation_research::seeds::{
            CoarseSeed, InterpolationSeed, LearnedSeed, OffsetFeatureRecorder, OffsetSeed,
            QuadraticSeed, ReferenceSeed,
        };

        let mut registry = fynd_core::AlgorithmRegistry::new();
        for name in self.names.iter().filter(|name| *name != "water_fill") {
            let log = self.log.clone();
            let seed: Arc<dyn FineAllocationInitializer> = match name.as_str() {
                "coarse" => Arc::new(CoarseSeed { log }),
                "interpolation" => Arc::new(InterpolationSeed { log }),
                "reference" => Arc::new(ReferenceSeed { log }),
                "quadratic" => Arc::new(QuadraticSeed { log }),
                "offset_features" => Arc::new(OffsetFeatureRecorder { log }),
                "offset" => Arc::new(OffsetSeed {
                    log,
                    model: self.offset_model.clone().context("offset without a model")?,
                    confidence: None,
                    certificate_step_divisor: None,
                }),
                "offset_confident" => Arc::new(OffsetSeed {
                    log,
                    model: self.offset_model.clone().context("offset_confident without a model")?,
                    confidence: Some(
                        self.confidence_model
                            .clone()
                            .context("offset_confident without a confidence model")?,
                    ),
                    certificate_step_divisor: None,
                }),
                "offset_certified" => Arc::new(OffsetSeed {
                    log,
                    model: self.offset_model.clone().context("offset_certified without a model")?,
                    confidence: Some(
                        self.confidence_model
                            .clone()
                            .context("offset_certified without a confidence model")?,
                    ),
                    certificate_step_divisor: Some(
                        self.certificate_step_divisor
                            .context("offset_certified without --certificate-step-divisor")?,
                    ),
                }),
                "learned" => Arc::new(LearnedSeed {
                    log,
                    model: self.model.clone().context("learned without a model")?,
                }),
                other => bail!("no seed for `{other}`"),
            };
            registry = registry.with_algorithm(Self::algorithm(name), move |config| {
                WaterFillAlgorithm::with_fine_initializer(config, seed.clone())
                    .expect("Water-fill accepts every pool config the solver built")
            })?;
        }
        Ok(registry)
    }

    async fn solve(
        &self,
        solver: &Solver,
        method: &str,
        case: &Case,
        block: u64,
        repeat: Option<usize>,
    ) -> Solved {
        #[cfg(feature = "learned")]
        self.log.lock().expect("seed log poisoned").clear();
        let amount: BigUint = case.amount_in.parse().expect("validated at load");
        let order = Order::new(
            case.direction.token_in(),
            case.direction.token_out(),
            amount,
            OrderSide::Sell,
            Address::zero(20),
        );
        let request = QuoteRequest::new(
            vec![order],
            QuoteOptions::default().with_worker_pools(vec![method.to_string()]),
        );
        // Marks which solve the per-stage simulation log that follows belongs to; warm-up solves
        // carry no repeat.
        tracing::debug!(target: "sar_bench", block, case = %case.id, method, ?repeat, "solve");
        let started = Instant::now();
        let quote = solver.quote(request).await;
        let elapsed_us = started.elapsed().as_micros();
        #[cfg(feature = "learned")]
        let seeds = std::mem::take(&mut *self.log.lock().expect("seed log poisoned"));

        let mut solved = Solved {
            status: String::new(),
            amount_out: None,
            net_out: None,
            gas: None,
            pool_amounts: None,
            elapsed_us,
            #[cfg(feature = "learned")]
            seeds,
        };
        match quote {
            Err(error) => solved.status = format!("error: {error}"),
            Ok(quote) => {
                let order_quote = &quote.orders()[0];
                if order_quote.status() != QuoteStatus::Success {
                    solved.status = format!("{:?}", order_quote.status());
                    return solved;
                }
                solved.status = "ok".to_string();
                solved.amount_out = Some(order_quote.amount_out().to_string());
                solved.net_out = Some(order_quote.amount_out_net_gas().to_string());
                solved.gas = Some(order_quote.gas_estimate().to_string());
                let mut per_pool = [BigUint::from(0u8), BigUint::from(0u8)];
                let mut foreign = false;
                for swap in order_quote.route().map(|route| route.swaps()).unwrap_or_default() {
                    match pool_index(swap.component_id()) {
                        Some(pool) => per_pool[pool] += swap.amount_in(),
                        None => foreign = true,
                    }
                }
                if foreign {
                    solved.status = "error: route left the two-pool market".to_string();
                }
                solved.pool_amounts = Some(per_pool.map(|amount| amount.to_string()));
            }
        }
        solved
    }
}
