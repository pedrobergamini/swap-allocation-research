//! Starting allocations for Water-fill's exchange refinement.
//!
//! Each seed replaces only the 256-chunk fine water-fill through the initializer hook; the coarse
//! pass and final selection stay Fynd's own. The offset seed with an allocation-regret
//! estimator may also ask the disjoint split to skip exchange refinement. A seed applies only when the
//! active set is exactly the two experiment pools, each a single hop. Otherwise it declines and
//! Water-fill runs unchanged, and the log says so, so invocation rates are reported rather than
//! assumed.

use std::sync::{Arc, Mutex};

use fynd_core::algorithm::water_fill::initializer::{
    ChunkGains, FineAllocation, FineAllocationInitializer, FineAllocationRequest, Refinement,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    allocation,
    features::{self, ProbeOutputs, PROBES_PER_POOL},
    model::AllocationModel,
    pools::{pool_index, Direction, USDC, WETH},
};

/// What one seed did in one solve.
#[derive(Debug, Clone, Serialize)]
pub struct SeedRecord {
    pub seed: &'static str,
    /// Proposed amounts in pool order, as decimal strings; absent when the seed declined.
    pub proposal: Option<[String; 2]>,
    /// Pool 1's share the seed chose, for diagnostics.
    pub fraction: Option<f64>,
    /// Quotes the seed asks the solve's cache for. Cache hits cost no simulation, so this bounds
    /// the seed's simulations from above; the per-stage log counts the actual ones.
    pub lookups: usize,
    /// The model's feature vector, for learned seeds; lets parity checks replay the prediction.
    pub features: Option<Vec<f64>>,
    /// The allocation-regret estimator's predicted starting loss, in log10 bp, when present.
    pub regret_estimate_log10_bp: Option<f64>,
    /// Whether the seed asked to skip exchange refinement. Only the disjoint split acts on it;
    /// fill-and-spill's request has no refinement to skip.
    pub skipped: bool,
    /// For the certified seed: the most the start can lose to the best split, in bp of its gross
    /// output, from quotes one step either side; absent when the check did not run or found a
    /// better neighbour.
    pub certified_bound_bp: Option<f64>,
    pub declined: Option<&'static str>,
}

/// A seed's proposal in pool order, with the features a learned seed read.
struct Proposal {
    amounts: [BigUint; 2],
    features: Option<Vec<f64>>,
    refinement: Refinement,
    regret_estimate_log10_bp: Option<f64>,
    certified_bound_bp: Option<f64>,
    /// Lookups beyond the seed's fixed ones, such as the certificate's.
    extra_lookups: usize,
}

impl From<[BigUint; 2]> for Proposal {
    fn from(amounts: [BigUint; 2]) -> Self {
        Self {
            amounts,
            features: None,
            refinement: Refinement::Exchange,
            regret_estimate_log10_bp: None,
            certified_bound_bp: None,
            extra_lookups: 0,
        }
    }
}

/// Seeds append here; the runner drains it after each solve. Solves run one at a time.
pub type SeedLog = Arc<Mutex<Vec<SeedRecord>>>;

/// The two active paths in pool order, or why the request is out of scope.
struct TwoPoolRequest {
    /// `paths` index of pool 1, then pool 2.
    path_of_pool: [usize; 2],
    direction: Direction,
}

fn two_pool_request(request: &FineAllocationRequest<'_>) -> Result<TwoPoolRequest, &'static str> {
    if request.paths.len() != 2 {
        return Err("active set is not two paths");
    }
    let mut path_of_pool = [usize::MAX; 2];
    for (path_index, path) in request.paths.iter().enumerate() {
        let [component] = path.components.as_slice() else {
            return Err("active path is not a single hop");
        };
        let Some(pool) = pool_index(component) else {
            return Err("active path is not an experiment pool");
        };
        path_of_pool[pool] = path_index;
    }
    if path_of_pool.contains(&usize::MAX) {
        return Err("active set repeats a pool");
    }
    let token_in = request.paths[0].token_in.to_string().to_lowercase();
    let direction = if token_in == WETH {
        Direction::WethUsdc
    } else if token_in == USDC {
        Direction::UsdcWeth
    } else {
        return Err("order is not WETH/USDC");
    };
    Ok(TwoPoolRequest { path_of_pool, direction })
}

/// Orders a pool-order allocation as the request's paths.
fn in_path_order(two: &TwoPoolRequest, pool_amounts: [BigUint; 2]) -> Vec<BigUint> {
    let mut by_path = vec![BigUint::zero(), BigUint::zero()];
    for (pool, amount) in pool_amounts.into_iter().enumerate() {
        by_path[two.path_of_pool[pool]] = amount;
    }
    by_path
}

fn probe_all(request: &mut FineAllocationRequest<'_>, two: &TwoPoolRequest) -> ProbeOutputs {
    let amounts = features::probe_amounts(request.amount_in);
    let mut outputs: ProbeOutputs = Default::default();
    for (pool, pool_outputs) in outputs.iter_mut().enumerate() {
        for (probe, amount) in amounts.iter().enumerate() {
            if !amount.is_zero() {
                pool_outputs[probe] =
                    request.probe(two.path_of_pool[pool], amount).map(|quote| quote.amount_out);
            }
        }
    }
    outputs
}

fn fraction_of(pool_amounts: &[BigUint; 2], amount: &BigUint) -> f64 {
    let share = pool_amounts[0].to_f64().unwrap_or(f64::NAN);
    share / amount.to_f64().unwrap_or(f64::NAN)
}

/// Shared bookkeeping: eligibility, the log and path ordering around one seed's proposal.
fn seed_with(
    log: &SeedLog,
    name: &'static str,
    request: &mut FineAllocationRequest<'_>,
    propose: impl FnOnce(
        &mut FineAllocationRequest<'_>,
        &TwoPoolRequest,
    ) -> Result<Proposal, &'static str>,
    lookups_per_solve: usize,
) -> Option<FineAllocation> {
    let mut record = SeedRecord {
        seed: name,
        proposal: None,
        fraction: None,
        lookups: 0,
        features: None,
        regret_estimate_log10_bp: None,
        skipped: false,
        certified_bound_bp: None,
        declined: None,
    };
    let outcome = two_pool_request(request).and_then(|two| {
        let proposal = propose(request, &two)?;
        Ok((two, proposal))
    });
    let result = match outcome {
        Ok((two, proposal)) => {
            let pool_amounts = proposal.amounts;
            record.features = proposal.features;
            record.regret_estimate_log10_bp = proposal.regret_estimate_log10_bp;
            record.skipped = proposal.refinement == Refinement::Skip;
            record.certified_bound_bp = proposal.certified_bound_bp;
            record.fraction = Some(fraction_of(&pool_amounts, request.amount_in));
            record.proposal = Some(pool_amounts.clone().map(|amount| amount.to_string()));
            record.lookups = lookups_per_solve + proposal.extra_lookups;
            Some(FineAllocation {
                amounts: in_path_order(&two, pool_amounts),
                refinement: proposal.refinement,
            })
        }
        Err(reason) => {
            record.declined = Some(reason);
            None
        }
    };
    log.lock().expect("seed log poisoned").push(record);
    result
}

/// Exchange refinement starting from the 20-chunk allocation that chose the active set.
pub struct CoarseSeed {
    pub log: SeedLog,
}

impl FineAllocationInitializer for CoarseSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        seed_with(
            &self.log,
            "coarse",
            request,
            |request, two| {
                Ok([
                    request.coarse[two.path_of_pool[0]].clone(),
                    request.coarse[two.path_of_pool[1]].clone(),
                ]
                .into())
            },
            0,
        )
    }
}

/// Grid the interpolation baseline chooses from: pool 1 gets `floor(k * Q / 64)`.
pub const INTERPOLATION_STEPS: u32 = 64;

/// Piecewise-linear interpolation of each pool's output through its probes and the origin, then
/// the best of the 65 grid splits under that estimate. Sees exactly the model's probes.
pub struct InterpolationSeed {
    pub log: SeedLog,
}

impl FineAllocationInitializer for InterpolationSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        seed_with(
            &self.log,
            "interpolation",
            request,
            |request, two| {
                let outputs = probe_all(request, two);
                interpolation_split(request.amount_in, &outputs).map(Proposal::from)
            },
            2 * PROBES_PER_POOL,
        )
    }
}

/// The interpolation baseline's split, shared with tests and the offline mirror.
pub fn interpolation_split(
    amount: &BigUint,
    outputs: &ProbeOutputs,
) -> Result<[BigUint; 2], &'static str> {
    let amounts = features::probe_amounts(amount);
    let knots: Vec<Vec<(f64, f64)>> = (0..2)
        .map(|pool| {
            let mut knots = vec![(0.0, 0.0)];
            for (probe, input) in amounts.iter().enumerate() {
                if let Some(out) = &outputs[pool][probe] {
                    knots.push((input.to_f64().unwrap_or(0.0), out.to_f64().unwrap_or(0.0)));
                }
            }
            knots.sort_by(|a, b| a.0.total_cmp(&b.0));
            knots.dedup_by(|a, b| a.0 == b.0);
            knots
        })
        .collect();
    if knots.iter().any(|pool| pool.len() < 2) {
        return Err("a pool has no successful probe");
    }
    let mut best: Option<(f64, [BigUint; 2])> = None;
    for step in 0..=INTERPOLATION_STEPS {
        let first = amount * step / INTERPOLATION_STEPS;
        let second = amount - &first;
        let estimate = [&first, &second]
            .iter()
            .zip(&knots)
            .map(|(input, pool)| interpolate(pool, input.to_f64().unwrap_or(0.0)))
            .sum::<Option<f64>>();
        let Some(estimate) = estimate else { continue };
        // Strictly greater keeps the smallest step among ties.
        if best.as_ref().is_none_or(|(value, _)| estimate > *value) {
            best = Some((estimate, [first, second]));
        }
    }
    best.map(|(_, split)| split).ok_or("no grid split falls inside both pools' probes")
}

/// Linear interpolation between knots; `None` beyond the last knot, which never extrapolates.
fn interpolate(knots: &[(f64, f64)], input: f64) -> Option<f64> {
    if input == 0.0 {
        return Some(0.0);
    }
    knots.windows(2).find(|pair| input <= pair[1].0).map(|pair| {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        y0 + (y1 - y0) * (input - x0) / (x1 - x0)
    })
}

/// The trained model: probes, the v1 features, one MLP evaluation, an exact integer split.
pub struct LearnedSeed {
    pub log: SeedLog,
    pub model: Arc<AllocationModel>,
}

impl FineAllocationInitializer for LearnedSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        let model = &self.model;
        seed_with(
            &self.log,
            "learned",
            request,
            |request, two| {
                let outputs = probe_all(request, two);
                let features = features::feature_vector(two.direction, request.amount_in, &outputs)
                    .ok_or("no reference price from probes")?;
                let numerator = allocation::numerator(model.share(&features, 0.0));
                Ok(Proposal {
                    features: Some(features.to_vec()),
                    ..allocation::split(request.amount_in, numerator).into()
                })
            },
            2 * PROBES_PER_POOL,
        )
    }
}

/// Pool order's coarse amounts, or why they cannot anchor a proposal.
fn coarse_in_pool_order(
    request: &FineAllocationRequest<'_>,
    two: &TwoPoolRequest,
) -> Result<[BigUint; 2], &'static str> {
    let coarse = two.path_of_pool.map(|path| request.coarse[path].clone());
    // A deadline can stop the coarse pass early; its split then no longer sums to the order.
    if &coarse[0] + &coarse[1] != *request.amount_in {
        return Err("coarse split does not cover the order");
    }
    Ok(coarse)
}

/// The `coarse-offset/2` features of a solve, with the coarse split they are built from. Costs no
/// simulation: the whole-order quotes are in the solve's cache from discovery and the quadratic
/// reads the coarse pass's chunk gains.
fn offset_features(
    request: &mut FineAllocationRequest<'_>,
    two: &TwoPoolRequest,
) -> Result<([BigUint; 2], [f64; features::OFFSET_FEATURE_COUNT]), &'static str> {
    let coarse = coarse_in_pool_order(request, two)?;
    let amount = request.amount_in.clone();
    let full =
        two.path_of_pool.map(|path| request.probe(path, &amount).map(|quote| quote.amount_out));
    // Training never sees a failed whole-order quote, so the model has no input for one; leave
    // the solve to Fynd rather than read it as a working pool.
    if full.iter().any(Option::is_none) {
        return Err("a pool could not quote the whole order");
    }
    let quadratic = free_quadratic_offset(request, two, &coarse)?;
    let features =
        features::offset_feature_vector(two.direction, &amount, &coarse[0], &full, quadratic)
            .ok_or("neither pool quoted the whole order")?;
    Ok((coarse, features))
}

/// The learned coarse correction: the coarse split moved by a trained offset of at most a few
/// coarse chunks, from features that cost no new simulation. With a confidence model, it asks to
/// skip exchange refinement when the model estimates the start is already close enough.
pub struct OffsetSeed {
    pub log: SeedLog,
    /// Validated by the runner to read [`FeatureSpec::CoarseOffset`](crate::model::FeatureSpec).
    pub model: Arc<AllocationModel>,
    /// Validated by the runner to be a regret-estimate model on the same features.
    pub confidence: Option<Arc<AllocationModel>>,
    /// With a step divisor `d`, skip only when quotes `amount_in / d` either side of the start
    /// also prove it loses at most `CERTIFIED_MAX_BP`.
    pub certificate_step_divisor: Option<u32>,
}

pub const CERTIFIED_MAX_BP: f64 = 0.02;

/// The most the proposed split can lose to the best split, in bp of its gross output, or `None`
/// when a neighbour one step either side quotes more (the best split may lie beyond it) or a quote
/// fails.
///
/// Gross output is concave in pool 1's amount (each pool's marginal rate only falls), so when
/// neither neighbour beats the start the best split lies within one step of it, and the secant
/// from each neighbour bounds how much higher it can be: at most the larger of the two drops.
/// Gas is left out: within one step it changes only by tick crossings.
fn certified_bound_bp(
    request: &mut FineAllocationRequest<'_>,
    two: &TwoPoolRequest,
    amounts: &[BigUint; 2],
    step_divisor: u32,
) -> Option<f64> {
    let step = request.amount_in / step_divisor;
    if step.is_zero() || amounts[0] < step || amounts[1] < step {
        return None;
    }
    let mut gross = |first: &BigUint, second: &BigUint| -> Option<BigUint> {
        let mut total = BigUint::zero();
        for (pool, amount) in [first, second].into_iter().enumerate() {
            if !amount.is_zero() {
                total += request.probe(two.path_of_pool[pool], amount)?.amount_out;
            }
        }
        Some(total)
    };
    let start = gross(&amounts[0], &amounts[1])?;
    let left = gross(&(&amounts[0] - &step), &(&amounts[1] + &step))?;
    let right = gross(&(&amounts[0] + &step), &(&amounts[1] - &step))?;
    loss_bound_bp(&start, &left, &right)
}

/// The concavity bound from the start's gross output and its two neighbours'.
fn loss_bound_bp(start: &BigUint, left: &BigUint, right: &BigUint) -> Option<f64> {
    if left > start || right > start {
        return None;
    }
    let drop = (start - left).max(start - right);
    Some(drop.to_f64()? / start.to_f64()? * 1e4)
}

impl FineAllocationInitializer for OffsetSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        let model = &self.model;
        seed_with(
            &self.log,
            "offset",
            request,
            |request, two| {
                let (_, features) = offset_features(request, two)?;
                let share = model.share(&features, features[2]);
                let mut proposal = Proposal {
                    features: Some(features.to_vec()),
                    ..allocation::split(request.amount_in, allocation::numerator(share)).into()
                };
                // Fynd asks again for fill-and-spill's pass, which ignores the refinement.
                if let Some(confidence) = &self.confidence {
                    let estimate = confidence
                        .regret_estimate(&features)
                        .expect("the runner loads only regret-estimate confidence models");
                    proposal.regret_estimate_log10_bp = Some(estimate.log10_bp);
                    let skip = if let (true, Some(divisor)) =
                        (estimate.skip, self.certificate_step_divisor)
                    {
                        let amounts = proposal.amounts.clone();
                        proposal.extra_lookups = 6;
                        proposal.certified_bound_bp =
                            certified_bound_bp(request, two, &amounts, divisor);
                        proposal.certified_bound_bp.is_some_and(|bound| bound <= CERTIFIED_MAX_BP)
                    } else {
                        estimate.skip
                    };
                    if skip {
                        proposal.refinement = Refinement::Skip;
                    }
                }
                Ok(proposal)
            },
            2,
        )
    }
}

/// Records the offset seed's features exactly as Fynd builds them, for training data, and starts
/// refinement from the coarse split. Declines wherever the offset seed would, so the cases it
/// records are the ones the model will be asked about.
pub struct OffsetFeatureRecorder {
    pub log: SeedLog,
}

impl FineAllocationInitializer for OffsetFeatureRecorder {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        seed_with(
            &self.log,
            "offset_features",
            request,
            |request, two| {
                let (coarse, features) = offset_features(request, two)?;
                Ok(Proposal { features: Some(features.to_vec()), ..coarse.into() })
            },
            2,
        )
    }
}

/// A pool's output near its coarse amount, as the quadratic through three points one coarse chunk
/// apart. `t` counts chunks from the coarse amount.
struct LocalQuadratic {
    /// Offset of the first point: -1 when the coarse pass priced the pool's next chunk, -2 when it
    /// did not (the pool won the last chunk), so the points are the last chunks it won.
    first: f64,
    /// `out(first + 1) - out(first)`.
    step: f64,
    /// `out(first + 2) - 2 out(first + 1) + out(first)`; negative for a concave curve.
    curvature: f64,
}

impl LocalQuadratic {
    /// From what the coarse pass's chunks added to the pool: the last chunk it won and its standing
    /// bid for the next one, or its last two chunks when it has no standing bid. A pool that won one
    /// chunk and has no bid is taken as linear.
    fn from_gains(gains: &ChunkGains) -> Result<Self, &'static str> {
        let as_f64 = |value: &BigInt| value.to_f64().ok_or("chunk gain out of range");
        let paid: Vec<BigInt> = gains.paid.iter().cloned().map(BigInt::from).collect();
        let (first, earlier, later) = match (paid.as_slice(), &gains.next) {
            ([.., last], Some(next)) => (-1.0, last.clone(), BigInt::from(next.clone())),
            ([.., before, last], None) => (-2.0, before.clone(), last.clone()),
            ([only], None) => (-1.0, only.clone(), only.clone()),
            ([], _) => return Err("an active pool won no coarse chunk"),
        };
        // Differences in exact integers first: gains reach 1e20 while their difference, which sets
        // the optimum, is a small fraction of them.
        Ok(Self { first, step: as_f64(&earlier)?, curvature: as_f64(&(later - &earlier))? })
    }

    /// Output at `t` minus output at the first knot.
    fn gain(&self, t: f64) -> f64 {
        let u = t - self.first;
        u * self.step + u * (u - 1.0) * self.curvature / 2.0
    }
}

/// The move `s`, in chunks within `[low, high]`, that maximizes `first.gain(s) + second.gain(-s)`:
/// pool 1 moves by `s` chunks and pool 2 by `-s`.
fn best_offset(first: &LocalQuadratic, second: &LocalQuadratic, low: f64, high: f64) -> f64 {
    let total = |s: f64| first.gain(s) + second.gain(-s);
    let mut candidates = vec![low, 0.0, high];
    // d/ds total = 0, solved in closed form when the sum is concave.
    let concavity = first.curvature + second.curvature;
    if concavity < 0.0 {
        let slope_at_zero =
            first.step - second.step - first.curvature * (2.0 * first.first + 1.0) / 2.0
                + second.curvature * (2.0 * second.first + 1.0) / 2.0;
        candidates.push((-slope_at_zero / concavity).clamp(low, high));
    }
    candidates
        .into_iter()
        .fold((0.0, total(0.0)), |best, s| {
            let value = total(s);
            if value > best.1 {
                (s, value)
            } else {
                best
            }
        })
        .0
}

/// Pool 1's move from the coarse split, in chunks within one chunk, that maximizes the two pools'
/// quadratics through the coarse pass's chunk gains. Costs no simulation.
fn free_quadratic_offset(
    request: &FineAllocationRequest<'_>,
    two: &TwoPoolRequest,
    coarse: &[BigUint; 2],
) -> Result<f64, &'static str> {
    let gains = two.path_of_pool.map(|path| request.coarse_gains.get(path));
    let [Some(first), Some(second)] = gains else {
        return Err("no chunk gains for an active path");
    };
    let first = LocalQuadratic::from_gains(first)?;
    let second = LocalQuadratic::from_gains(second)?;
    let amount_f = request.amount_in.to_f64().ok_or("order too large")?;
    let chunk_f = amount_f / 20.0;
    let coarse_f = coarse[0].to_f64().ok_or("order too large")?;
    // Pool 1 moves by `s` chunks and pool 2 by `-s`, staying within one chunk and inside the order.
    let low = (-1.0f64).max(-coarse_f / chunk_f);
    let high = 1.0f64.min((amount_f - coarse_f) / chunk_f);
    Ok(best_offset(&first, &second, low, high))
}

/// A non-learned local refinement of the coarse split: move pool 1 to where the quadratics through
/// the coarse pass's chunk gains put the best split, within one chunk. No new simulation.
pub struct QuadraticSeed {
    pub log: SeedLog,
}

impl FineAllocationInitializer for QuadraticSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        seed_with(
            &self.log,
            "quadratic",
            request,
            |request, two| {
                let coarse = coarse_in_pool_order(request, two)?;
                let offset = free_quadratic_offset(request, two, &coarse)?;
                let amount = request.amount_in.clone();
                let amount_f = amount.to_f64().ok_or("order too large")?;
                let coarse_f = coarse[0].to_f64().ok_or("order too large")?;
                let share = ((coarse_f + offset * amount_f / 20.0) / amount_f).clamp(0.0, 1.0);
                Ok(allocation::split(&amount, allocation::numerator(share)).into())
            },
            0,
        )
    }
}

/// Grid for the reference seed: pool 1 gets `floor(k * Q / 256)`.
pub const REFERENCE_STEPS: u32 = 256;

/// A deliberately expensive seed: the best gross split on a 257-point grid of exact quotes.
/// It measures how much refinement a near-ideal start still needs; its time is not a candidate's.
pub struct ReferenceSeed {
    pub log: SeedLog,
}

impl FineAllocationInitializer for ReferenceSeed {
    fn initial_allocation(
        &self,
        request: &mut FineAllocationRequest<'_>,
    ) -> Option<FineAllocation> {
        seed_with(
            &self.log,
            "reference",
            request,
            |request, two| {
                let amount = request.amount_in.clone();
                let mut best: Option<(BigUint, [BigUint; 2])> = None;
                for step in 0..=REFERENCE_STEPS {
                    let first = &amount * step / REFERENCE_STEPS;
                    let second = &amount - &first;
                    let mut total = BigUint::zero();
                    let mut quoted = true;
                    for (pool, input) in [&first, &second].into_iter().enumerate() {
                        if input.is_zero() {
                            continue;
                        }
                        match request.probe(two.path_of_pool[pool], input) {
                            Some(quote) => total += quote.amount_out,
                            None => quoted = false,
                        }
                    }
                    if quoted && best.as_ref().is_none_or(|(value, _)| &total > value) {
                        best = Some((total, [first, second]));
                    }
                }
                best.map(|(_, split)| split.into()).ok_or("no grid split quotes")
            },
            2 * REFERENCE_STEPS as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `out(t) = slope * t - t^2`, t in chunks from the coarse amount, knots from `first`.
    fn parabola(slope: f64, first: f64) -> LocalQuadratic {
        let out = |t: f64| slope * t - t * t;
        LocalQuadratic {
            first,
            step: out(first + 1.0) - out(first),
            curvature: out(first + 2.0) - 2.0 * out(first + 1.0) + out(first),
        }
    }

    #[test]
    fn quadratic_offset_equalizes_marginal_outputs() {
        // total(s) = (s - s^2) + (-0.2 s - s^2) peaks at s = 0.2, whichever knots each fit used.
        for (first, second) in [(-1.0, -1.0), (0.0, -1.0), (-2.0, 0.0)] {
            let offset = best_offset(&parabola(1.0, first), &parabola(0.2, second), -1.0, 1.0);
            assert!((offset - 0.2).abs() < 1e-12, "knots {first}, {second}: {offset}");
        }
    }

    #[test]
    fn quadratic_offset_stays_within_its_range() {
        // Unconstrained optimum at s = 2.45, beyond one chunk.
        assert_eq!(best_offset(&parabola(10.0, -1.0), &parabola(0.2, -1.0), -1.0, 1.0), 1.0);
        // Pool 1 already at zero cannot move down.
        assert_eq!(best_offset(&parabola(-10.0, 0.0), &parabola(0.2, -1.0), 0.0, 1.0), 0.0);
        // A convex sum has no interior optimum: take the better endpoint.
        let convex = |slope| LocalQuadratic { first: -1.0, step: slope, curvature: 2.0 };
        assert_eq!(best_offset(&convex(1.0), &convex(0.0), -1.0, 1.0), 1.0);
    }

    #[test]
    fn loss_bound_is_the_larger_drop_and_refuses_a_better_neighbour() {
        let bound = |start: u64, left: u64, right: u64| {
            loss_bound_bp(&BigUint::from(start), &BigUint::from(left), &BigUint::from(right))
        };
        // Output 1,000,000 at the start, 20 and 50 less either side: 50 / 1e6 = 0.5 bp.
        assert_eq!(bound(1_000_000, 999_980, 999_950), Some(0.5));
        assert_eq!(bound(1_000_000, 1_000_000, 1_000_000), Some(0.0));
        assert_eq!(bound(1_000_000, 1_000_001, 999_950), None, "the best split may lie left");
        assert_eq!(bound(1_000_000, 999_950, 1_000_001), None, "the best split may lie right");
    }

    fn outputs(pool1: [u64; 4], pool2: [u64; 4]) -> ProbeOutputs {
        [pool1.map(|v| Some(BigUint::from(v))), pool2.map(|v| Some(BigUint::from(v)))]
    }

    #[test]
    fn interpolation_splits_equal_linear_pools_at_first_best_step() {
        // Identical, perfectly linear pools: every split ties, so the smallest step wins.
        let amount = BigUint::from(6_400u32);
        let split = interpolation_split(
            &amount,
            &outputs([800, 3200, 6400, 12800], [800, 3200, 6400, 12800]),
        )
        .unwrap();
        assert_eq!(split, [BigUint::zero(), amount]);
    }

    #[test]
    fn interpolation_moves_flow_to_the_pool_that_saturates_later() {
        // Pool 2 stops paying beyond Q/4; pool 1 stays linear. The best split gives pool 2 its
        // flat-top threshold and pool 1 the rest.
        let amount = BigUint::from(6_400u32);
        let split = interpolation_split(
            &amount,
            &outputs([400, 1600, 3200, 6400], [500, 2000, 2000, 2000]),
        )
        .unwrap();
        assert_eq!(split, [BigUint::from(4_800u32), BigUint::from(1_600u32)]);
        assert_eq!(&split[0] + &split[1], amount);
    }

    #[test]
    fn interpolation_never_extrapolates_past_failed_probes() {
        let amount = BigUint::from(6_400u32);
        let mut probes = outputs([400, 1600, 3200, 6400], [500, 2000, 4000, 8000]);
        probes[1][2] = None;
        probes[1][3] = None;
        let split = interpolation_split(&amount, &probes).unwrap();
        assert!(split[1] <= BigUint::from(1_600u32), "pool 2 is known only up to Q/4");
    }
}
