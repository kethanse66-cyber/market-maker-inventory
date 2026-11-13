# Pre-registration addendum v2: round-number test

**Status:** frozen on November 13, 2026 (git tag `prereg-v2`), after the results of `prereg-v1` were seen
(main test, three secondary outcomes, eight placebo levels, PENGU $0.01) and **before** any outcome was
computed for STRK, FARTCOIN or the 15 replication coin-boundaries. Everything in `prereg-v1` stays in force;
this addendum only adds tests.

## 1. Why this addendum exists (honest origin)

Under `prereg-v1`, price shading per unit of inventory fell above the boundary at SOL $100 (tau +0.031,
p = 0.012) and XRP $1.00 (+0.036, p = 0.006). The deciding clean placebo, PENGU $0.01, where the tick does
not change, showed the same effect (+0.033, p = 0.014). So the tick claim for price fails, as `prereg-v1`
section 7 requires. All three levels are round powers of ten; the four SOL placebo levels ($95-$120) showed
nothing, XRP $1.40 showed a smaller effect (+0.020, p = 0.012) and XRP $1.10, $1.30, $1.50 did not.
This pattern was found by looking. It is a hypothesis, not a result, until it holds on data not yet examined.

## 2. Hypothesis

**H-R:** at a price level that is a power of ten, makers shade price **less** per unit of inventory above the
level than below it, **whether or not the tick changes there** (tau_price > 0).

## 3. Confirmatory test (decides H-R)

- **Data:** STRK $0.10 and FARTCOIN $0.10 (delivery 3). The tick does not change at either level. No quote,
  inventory or outcome of these coins has been computed.
- **Code:** the frozen pipeline (`day16a`-`day16d`) and `day15_placebos.py`, unchanged: same sample rules,
  model, maker weights, minute labels, circular shifts.
- **Estimate:** tau_price pooled over the two coins (one model, maker x side x episode fixed effects).
- **Inference:** circular-shift randomization, **2,000 draws**, seed 20261114, two-sided, 5% level.
- **Exact run:** `day15_placebos.py` with `P3_EXTRA="STRK+FARTCOIN:0.1"`, `P3_ONLY=STRK`, `P3_TEST_DRAWS=2000`,
  `P3_SEED=20261114`, `P3_ALL_DAYS=1`, `P3_EPISODES=episodes_d3.csv` (one pooled model over both coins).
- **Decision:** H-R is supported if tau_price > 0 with p < 0.05.
  If p >= 0.05 it is reported as not confirmed, with its interval.
- **Known limit, stated now:** these coins have few episodes (STRK 5 coin-days, FARTCOIN 3), so power is low.
  A non-significant result here is weak evidence against H-R, and will be described that way.

## 4. Supporting tests (reported, not decisive)

1. **Replication boundaries:** tau_price at each of the 15 replication coin-boundaries and pooled. They are
   powers of ten **with** a tick change, so they test whether the price effect generalizes, but they cannot
   separate round numbers from the tick.
2. **Non-round controls:** for every replication coin with enough crossings, the levels 0.95x and 1.05x its
   boundary (not powers of ten, same tick on both sides). H-R predicts tau_price near 0 there.
3. **PENGU at non-round levels** $0.009 and $0.011 (if crossed in its delivered days): H-R predicts ~0.

## 5. What would change the story

- Confirmed at STRK/FARTCOIN and absent at non-round controls: **round-number effect on maker pricing**,
  independent of tick size. Reported as the paper's main pricing finding.
- Not confirmed: the price pattern is reported as exploratory, possibly specific to the three coins seen.
- Either way, the `prereg-v1` results stand as reported: side null, tick claim for price not supported,
  at-best-price jump tied to the tick.
