# Pre-registration addendum v3: Part 1 - do real makers manage inventory the way GLFT says?

**Status:** frozen on November 14, 2026 (git tag `prereg-v3`), before any of the tests below was run.
What is already known and is NOT a result of this addendum: in the boundary windows of `prereg-v1`, side
responded to inventory (about -0.055 bid probability per 1 sd of q) and price shading was small
(about -0.085 bp per sd). Those numbers motivated the hypotheses; the tests below use the full 86 days,
five coins and all makers, with fixed rules written here first.

## 1. Theory being tested

In the Gueant-Lehalle-Fernandez-Tapia (GLFT) market-making model, a maker always quotes both sides and
manages inventory by **shifting both quote prices** against its position (long -> both prices lower). Pulling
a side happens only at the inventory limit. The size of the price shift grows with volatility.

## 2. Data and sample

- Coins: SOL, XRP, BTC, ETH, HYPE; all 86 days (Jun 1-14 and Aug 15-Oct 25, 2026); quote tables built by
  the frozen Day 6 code.
- Quotes: the main-sample rules of `prereg-v1` section 2 (good book hour, outside gaps, not crossing, within
  5 bp of the best, inventory known), every maker account with at least 1,000 such quotes.
- Sampling for speed: quotes whose order id is divisible by 5 (a fixed 20% sample, chosen before any result).
- q = inventory / std of the minute-by-minute position over the previous 1,440 minutes, capped at +-5
  (as `prereg-v1`); vol5 as in the frozen balance code. Quotes without q or vol5 drop out.

## 3. Tests

All regressions: weighted least squares, every maker weighted equally within its coin, fixed effects
maker x day (side) or maker x side x day (price, size); standard errors clustered by coin-day.
Per-maker estimates use the same models on one maker's quotes (fixed effects side x day or day).

**T1 - Which tool? (headline)**
- side response b_side: bid = b_side * q + d * vol5 + FE.
- price response b_price: (price - mid) / mid in bp = b_price * q + d * vol5 + FE; also in ticks, dividing by
  the coin's median tick in bp.
- size response b_size: log size = b_size * (q * s) + d * vol5 + FE, s = +1 ask, -1 bid.
- **Prediction from GLFT:** price is the main tool, so per 1 sd of q the price shift is at least one tick.
- **"Side, not price" is supported** if, pooled over the five coins: (a) b_side < 0 with p < 0.05, and
  (b) |b_price| is below one tick per sd of q (upper end of its 95% interval below 1 tick), and
  (c) among makers with a significant response in either tool (p < 0.05), a majority have a significant side
  response but a price response below one tick per sd.

**T2 - Does skew grow with volatility?** Add q x vol5 (vol5 standardized within coin) to the price and side
models. GLFT predicts a more negative price response when volatility is higher (coefficient < 0).
Reported for both tools; the price coefficient is the test (p < 0.05, two-sided).

**T3 - Where along inventory does side respond?** Side model with q as a piecewise-linear spline, knots at
|q| = 1 and |q| = 3 (slopes for |q| < 1, 1-3, 3-5). GLFT with an inventory limit predicts side responds only
near the limit (slope for |q| < 1 about zero). Test: slope for |q| < 1 is different from 0 (p < 0.05).

**T4 - Does side skew unwind inventory?** Across maker accounts with a per-maker b_side and an inventory
half-life (from the frozen `natives.csv`), Spearman correlation between b_side and half-life. Prediction:
positive (stronger negative side response -> shorter half-life). Permutation p, 10,000 shuffles.

## 4. Robustness (reported, not decisive)
Each coin separately; hedgers vs natives (`natives.csv` labels); q capped at +-3 and uncapped; June vs
Aug-Oct; firm level (sleeves summed, `sleeves.csv`) instead of account level.

## 5. Reading the results
- T1 supported: "makers manage inventory mainly by choosing which side to quote, not by skewing prices",
  the paper's main descriptive finding, and a direct departure from GLFT.
- T1 not supported: reported as is; makers then use price as GLFT predicts.
- T2-T4 qualify the headline (does any of GLFT's other predictions hold?) and are reported whatever they show.
