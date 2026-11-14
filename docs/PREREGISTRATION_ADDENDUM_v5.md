# Pre-registration addendum v5: why side and not price? An asymmetry test (queue priority vs GLFT)

**Status:** frozen on November 14, 2026 (git tag `prereg-v5`), before the test below was run on any coin.

## 0. What is already known, and why this test is new
Part 1 (prereg-v3) and its replication on 16 coins: makers respond to inventory mainly by quote side; the
average price response is a fraction of a tick. Every price model run so far pooled the two sides with one slope
on q. **No model so far let the slope differ between the side a maker wants filled and the side it does not.**
That split is the test here. Nothing in it has been estimated before.

## 1. Two explanations and what tells them apart
Define for each quote
- s = +1 for an ask, -1 for a bid, and **u = q x s** (u > 0: the quote is on the side that **reduces** the
  maker's inventory, the "unwinding" side; u < 0: the side that would **add** to it, the "accumulating" side);
- **aggressiveness** a, in ticks of the coin's median tick: (price - mid) for a bid, (mid - price) for an ask,
  divided by the tick (higher = closer to or through the mid = more likely to fill).

**GLFT:** both quotes shift by the same amount against inventory, so a = c + k u with ONE slope k > 0 for all u:
the unwinding quote gets more aggressive by exactly as much as the accumulating quote gets less aggressive.

**Queue priority:** orders at a price fill first come, first served, so a place in the queue is worth money.
Moving the accumulating quote one tick away loses its place and still leaves a fill risk; not posting it at all is
cheaper. So the accumulating side is pulled (side response, already measured) rather than shaded: **its slope is
near zero**, while the unwinding side is made more aggressive to sell (or buy) faster.

## 2. Model
a = k_plus x max(u, 0) + k_minus x min(u, 0) + d x vol5 + FE (maker x side x day)
Same sample, weights (every maker equal within the sample), and clustering (coin-day) as prereg-v3 T1 (the
script prints its maker, quote and coin-day counts next to day22's "all" row, which reproduced day21).
- k_plus: change in aggressiveness per unit of u on the unwinding side.
- k_minus: the same on the accumulating side (aggressiveness = k_minus x u with u < 0, so k_minus > 0 means the
  quote is moved away from the mid).
- Both are **signed slopes with respect to u**; H5 compares the two signed slopes (GLFT: one straight line through
  both sides, so they are equal).

## 3. Test (confirmatory sample: SOL, XRP, BTC, ETH, HYPE, all 86 days)
- **H5 (asymmetry):** k_plus - k_minus > 0, p < 0.05 (two-sided, coin-day clusters).
- GLFT predicts k_plus = k_minus. If H5 holds, the paper reports: "the evidence fits queue priority, not GLFT's
  symmetric skew". It is not proof of the makers' reason.

Secondary (reported, not decisive):
- the same model with **at_best** (1 if at or better than the best on its side) as the outcome, reported in
  percentage points of the probability of being at the best (not in ticks);
- each coin separately; the 16 replication coins (descriptive);
- **BTC and HYPE only** (the 5 bp sample band equals 40 ticks there, so dropping quotes far from the best cannot
  create a flat accumulating-side slope): the main guard against a sample-truncation artefact.

## 4. Unchanged
All prereg-v1 to v3 definitions (quotes, q capped at +-5, vol5, coins' median tick, privacy). Full wallet
addresses never leave the PC.
