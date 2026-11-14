# Pre-registration addendum v4: does TICK SIZE decide how makers manage inventory?

**Status: WITHDRAWN on November 14, 2026, before any test below was run** (the tag `prereg-v4` marks the
original text, unchanged below). Reasons, found while writing the code:
1. Test B is not new. It is the confirmatory test of `prereg-v1` (side response x large tick at the boundary),
   already run on November 13: tau = +0.0037, p = 0.416, a null that replicated on 9 delivery-3 coins. The price
   version is the `prereg-v1` secondary outcome, whose boundary effect failed the PENGU placebo. Re-registering a
   test with a known result would not be confirmatory.
2. The motivating pattern in section 0 is a unit artifact. Converted from ticks to basis points, price shading
   per sd of q is about the same in all five coins (SOL -0.05, XRP -0.04, BTC -0.05, ETH -0.05, HYPE -0.06 bp);
   it only looked larger in ticks where the tick is small.
No analysis in this addendum was run. The delivery-3 Part 1 tables built afterwards are used only as a descriptive
out-of-sample replication of the prereg-v3 T1 result, labelled as such.

---

## 0. What is already known (NOT results of this addendum)
Part 1 (prereg-v3) on five coins: makers respond to inventory mainly by quote side, and price shading is under one
tick. Across those five coins, price shading measured in ticks was largest where the tick is smallest relative to
the price (BTC -0.38, HYPE -0.52 ticks per sd of q) and smallest where the tick is large (SOL -0.06, XRP -0.05).
**This pattern motivated the hypothesis. The five coins are therefore NOT part of the confirmatory sample of
Test A**; they are reported alongside it.

## 1. Hypothesis
A maker that wants to skew its quotes by less than one tick cannot do it with price. When the tick is large
relative to how far prices move, makers manage inventory by choosing the side; when the tick is small, they also
use price. Measure: the **relative tick** R = tick (bp) / sigma_1min, where sigma_1min (bp) = vol5 x sqrt(60) (vol5 =
std of 1-second mid returns over the last 300 s, frozen definition). R is the tick in units of a typical
one-minute price move.

Predictions:
- **H1 (side):** the side response to q is more negative when R is larger.
- **H2 (price):** the price response to q, measured in **volatility units** (price offset in bp / sigma_1min), is
  closer to zero when R is larger. Volatility units are used so that a bigger tick cannot shrink the price
  response mechanically (as it would when the response is measured in ticks).

## 2. Data
- **Test A confirmatory coins:** the 16 delivery-3 coins built with the frozen pipeline (day16a-d, day18):
  AAVE, CASHCAT, DOGE, LINK, LIT, ORDI, RESOLV, SKY, SUI, UNI, VVV, XPL, ZEC, PENGU, STRK, FARTCOIN; all days with
  built quote tables. CHIP is excluded (no 0.5 s book). Coins whose Part 1 tables cannot be built are listed
  with the reason; nothing else is excluded.
- Part 1 tables for these coins are built with **unchanged** `day20_part1_build.py` rules (main-sample quotes,
  order id divisible by 5, q capped at +-5, vol5, makers with at least 1,000 quotes per coin).
- The five Part 1 coins are reported as a separate, non-confirmatory row.

## 3. Test A - relative tick across coin-days (confirmatory: 16 new coins)
Stage 1, per coin-day: the T1 models of prereg-v3 on that coin-day's quotes (maker x day FE for side; maker x side
x day FE for price), every maker weighted equally, estimating
- b_side (change in P(bid) per 1 unit of q), and
- b_price_sigma (price offset in bp / sigma_1min of that coin-day, per 1 unit of q),
with standard errors clustered by maker. A coin-day enters if at least 5 makers have at least 200 quotes that day.
R_cd = coin-day median tick_bp / coin-day median sigma_1min.

Stage 2: weighted least squares of each stage-1 estimate on log(R_cd), with **coin fixed effects** (only the
variation of R within a coin over days is used: price-level changes move tick_bp, volatility changes move sigma),
weights 1 / se^2 (se floored at its 5th percentile so a few days cannot dominate), standard errors clustered by
coin, p-values from a wild cluster bootstrap (Webb weights, 9,999 draws, seed 20261115).

- **H1 supported:** coefficient on log R in the b_side model < 0, p < 0.05.
- **H2 supported:** coefficient on log R in the b_price_sigma model > 0 (closer to zero, since b_price < 0),
  p < 0.05.
- Reported, not decisive: the same without coin fixed effects (across coins), and the five Part 1 coins.

## 4. Test B - the tick jumps at round prices (natural experiment)
Hyperliquid prices carry at most 5 significant figures, so the tick becomes 10 times larger when the price crosses
a power of ten (and changes at the other frozen boundaries of prereg-v1 / v2). The makers and the market are the
same seconds before and after; only the tick changes.

- Episodes: every boundary episode already defined by the frozen rules (prereg-v1 SOL/XRP main episodes; the
  delivery-3 boundary episodes of day19 for coins with Part 1 tables), +-30 minutes, the straddle rules unchanged.
  Quotes come from the Part 1 tables (so q is defined as everywhere else).
- Model: outcome = b q + c (q x big) + d big + e vol5 + FE, where big = 1 while the market sits on the large-tick
  side of the boundary (book-minute median labels of prereg-v1). FE maker x episode (side), maker x side x episode
  (price). Every maker weighted equally within an episode; episodes weighted equally.
- Inference: circular-shift randomization of the minute labels within each episode (as prereg-v1), 2,000 draws,
  seed 20261115; the p-value of c is the share of draws with |c| at least as large.
- **Primary:** c in the price model (price offset in bp per unit q) **> 0**: a bigger tick shrinks price shading.
  **Secondary:** c in the side model **< 0**: a bigger tick strengthens the side response.
- Power note: within 1-hour windows inventory varies less than over days; a null here is reported as
  "not detected", not as evidence against H1/H2.

## 5. Reading the results
- **"Tick size decides the tool" is supported** if H1 and H2 are both supported in Test A AND Test B's primary
  coefficient has the predicted sign (> 0). If Test B's primary is also p < 0.05, the claim is called causal for
  the boundary moments.
- Only one of H1/H2: reported as partial support, named exactly.
- Neither: reported as is; Part 1's headline stands without an explanation.

## 6. Unchanged
All prereg-v1/v2/v3 definitions (quotes, q, vol5, gaps, outages, labels, privacy). Full wallet addresses never
leave the PC; outputs print 8 characters.
