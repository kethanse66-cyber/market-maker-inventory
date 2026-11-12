# Pre-registration: Tick Size and Inventory Control by Market Makers on Hyperliquid

**Author:** Kethan S E
**Status:** draft for pre-registration; no outcome near the SOL $100 or XRP $1.00 boundary has been examined
**Planned freeze:** November 12, 2026. The frozen version will be tagged `prereg-v1`, with file hashes
recorded in `docs/FREEZE_HASHES.txt`. This line changes to "frozen" only once both exist.

Nothing in this document was chosen by looking at quote sides or quote prices around the boundaries.
The June data (no boundary crossings) served only as a noise pilot. Every rule below was fixed from
data-quality checks, the June pilot, and pre-quote covariates.

---

## 1. Question and hypothesis

Hyperliquid prices have at most 5 significant figures and at most 6 − szDecimals decimals. So the tick
grows 10x when a price crosses a power of ten:

| Coin | szDecimals | Tick below | Tick at or above |
|---|---|---|---|
| SOL at $100 | 2 | 0.001 | 0.01 |
| XRP at $1.00 | 0 | 0.00001 | 0.0001 |

Nothing else about the market changes at the boundary. This gives a within-coin natural experiment:
the same coin, the same makers, nearby hours, and a 10x different tick.

**Hypothesis.** With the coarser tick, moving a quote by one step is a much larger price change, so
makers rely less on price and more on *which side they quote* to manage inventory.

**Primary prediction.** The response of quote side to inventory is stronger above the boundary
(τ < 0 in the model of section 4). The test is two-sided.

---

## 2. Data

- **Source:** 0xArchive Hyperliquid order history, fills and resting-order snapshots (export deliveries 1
  and 2), whole-market trades and top of book (API), and Binance 1-second prices.
- **Test window:** Aug 15 – Oct 25, 2026. **Noise pilot:** Jun 1 – 14, 2026 (no crossings).
- **Validation:** the order-book rebuild matches the Oct 25 exchange snapshot exactly (SOL 244/244,
  XRP 445/445 orders). The inventory chain matches on 99.96% (SOL) and 99.99% (XRP) of fills. See
  `docs/DATA_VALIDATION.md`.

### 2.1 What a quote is

A quote is the placement of a post-only (`Alo`) limit order that is neither reduce-only nor a trigger
order. TWAP slices and liquidation orders are excluded.

### 2.2 Main sample

A quote enters the main sample only if all of the following hold:

1. it lies in a **good-book hour**: book clock delay measured hour by hour, and at most 2% of quotes in
   the hour still cross the book (decision 6);
2. it is **outside the data gaps** of the test window: Aug 29 20:21–20:32 and Oct 25 09:32–10:29 UTC; orders
   open across the Aug 29 gap are dropped (the pilot period also excluded Jun 5 22:00–22:24 and the
   book gaps of Jun 7–9 and Jun 14);
3. it **does not cross the book** at the measured delay;
4. it is **within 5 bp of the maker's own best price**: ticks_behind_best × tick / mid ≤ 5 bp, the same band
   in both tick regimes (decision 20);
5. it is **not in the straddle zone**, meaning the best bid is not below the boundary while the best ask
   is at or above it (decision 19).

### 2.3 Episodes and window

- **Episode:** a minute-mid crossing of the boundary, with a ±30-minute window. Windows that overlap are
  merged (60-minute merge).
- **Primary episodes:** the **31 good-book episodes on 14 days**: SOL $100 has 15 on 9 days, and XRP $1.00
  has 16 on 5 days (Aug 15–19).
- **Window around the boundary:** |mid − boundary| / boundary ≤ **0.5%** (decision 21). This is the widest
  window that passed the balance rule fixed in `day11_balance.py`: every covariate (volatility, inventory,
  time of day as sine and cosine, Binance basis) has |standardized difference| ≤ 0.10, and at least
  **20** episodes contribute valid values on both sides. In the data, every covariate had 29 or more.

### 2.4 Inventory

- q = the account's position just before the quote / the standard deviation of its minute-by-minute
  position over the previous 24 hours. Flat minutes count, and all 1,440 prior minutes are required.
- q is **capped at ±5** (decision 17). Without the cap, a maker that was nearly flat the previous day
  produced |q| up to 12,960, and those values swamped every fit in the pilot.

---

## 3. Units of analysis

- **Main unit:** the quoting account. Its own Hyperliquid position is its inventory (decision 8).
- **Also reported:** results at the **sleeve** level (accounts of one firm that trade as one unit,
  inferred in `sleeves.csv`) and the **firm** level (on-chain firm label).
- **Native vs hedger:** SOL and XRP accounts show no detectable Binance hedging (decision 23). An account's
  calibrated p is the share of wrong-day placebo tests (its coin, same speed) at least as extreme as its
  real Granger test. The share of accounts with calibrated p ≤ 5% was 6% at 1 s (SOL, of 137 accounts)
  and 5% (XRP, of 118 accounts), which is what chance gives. So the tick test uses all makers.
  Natives-only results are a robustness check.

---

## 4. Primary estimand

**Outcome:** y = 1 if the quote is a bid, 0 if it is an ask.

**Model:**

    y = b·q + c·above + τ·(q × above) + d·vol5 + e·(vol5 × above) + α(maker × episode) + ε

- **above** = 1 if the Hyperliquid mid is at or above the boundary (side cannot depend on the quote's
  own price).
- **vol5** = realized volatility of the Hyperliquid mid over the previous 5 minutes, a pre-treatment control.
- **α(maker × episode)** = fixed effects for each maker in each episode.
- **Weights:** every maker counts equally (inverse quote-count weights; decision 19).
- **Spread is not a control.** The tick mechanically widens it (standardized difference +1.5 at every
  window), so it is a consequence of the treatment (decision 21). It is reported as the first stage.

**The primary estimand is τ:** the change in how strongly makers unwind inventory by quote side when the
tick is 10x coarser.

**Hierarchy (confirmatory = one number):** the confirmatory result is τ from **SOL and XRP pooled**, with
its randomization p-value. SOL alone and XRP alone are component results, always reported whatever their
p-values. The primary result is never switched to whichever coin looks better.

---

## 5. Inference

- **Primary: a circular-shift randomization test.** It does not claim the boundary was randomly
  assigned; it asks whether the observed τ is unusual among label sequences with the same runs and
  timing, rotated against the quotes.
  1. Each episode window is cut into one-minute blocks, including minutes with no quotes. A minute's label is "above" if the median
     Hyperliquid mid in that minute is at or above the boundary; every quote in the minute takes that label.
  2. In each draw, every episode's sequence of minute labels is rotated by its own random offset k
     (uniform on 1 … M − 1, where M is the number of minutes in the episode window): minute m receives the
     label of minute (m + k) mod M. Rotation keeps the runs, the share of above-minutes and the time
     dependence, and breaks only the link between the label and that minute's quotes.
  3. Episodes with fewer than 5 above-minutes or fewer than 5 below-minutes keep their real labels in
     every draw (they add no permutation variation).
  4. SOL and XRP episodes are rotated independently in the same draw; τ is re-estimated on the pooled
     sample each draw, with the same model, sample, weights and fixed effects as the real estimate.
     Only the labels move. 2,000 draws, seed 20261112.
  5. Reported before τ: how many of the 31 episodes are rotated and how many keep fixed labels.
  - Two-sided p = (1 + #draws with |τ_perm| ≥ |τ_obs|) / 2,001.
  - Significance level: **5%**.
- **Secondary (shown, not decisive):**
  - the June-pilot null: the share of the 300 no-effect designs in `power_check.csv` with |τ| ≥ |τ_obs|;
  - day-clustered standard errors with a wild cluster bootstrap (14 clusters).

  The plain clustered t-test gave 9–14% false alarms in the pilot, so it is never used to decide significance.
- **Power (pilot estimate):** in 300 copies of the 31-episode, 14-day design on June data, an effect
  injected into every quote's outcome (refits matched the expected shift to 1e-15) was detected 80% of
  the time at a change in side skew of 14% of its baseline (baseline −0.108 bid share per 1 std of
  inventory), using the June-null rule. This is a pilot-based estimate under that rule, not a guarantee
  for the circular-shift test above.

---

## 6. Secondary outcomes (Holm-corrected as one family of 3)

1. **Price shading:** y = (quote price − mid) / mid in bp, within 5 bp of own best. Here **above = the
   quote's own price** at or above the boundary. Fixed effects are maker × side × episode. τ on q × above.
2. **Size skew:** y = log quote size. The regressor is q·s (s = +1 for asks, −1 for bids). τ on (q·s) × above.
3. **Presence at the best price:** y = 1 if the quote sits at the best price. The estimand is the level
   jump c at the boundary.

Price shading within 2 ticks of best (the original outcome) is **descriptive only**. Its baseline was
clear in just 15% of pilot designs.

**First stage (reported, not tested):** spread in bp and the share of quotes on the coarse grid, above vs below.

---

## 7. Falsification and robustness

| Check | Rule |
|---|---|
| **8 placebo levels** (SOL $95, $105, $110, $120; XRP $1.10, $1.30, $1.40, $1.50) | Same model at each level. Each placebo τ must be shown smaller than 25% of the main τ (90% interval inside ±0.25·|τ_main|), otherwise no tick claim |
| **Clean round-number placebo** (decision 22): powers of ten with no tick change and ≥ $250K/hour traded near the level: **PENGU $0.01** (59 episodes, 22 days), **STRK $0.10** (2 episodes, 2 days), **FARTCOIN $0.10** (1 episode, 1 day), 62 episodes on 25 days, as listed in the frozen `clean_placebo_scan.csv`. **PENGU alone decides** whether the clean placebo passes, using the same estimator, the same circular-shift procedure and the same 90%-interval equivalence rule; STRK and FARTCOIN are descriptive only and cannot fail the test on their own | Same rule. If the effect appears there too, it is reported as a round-number effect, not a tick effect |
| **Donut** | Drop quotes with the mid within ±0.05% of the boundary |
| **Direction** | Upward and downward crossings separately |
| **Per maker** | τ per maker with at least 1,000 quotes on each side of the boundary. The effect must have the same sign as the pooled τ in strictly more than 50% of these makers |
| **Binance running variable** | Binance mid at or before each quote (fuzzy design); all 53 episodes including poor-book hours; Binance prices older than 2 s dropped |
| **Weights and controls** | Quote-weighted τ; model with spread as a control |
| **Units** | Account, sleeve and firm levels; natives only; without the 4 firms with 30+ accounts; without accounts below 99% inventory-chain match |

---

## 8. Replication

The **15 coin-boundaries** with a 10x tick change, crossed and liquid (≥ $250K/hour near the boundary;
`request_coin_days.csv`) are analysed with the **identical frozen code** once delivered (decision 15).
If quote data cannot be obtained, the replication uses fills from the public API (side of maker fills vs
inventory). This is stated as weaker.

**Delivery 3 (received Nov 11, 2026; 385 of 386 coin-days, 18 coins).** Rules fixed from the vendor's README,
before any of its data was opened:

1. **What is exact on every day:** block numbers, timestamps, every placement and its price, side and size,
   and all fills (no fill gaps). The primary outcome (quote side) and inventory (fills) use only these, so
   they are used on all delivered days.
2. **Before May 1, 2026** (118 coin-days): the order of events *inside* a block is not the execution order,
   and some cancels are missing around the vendor's capture restarts. So on these days nothing that
   depends on within-block order or on rebuilding a maker's open orders is used (queue position, resting
   depth, presence at the best price from the rebuilt book). Rebuilt-book checks use May-onwards days only,
   where all 51 snapshot points rebuild exactly.
3. **Capture gaps** (no order events; listed in the vendor README: Mar 16–17 for DOGE, LINK, SUI; Apr 2 for
   AAVE, LIT, XPL; Apr 10 for LIT, 6 h 50 min; Jun 5 for FARTCOIN): quotes from the start of each gap to 10
   minutes after it ends are excluded, the same treatment as the Aug 29 gap. Measures that need a rebuilt
   book are excluded from the gap to its restart snapshot (for DOGE, LINK, SUI from Mar 16 20:21 to Mar 17
   22:00 UTC).
4. **Runs starting Mar 10–11** (DOGE, LIT, RESOLV, SUI, XPL): order events start at 22:00 UTC on Mar 10, so
   episodes before that time have no quotes and drop out; this is reported, not filled in.
5. The same book-clock, balance, window, q and episode rules as the main test apply. A coin-boundary
   with fewer than 3 episodes after these rules is reported descriptively only.

---

## 9. How results will be reported

| Result | Statement |
|---|---|
| τ significant (RI p < 5%), placebos pass, clean placebo passes | The coarser tick changes how makers manage inventory by side (causal, within the window) |
| τ significant, but a placebo or the clean placebo fails | An effect at round-number boundaries, not attributable to the tick |
| τ not significant | The estimate and its confidence interval are reported. The June pilot suggests effects of about 14% of the baseline would usually be detected under the June-null rule; this is not a formal equivalence conclusion for the primary test |
| Balance fails on the replication coins | Replication reported as descriptive |

All results are reported, including nulls.

---

## 10. Pre-analysis decisions

The 23 decisions in the project plan (Oct 29 – Nov 10, 2026) are part of this pre-registration. Each was
taken before the outcome was examined:

- 1–5: episodes, June as pilot, placebo levels, tick rule, quote definition
- 6: clock alignment
- 7–9: wallet ownership, within-firm coordination, large multi-account firms
- 10–11: inventory chain, data gaps
- 12–13: book depth, firm identity
- 14: episodes by book quality
- 15: replication coins
- 16: toxicity controls
- 17: inventory scaling and cap
- 18–19: native vs hedger, weighting and straddle
- 20: power check and primary outcome
- 21: balance and window
- 22: clean round-number placebo
- 23: native/hedger calibration

---

## 11. Deviations

Any change after the freeze is listed in `docs/DEVIATIONS.md` with its date and reason. Results under
the frozen rules are always shown next to any deviation.

---

*Data: 0xArchive, Hyperliquid historical market data, https://0xarchive.io, accessed November 2026.
Wallet addresses are shortened and firms are anonymous.*
