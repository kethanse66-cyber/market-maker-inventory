# Pre-registration addendum v6: is the defence against inventory driven by toxic fills (adverse selection)?

**Status:** frozen on November 14, 2026 (git tag `prereg-v6`), before the test below was run on any coin.

## 0. What is already known, and why this test is new
- prereg-v3 T1: makers respond to inventory mainly by quote side.
- prereg-v5 (H5 not supported, opposite sign): the price response is defensive. The quote on the side that would
  ADD inventory moves away from the mid about twice as much as the quote on the unwinding side moves toward it.
- **No model so far has used what happened to the price after a maker's own fills.** That is the new variable here.

## 1. Hypothesis
If makers defend against inventory because the trades that build it often come from better-informed traders
(adverse selection), the defence should be stronger right after the maker's recent fills turned out **toxic**
(the price moved against the maker after the fill), and weaker after harmless fills. GLFT has no such link: its
skew depends on inventory, volatility and risk aversion only.

## 2. Toxicity measure (only information the maker already had at the quote time)
For each fill of the maker account (inventory tables, one row per fill): direction d = +1 if it raised the position
(maker bought), -1 if it lowered it; second of the fill t. **Markout** (bp) = d x (log mid at t+H - log mid at
t-1) x 10,000, with the per-second market mid from the quote tables (the frozen vol5 source). A mid is used only
if its last observation is at most 5 s old (the frozen vol5 staleness rule); otherwise that markout (or return) is
missing, never filled in.
Negative markout = the price moved against the maker = toxic.

For a quote at time tau: fills of the same account with fill time in [tau - 300 s, tau) AND markout already
complete (t + H + 1 s <= tau). **T** = - (notional-weighted mean markout of those fills) / (sigma_H), where
sigma_H = coin-day median vol5 x sqrt(H) (bp), capped at +-5. T > 0 = recent fills were toxic.
Main horizon **H = 10 s**. Quotes with no qualifying fill are not in the test sample (they have no toxicity).

## 3. Test (confirmatory sample: SOL, XRP, BTC, ETH, HYPE, all 86 days; Part 1 tables; same makers as day21)
Side model:
bid = b q + **c (q x T)** + e T + d vol5 + f1 r10 + f2 r60 + f3 r300 + h (q x vz) + FE (maker x day)
- r10, r60, r300 = market log return (bp) over the 10, 60, 300 s up to the second before the quote; vz = vol5
  standardized within coin (as T2). These keep "toxic" from meaning just "the price was falling" or "volatile".
- Every maker weighted equally; coin-day clusters (as prereg-v3).
- **H6:** c < 0, p < 0.05 (two-sided): the side response to inventory is stronger after toxic fills.
- Only quotes with T and all three returns known enter (complete cases); the script reports both shares.
  Maker- and day-clustered p-values are reported next to the coin-day one (not decisive).

Secondary (reported, not decisive):
- the aggressiveness model of prereg-v5 with u+ x T and u- x T added (prediction: the accumulating-side term
  moves the quote further away when T is high);
- H = 5 s and H = 30 s;
- each coin; the 16 replication coins (descriptive).

## 4. Reading the results
- H6 supported: "the defence is stronger after toxic fills - consistent with adverse selection, which GLFT
  omits". Still not proof of the makers' reason.
- Not supported: reported as is; the paper stays descriptive about why.

## 5. Unchanged
All prereg-v1 to v5 definitions. Full wallet addresses never leave the PC.
