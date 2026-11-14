# Pre-registration addendum v7: does makers' inventory move the price? (price pressure vs Binance)

**Status:** frozen on November 14, 2026 (git tag `prereg-v7`), before the test below was run on any coin.

## 0. What is already known, and why this test is new
Part 1 (prereg-v3/v5/v6): makers respond to their own inventory mainly by pulling the side that would add to it,
and move that side's remaining quotes away. **No test so far has asked whether this changes the price.**
Here the unit is the market (coin x minute), not the quote, and the outcome is the Hyperliquid price relative to
Binance. In stocks, specialists' inventory predicts reversals (Hendershott & Seasholes; Hendershott & Menkveld).

## 1. Hypothesis (price pressure)
When the makers as a group are long, buying support on Hyperliquid thins out (bids pulled), so the Hyperliquid
price sits **below** Binance; as makers unwind, the gap closes, so Hyperliquid **outperforms** Binance next.
The opposite explanation (arbitrageurs buy on Hyperliquid when it is cheap, from makers' asks, making makers
SHORT when Hyperliquid is cheap) predicts the opposite sign in H7a. The sign therefore separates the two.

## 2. Variables (coin x minute; SOL, XRP, BTC, ETH, HYPE; all days with both prices)
- **Q_t** = sum of all maker firms' positions at the end of minute t (`<COIN>_firm_minute.parquet`, frozen Day 5).
  **z_t** = Q_t / std of Q over the previous 1,440 minutes, capped at +-5 (the q rule of prereg-v1 at market level).
- Hyperliquid mid at minute end: (bid + ask) / 2 of the last book snapshot in the minute, used only if it is at
  most 5 s before the minute's end. Binance price at minute end: the last 1-second price in the minute, same 5 s
  rule. Minutes without both are missing.
- **B_t** = (log HL mid - log Binance price) x 10,000 (bp). A constant basis (contract differences, funding) is
  removed by coin x day fixed effects.
- **dB_t(h)** = B_(t+h) - B_t: Hyperliquid's return minus Binance's over the next h minutes. Main h = 5.
- Controls: Hyperliquid and Binance log returns over the previous 5 minutes (bp).

## 3. Tests (confirmatory)
Every coin weighted equally; coin x day fixed effects; standard errors clustered by day (all coins share a day,
and a day absorbs within-day overlap of the 5-minute outcomes).
- **H7a (level):** B_t = a z_t + FE. Prediction **a < 0**, p < 0.05 (two-sided).
- **H7b (reversal):** dB_t(5) = b z_t + c1 r_HL(-5) + c2 r_BN(-5) + FE. Prediction **b > 0**, p < 0.05.
- **"Maker inventory moves the price" is supported if H7a AND H7b hold.** One alone: partial, named exactly.

## 4. Secondary (reported, not decisive)
- H7b adding B_t as a control (does inventory predict beyond the visible gap?);
- h = 1, 15, 60 minutes; each coin separately;
- **natives vs hedgers:** Q split by the firm-level labels of `natives.csv` (NATIVE vs HEDGER firms), each scaled
  as above, both in one model. Prediction: the pressure comes from natives (a hedger's Hyperliquid position is
  offset elsewhere);
- Hyperliquid's own future return (not relative to Binance), h = 5.

## 5. Limits stated in advance
Descriptive at the market level: makers' inventory and prices are both moved by order flow. A supported H7a/H7b
shows that maker inventory carries price pressure that later reverses, not that inventory alone causes it.

## 6. Unchanged
All prereg-v1 to v6 definitions. Full wallet addresses never leave the PC; firm ids are printed with 8 characters.
