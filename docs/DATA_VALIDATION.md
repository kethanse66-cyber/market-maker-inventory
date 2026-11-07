# Data validation report

Status as of 7 November 2026. Every number below comes from a script in this repository; the script is named
next to each section. Firms are anonymous; wallet addresses are never shown in full.

## 1. Sources and coverage

| Source | What | Coverage |
|---|---|---|
| 0xArchive export, delivery 1 | Order events (all statuses), fills, resting-order snapshots for 124 market-maker wallets | BTC, SOL, XRP, HYPE |
| 0xArchive export, delivery 2 | Same, ETH for 44 wallets plus 80 further wallets on all coins | ETH + all coins |
| 0xArchive API | Whole-market trades, top of book every 0.5 s, 10-level book on 35 near-boundary days | 5 coins |
| Binance public data | 1-second prices (reference price) | BTC, ETH, SOL, XRP; HYPE from Sep 24 |

Windows: **Jun 1-14** and **Aug 15-Oct 25, 2026** (86 days). About 4.09 billion export rows in total.
Snapshots exist at May 31 23:58:55, Jun 5 22:38:51, Aug 14 23:58:42 and Oct 25 12:52:39 UTC.

## 2. Export against independent data

- Fills in the export match our own API trades on **344 of 344** coin-days.
- Order placements match on **1,332 of 1,332** sampled wallet-days.
- The only difference (Aug 15) is explained: the first export file starts at the snapshot block just before midnight.
- No wallet appears in both deliveries; no duplicate order events; no duplicate trade ids.

## 3. Rebuilding the order book (`day3_rebuild.py`, `day4_rebuild_all.py`)

**One wallet, one day (SOL, Jun 1, busiest maker).** 1,039,618 orders placed and 1,039,619 endings: every
ending hits a known order (0 misses). 3,237 fills: 0 invalid (1,782 share a block with the order's birth or
end and cannot be ordered within the block). All 1,604 fully filled orders executed exactly their size;
1,085 orders were partly filled, then cancelled. No order opened or ended twice.

**Every wallet, every day, checked against the next snapshot.**

| | SOL Jun | SOL Aug-Oct | XRP Jun | XRP Aug-Oct |
|---|---|---|---|---|
| Events replayed | 83.7 M | 419.0 M | 32.7 M | 344.0 M |
| Snapshot orders found in the rebuild | 102 / 122 | **244 / 244** | 189 / 212 | **445 / 445** |
| ...with the same owner and remaining size | 100% | 100% | 100% | 100% |
| Snapshot-only orders not explained by an outage | 0 | 0 | 0 | 0 |
| Rebuild-only orders not explained | 0 | 0 (93 ambiguous) | 0 | 0 (115 ambiguous) |

Every June difference is an order born inside, or ending inside, the Jun 5 outage. All ambiguous Aug-Oct
orders were born on Aug 29, which led to the gap in section 4.

**Not quotes, excluded by definition:** about 4,500 fills from TWAP slices and 7 liquidation orders have no
`open` event in the order stream.

## 4. Data gaps (`day4b_gaps.py`, `day4c_aug29.py`)

| Gap | Data | Treatment |
|---|---|---|
| Jun 5 22:00:10-22:24:16 UTC | order events | excluded |
| **Aug 29 20:21:29-20:21:37 UTC** (found by the rebuild) | order events, SOL and XRP | 20:21-20:32 excluded; orders open across it dropped |
| Oct 25 09:32:20-10:28:57 UTC | order events | excluded |
| Jun 7-9 and Jun 14 | top of book | quotes kept, market unknown |
| Saturdays ~08:30 UTC, ~1 minute | all coins at once (chain pause) | kept: zero mismatches |

## 5. Inventory (`day5_inventory.py`)

Each fill reports the account's position just before it. Check: one fill's position after must equal the
account's next position before.

| Coin | Firm fills | Links that match | Accounts below 99% |
|---|---|---|---|
| SOL | 6.67 M | 99.96% | 0 of 194 |
| XRP | 3.19 M | 99.99% | 0 of 145 |
| BTC | 24.45 M | 99.82% | 4 of 316 |
| ETH | 12.19 M | 99.89% | 3 of 267 |
| HYPE | 20.33 M | 99.93% | 3 of 195 |

No missing days, no duplicate trade ids. The BTC and ETH shortfall comes mostly from two accounts of one
firm; without them BTC is 99.92% and ETH 99.91%. A rare exchange-side reporting quirk (one order hit by two
counterparties in the same instant, both fills reporting the same position before) appears 74 times in
total. Because every fill carries its own position before, a break only affects the interval up to the next
fill and never accumulates. Accounts below 99% are flagged, and every inventory result is also shown without
them.

## 6. Book clock (`day6b_clock.py`, `day6c_lag.py`)

A post-only quote cannot cross the real book, so for each hour the book delay is the one at which the fewest
quotes look crossed (grid -0.5 to +3.0 s; on simulated data with planted delays it recovers them exactly).

| | SOL | XRP | BTC | ETH | HYPE |
|---|---|---|---|---|---|
| Median delay | 0.375 s | 0.375 s | 0.625 s | 0.625 s | 0.375 s |
| Hours flagged poor (> 2% still crossed) | 13% | 25% | 20% | 19% | 48% |

The delay is a steady 0.25 s in June and from Oct 2 onwards, and 0.4-2 s and unsteady from Aug 15 to Oct 1.
The worst hours (up to 49% crossed) are 12:00-14:00 UTC on Sep 4, 11 and 17, around US macro releases.

## 7. Quote table (`day6_quotes.py`)

A quote is the placement of a post-only limit order that is neither reduce-only nor a trigger order.

| | SOL | XRP | BTC | ETH | HYPE |
|---|---|---|---|---|---|
| Quotes | 288 M | 205 M | 696 M | 398 M | 430 M |
| On the legal tick grid | 100% | 100% | 100% | 100% | 100% |
| Inventory known | 99.95% | 99.95% | 100.0% | 99.6% | 100.0% |
| Still crossing the book (flagged) | 3.8% | 4.5% | 5.8% | 5.3% | 5.6% |
| **Main sample**: within 2 ticks of the best price, good-book hour, outside gaps, not crossing | 64.3 M | 42.2 M | 114.6 M | 109.6 M | 37.3 M |

Hand checks of 20 random quotes per coin: distances, ticks and inventory reproduce by hand; the only odd rows
are quotes already flagged as crossing the book.

## 8. Tick-boundary episodes (`day7_episodes.py`, `day7b_tick_scan.py`)

Crossings of the boundary by the minute mid price, with windows of +-30 minutes and overlapping windows merged.

| | Episodes | Days | In good-book hours |
|---|---|---|---|
| SOL $100 | 32 | 13 | 15 (9 days) |
| XRP $1.00 | 21 | 5 | 16 (5 days) |

The primary test uses the 31 good-book episodes; all 53 are used with Binance prices as a robustness check.
A scan of all 179 perps (Mar 10-Oct 25) found 15 further coin-boundaries with a 10x tick change, crossed and
liquid (at least $250K traded per hour near the boundary); 344 coin-days have been requested as an independent
replication sample.

## 9. Known limitations

- Rejected post-only orders are not in the order history, so queue races are not modelled in the panel.
- Book quality is weaker from Aug 15 to Oct 1, which overlaps the main SOL/XRP episodes (sections 6 and 8).
- An account's inventory is its Hyperliquid position only; positions on other venues are not visible.
- Firm groupings come from on-chain sub-account links and can miss accounts a firm keeps separate.
