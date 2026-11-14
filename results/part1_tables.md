# Part 1 tables

## Table 1. Which tool? (T1, per 1 sd of inventory q; 230 makers, 386 coin-days)

| Response | Estimate | 95% interval | p |
|---|---|---|---|
| Side: change in P(bid) | -0.0620 | [-0.0675, -0.0566] | 0.0000 |
| Price: shift in ticks | -0.2445 | [-0.2847, -0.2044] | 0.0000 |
| Size: log size on the unwinding side | +0.1466 | [+0.1241, +0.1692] | 0.0000 |

## Table 2. GLFT's other predictions (T2-T4)

| Test | GLFT predicts | Estimate | p | Verdict |
|---|---|---|---|---|
| T2 price x volatility | < 0 | +0.0541 | 0.0000 | not as GLFT |
| T3 side slope for small inventory | ~0 | -0.0957 | 0.0000 | not as GLFT |
| T4 Spearman(side response, half-life) | > 0 | +0.476 | 0.0001 | supported (202 makers) |

## Table 3. Robustness (side | price, per 1 sd of q)

| Version | Side | Price (ticks) |
|---|---|---|
| Main estimate (T1) | -0.0620 (se 0.0028) | -0.2445 (se 0.0205) |
| SOL only | -0.0512 (se 0.0054) | -0.0624 (se 0.0078) |
| XRP only | -0.0734 (se 0.0046) | -0.0542 (se 0.0088) |
| BTC only | -0.0449 (se 0.0042) | -0.3812 (se 0.0275) |
| ETH only | -0.0513 (se 0.0036) | -0.1219 (se 0.0106) |
| HYPE only | -0.0949 (se 0.0053) | -0.5194 (se 0.0937) |
| June only | -0.0568 (se 0.0075) | -0.2240 (se 0.0360) |
| Aug-Oct only | -0.0603 (se 0.0018) | -0.2666 (se 0.0198) |
| no fill within 1s | -0.0648 (se 0.0029) | -0.2560 (se 0.0200) |
| no fill within 5s | -0.0688 (se 0.0030) | -0.2554 (se 0.0192) |
| no fill within 10s | -0.0717 (se 0.0030) | -0.2554 (se 0.0191) |
| no fill within 30s | -0.0787 (se 0.0031) | -0.2564 (se 0.0196) |
| no fill within 60s | -0.0855 (se 0.0035) | -0.2646 (se 0.0195) |
| control past returns | -0.0608 (se 0.0027) | -0.2404 (se 0.0204) |
| no fill 60s + returns | -0.0846 (se 0.0035) | -0.2645 (se 0.0195) |
| natives only | -0.0731 (se 0.0062) | -0.2125 (se 0.0270) |
| hedgers only | -0.0497 (se 0.0038) | -0.3108 (se 0.0423) |
| q capped at 3 | -0.0762 (se 0.0037) | -0.2928 (se 0.0234) |
| rank q | -0.0710 (se 0.0032) | -0.2764 (se 0.0214) |
| without cap-heavy makers | -0.0698 (se 0.0032) | -0.2647 (se 0.0226) |
