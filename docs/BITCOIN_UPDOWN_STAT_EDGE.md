# Bitcoin Up/Down Statistical Edge Analysis

This document describes `bitcoin_updown_stat_edge`, a Polymarket analysis that
extracts relative statistical edge from `Bitcoin Up or Down` markets using
probability/statistics only (no forecasting model, no technical indicators).

## What It Does

The analysis:

1. Filters markets to questions that start with `Bitcoin Up or Down`.
2. Keeps only semantic `["Up","Down"]` outcome pairs.
3. Marks a market as resolved only when outcome prices satisfy:
   - `max(price) >= 0.999`
   - `min(price) <= 0.001`
4. Maps CTF trade rows to market outcomes through token IDs.
5. Computes trade-level edge:
   - `edge = won - p`
   - `edge_bps = 10000 * edge`
6. Builds peer-relative features by `tenor_bucket + tte_bucket`:
   - `rel_price_z` via median/MAD normalization
   - `rel_notional_pct` via within-peer percentile rank
7. Screens candidate signals with blocked train/val/test splits on `end_ts`.

## Tenor Buckets

- `15m`: explicit ranged questions like `5:00PM-5:15PM ET`
- `5m`: explicit ranged questions like `5:00PM-5:05PM ET`
- `1h`: hourly questions like `5PM ET`
- `1d`: daily questions like `Bitcoin Up or Down on February 5?`
- `other`: excluded from V1 signal estimation

## TTE Buckets

- `[0,5m)`
- `[5,15m)`
- `[15,60m)`
- `[1,4h)`
- `[4,12h)`
- `[12,24h)`
- `[24,72h)`
- `[72h+]`

## Signal Selection Rules

Blocked split by market `end_ts`:

- train: 70%
- val: 15%
- test: 15%

Candidate grid:

- `rel_price_z` bins: `(-inf,-2]`, `(-2,-1]`, `(-1,1]`, `(1,2]`, `(2,inf)`
- `rel_notional_pct` bins: `[0,50)`, `[50,80)`, `[80,95)`, `[95,100]`
- crossed with `tenor_bucket` and `tte_bucket`

Train gate:

- `n_trades_train >= 300`
- `n_markets_train >= 30`
- `|ev_train| >= 0.0075`
- BH-FDR `q_value < 0.10`
- equal-market and equal-notional signs agree

Validation gate:

- `sign(ev_val) == sign(ev_train)`

Final gate:

- `sign(ev_test) == sign(ev_train)`

## Output Files

Running:

```bash
uv run main.py analyze bitcoin_updown_stat_edge
```

produces:

- `output/bitcoin_updown_stat_edge.csv`
- `output/bitcoin_updown_edge_candidates.csv`
- `output/bitcoin_updown_coverage.json`
- `output/bitcoin_updown_stat_edge.png`
- `output/bitcoin_updown_relative_edge_heatmap.png`
- `output/bitcoin_updown_stat_edge.json` (chart config)

## CSV Contracts

### `bitcoin_updown_stat_edge.csv`

Columns:

- `market_id`
- `question`
- `tenor_bucket`
- `trade_ts`
- `end_ts`
- `tte_bucket`
- `token_id`
- `outcome_label`
- `p`
- `shares`
- `notional`
- `won`
- `edge`
- `edge_bps`
- `peer_key`
- `rel_price_z`
- `rel_notional_pct`

### `bitcoin_updown_edge_candidates.csv`

Columns:

- `signal_id`
- `rule_def`
- `n_trades_train`
- `n_markets_train`
- `ev_train`
- `ev_val`
- `ev_test`
- `win_gap_test`
- `q_value`
- `effect_size`
- `equal_market_sign`
- `equal_notional_sign`

### `bitcoin_updown_coverage.json`

Fields:

- `market_total`
- `resolved_market_total`
- `token_total`
- `token_seen_in_trades`
- `token_coverage_rate`
- `trade_rows_mapped`
- `unmapped_trade_rows`
