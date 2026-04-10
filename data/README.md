# Data Directory

This directory stores Binance BTCUSDT trade data downloaded from
`https://data.binance.vision/`.

## How to download

```bash
cd /path/to/mf796_project
python -m calibration.download_binance --days 7
```

## File format

Each file is named `BTCUSDT-aggTrades-YYYY-MM-DD.csv` and contains
aggregated trade records with **no header row**. Columns are:

| # | Column           | Type    | Description                          |
|---|------------------|---------|--------------------------------------|
| 0 | agg_trade_id     | int     | Aggregate trade ID                   |
| 1 | price            | float   | Trade price                          |
| 2 | quantity         | float   | Trade quantity (BTC)                 |
| 3 | first_trade_id   | int     | First individual trade ID in agg     |
| 4 | last_trade_id    | int     | Last individual trade ID in agg      |
| 5 | timestamp        | int     | Unix timestamp in microseconds       |
| 6 | is_buyer_maker   | bool    | True = seller initiated (taker sell) |
| 7 | is_best_match    | bool    | Best price match                     |

## Side convention

- `is_buyer_maker = True` means the buyer's limit order was filled by
  the seller's market order, so the **taker is selling** (side = -1).
- `is_buyer_maker = False` means the **taker is buying** (side = +1).

## Source

Free public data from Binance:
https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/
