import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

API_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
RATE_LIMIT_SLEEP = 0.2  # ~5 requests per second
MAX_CANDLES_PER_REQUEST = 300

def fetch_btc_5min_ohlcv(start: str, end: str) -> pd.DataFrame:
    """Fetches BTC-USD 5-minute OHLCV data from Coinbase within the given date range."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    all_data = []
    current_start = start_dt
    delta = timedelta(minutes=5 * MAX_CANDLES_PER_REQUEST)
    
    print(f"Fetching Coinbase BTC-USD 5-min OHLCV from {start} to {end}...")
    
    while current_start < end_dt:
        current_end = min(current_start + delta, end_dt)
        
        params = {
            "granularity": 300,  # 5 minutes in seconds
            "start": current_start.isoformat(),
            "end": current_end.isoformat()
        }
        
        print(f"  Requesting {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}...")
        
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            
            chunk = response.json()
            if not chunk:
                print("    No data returned for this chunk.")
            else:
                all_data.extend(chunk)
                print(f"    Fetched {len(chunk)} candles.")
                
        except Exception as e:
            print(f"    Error fetching chunk: {e}")
            
        current_start = current_end
        time.sleep(RATE_LIMIT_SLEEP)
        
    if not all_data:
        print("Warning: No data fetched.")
        return pd.DataFrame()
        
    # Coinbase returns: [[time, low, high, open, close, volume], ...]
    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download Coinbase BTC-USD public 5-min OHLCV data")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-04-08", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = fetch_btc_5min_ohlcv(args.start, args.end)
    
    if df.empty:
        print("No data to save.")
        return
        
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "coinbase_btc_5min.json")
    
    df.to_json(out_file, orient="records", indent=2)
    print(f"Saved {len(df)} records to {out_file}")

if __name__ == "__main__":
    main()