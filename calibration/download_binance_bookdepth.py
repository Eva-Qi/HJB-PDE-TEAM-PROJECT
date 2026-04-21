import argparse
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta

# NOTE 2026-04-20 audit: Binance 'bookDepth' archive exists ONLY for
# USDⓈ-M futures, not spot (spot path → 404). Our execution model is
# spot BTCUSDT but BTC futures correlate >0.99 with spot, so futures
# bookDepth is a usable proxy for order-book depth (strictly better
# than the current VWAP-as-midprice proxy). Document this caveat in
# any downstream analysis.
BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{date}.zip"

def download_and_unzip(date_str: str, out_dir: str) -> bool:
    """Download and unzip bookDepth for a specific date. Returns True if downloaded."""
    zip_filename = f"BTCUSDT-bookDepth-{date_str}.zip"
    csv_filename = f"BTCUSDT-bookDepth-{date_str}.csv"
    zip_path = os.path.join(out_dir, zip_filename)
    csv_path = os.path.join(out_dir, csv_filename)

    # Skip if CSV already exists
    if os.path.exists(csv_path):
        print(f"[SKIP] {csv_filename} (CSV already exists)")
        return False

    url = BASE_URL.format(date=date_str)
    
    try:
        print(f"[DOWNLOADING] {url}")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[EXTRACTING] {zip_filename}")
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
            
        print(f"[SAVED] {csv_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to fetch {date_str}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Binance BTCUSDT L2 order book snapshot dumps (bookDepth)")
    parser.add_argument("--days", type=int, default=7, help="Number of days to download (default: 7)")
    parser.add_argument("--out-dir", type=str, default="data/", help="Output directory (default: data/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Downloading Binance BTCUSDT bookDepth for {args.days} days...")
    
    downloaded_count = 0
    skipped_count = 0

    for i in range(args.days):
        date_str = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        was_downloaded = download_and_unzip(date_str, args.out_dir)
        if was_downloaded:
            downloaded_count += 1
        else:
            skipped_count += 1

    print("-" * 40)
    print(f"Done. Downloaded: {downloaded_count}, Skipped/Failed: {skipped_count}")

if __name__ == "__main__":
    main()