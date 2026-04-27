# ARCADIA CONTRIBUTION — pending review
# Tardis.dev historical order book collection (requires API key)
import os
import json
import gzip
import argparse
import asyncio
from datetime import datetime, timedelta

from tardis_client import TardisClient, Channel


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def date_range_inclusive(start_date: str, end_date: str):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    current = start_dt
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


async def collect_one_day(
    day: str,
    symbol: str,
    api_key: str,
    out_dir: str,
    channel_name: str,
    max_messages: int | None = None,
) -> None:
    ensure_dir(out_dir)

    next_day = (
        datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    out_file = os.path.join(out_dir, f"{symbol}_{channel_name}_{day}.jsonl.gz")
    count = 0

    print(f"[INFO] Collecting {channel_name} for {symbol} on {day}")

    client = TardisClient(api_key=api_key)

    with gzip.open(out_file, "wt", encoding="utf-8") as f:
        async for msg in client.replay(
            exchange="binance",
            from_date=day,
            to_date=next_day,
            filters=[Channel(name=channel_name, symbols=[symbol])],
        ):
            record = {
                "local_timestamp": msg.local_timestamp,
                "message": msg.message,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

            if count == 1 or count % 10000 == 0:
                print(f"[OK] {day} {channel_name}: saved {count} messages")

            if max_messages is not None and count >= max_messages:
                print(f"[INFO] Reached max_messages={max_messages} for {day}")
                break

    print(f"[DONE] {day} {channel_name}: total saved = {count}")
    print(f"[FILE] {out_file}")


async def main_async(args):
    symbol = args.symbol.upper()
    root_dir = args.out_dir
    ensure_dir(root_dir)

    for day in date_range_inclusive(args.start_date, args.end_date):
        if args.collect_depth_snapshot:
            snapshot_dir = os.path.join(root_dir, "tardis_depthSnapshot")
            await collect_one_day(
                day=day,
                symbol=symbol,
                api_key=args.api_key,
                out_dir=snapshot_dir,
                channel_name="depthSnapshot",
                max_messages=args.max_messages,
            )

        if args.collect_depth:
            depth_dir = os.path.join(root_dir, "tardis_depth")
            await collect_one_day(
                day=day,
                symbol=symbol,
                api_key=args.api_key,
                out_dir=depth_dir,
                channel_name="depth",
                max_messages=args.max_messages,
            )

    note_path = os.path.join(root_dir, "TARDIS_ORDERBOOK_NOTE.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(
            "Historical Binance Spot order-book collection via Tardis\n"
            f"symbol={symbol}\n"
            f"start_date={args.start_date}\n"
            f"end_date={args.end_date}\n"
            f"collect_depth_snapshot={args.collect_depth_snapshot}\n"
            f"collect_depth={args.collect_depth}\n"
            f"max_messages={args.max_messages}\n"
            "\n"
            "Files are saved as gzipped JSONL.\n"
            "depthSnapshot contains snapshot-style order-book messages.\n"
            "depth contains incremental depth update messages.\n"
        )
    print(f"[DONE] Wrote note file: {note_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--start_date", type=str, default="2026-03-17")
    parser.add_argument("--end_date", type=str, default="2026-03-21")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--collect_depth_snapshot", action="store_true")
    parser.add_argument("--collect_depth", action="store_true")
    parser.add_argument("--max_messages", type=int, default=None)

    args = parser.parse_args()

    if not args.collect_depth_snapshot and not args.collect_depth:
        raise ValueError(
            "Select at least one of --collect_depth_snapshot or --collect_depth"
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()