#!/usr/bin/env python3
"""Find all Bitcoin-related markets in Polymarket data."""

import signal
from pathlib import Path

import pandas as pd

# Handle broken pipe gracefully
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def find_bitcoin_markets():
    """Find all markets with 'bitcoin' or 'btc' in the question or slug."""
    data_dir = Path("data/polymarket/markets")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        print("No parquet files found")
        return

    all_bitcoin_markets = []

    for pf in parquet_files:
        df = pd.read_parquet(pf)

        # Search for bitcoin/btc in question and slug (case insensitive)
        mask = (
            df["slug"].str.lower().str.contains("btc-updown-5m", na=False)
            # df["question"].str.lower().str.contains("bitcoin", na=False)
            # | df["question"].str.lower().str.contains("btc", na=False)
            # | df["slug"].str.lower().str.contains("bitcoin", na=False)
            # | df["slug"].str.lower().str.contains("btc", na=False)
        )

        bitcoin_markets = df[mask]
        if not bitcoin_markets.empty:
            all_bitcoin_markets.append(bitcoin_markets)

    if not all_bitcoin_markets:
        print("No Bitcoin-related markets found")
        return

    result = pd.concat(all_bitcoin_markets, ignore_index=True)

    # Sort by volume descending
    result = result.sort_values("volume", ascending=False)

    print(f"Found {len(result)} Bitcoin-related markets\n")
    print("=" * 80)

    for _, row in result.iterrows():
        print(f"ID: {row['id']}")
        print(f"Question: {row['question']}")
        print(f"Slug: {row['slug']}")
        print(f"Volume: ${row['volume']:,.2f}")
        print(f"Active: {row['active']}, Closed: {row['closed']}")
        print("-" * 80)

    # Also save to CSV for further analysis
    output_path = Path("output/bitcoin_markets.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    find_bitcoin_markets()
