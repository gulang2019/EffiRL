#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic parquet validation subset.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--size", type=int, required=True, help="Number of rows to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("inputs", nargs="+", help="Input parquet files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")

    tables = [pq.read_table(path) for path in args.inputs]
    table = pa.concat_tables(tables, promote_options="default")
    num_rows = table.num_rows

    if args.size >= num_rows:
        subset = table
    else:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(num_rows, size=args.size, replace=False))
        subset = table.take(pa.array(indices))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, output_path)
    print(f"wrote {subset.num_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
