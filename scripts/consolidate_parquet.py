#!/usr/bin/env python3
"""
Consolidate over-partitioned Parquet dataset into fewer, larger files.

This script reads a partitioned Parquet dataset and rewrites it with:
1. Option: No partitioning (single file or row-grouped files)
2. Option: Partition by Z only (one file per element)
3. Option: Partition by Z, A (one file per isotope)

For a 702MB dataset, this will reduce from 57K files to 3-100 files,
reducing load time from 10 minutes to seconds.
"""

import argparse
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
from pathlib import Path
import time


def consolidate_parquet(
    input_path: str,
    output_path: str,
    partition_cols: list[str] | None = None,
    row_group_size: int = 100_000,
):
    """
    Consolidate a partitioned Parquet dataset.

    Args:
        input_path: Path to input partitioned dataset
        output_path: Path to output consolidated dataset
        partition_cols: Partition columns (None for no partitioning, ['Z'] for Z only, ['Z', 'A'] for Z/A)
        row_group_size: Rows per row group (larger = fewer files)
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Partition columns: {partition_cols or 'None (consolidated)'}")
    print(f"Row group size: {row_group_size:,}")
    print("=" * 80)

    # Read the entire dataset
    print("\n1. Reading partitioned dataset...")
    start = time.time()

    dataset = ds.dataset(input_path, format='parquet')
    table = dataset.to_table()

    elapsed = time.time() - start
    print(f"   ✓ Read {len(table):,} rows in {elapsed:.1f}s")
    print(f"   Schema: {table.schema}")
    print(f"   Memory size: {table.nbytes / 1e6:.1f} MB")

    # Write consolidated dataset
    print("\n2. Writing consolidated dataset...")
    start = time.time()

    output_path = Path(output_path)
    if output_path.exists():
        print(f"   ⚠️  Output path exists, removing: {output_path}")
        import shutil
        shutil.rmtree(output_path)

    if partition_cols:
        # Write with partitioning
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=partition_cols,
            existing_data_behavior='overwrite_or_ignore',
            max_rows_per_file=row_group_size,
            use_dictionary=True,
            compression='snappy',
        )
        print(f"   ✓ Wrote partitioned dataset (partition_cols={partition_cols})")
    else:
        # Write as single file (or chunked by row groups)
        pq.write_table(
            table,
            str(output_path / 'data.parquet'),
            row_group_size=row_group_size,
            use_dictionary=True,
            compression='snappy',
        )
        print(f"   ✓ Wrote single file dataset")

    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.1f}s")

    # Verify output
    print("\n3. Verifying output...")
    start = time.time()

    if partition_cols:
        # Count files
        files = list(output_path.rglob('*.parquet'))
        print(f"   ✓ Created {len(files):,} files")

        # Show file sizes
        total_size = sum(f.stat().st_size for f in files)
        avg_size = total_size / len(files)
        print(f"   Total size: {total_size / 1e6:.1f} MB")
        print(f"   Avg file size: {avg_size / 1e6:.1f} MB")

        # Show sample paths
        print(f"\n   Sample files:")
        for f in sorted(files)[:5]:
            rel_path = f.relative_to(output_path)
            size = f.stat().st_size / 1e6
            print(f"     {rel_path} ({size:.2f} MB)")
    else:
        files = list(output_path.glob('*.parquet'))
        total_size = sum(f.stat().st_size for f in files)
        print(f"   ✓ Created {len(files)} file(s), {total_size / 1e6:.1f} MB total")
        for f in files:
            size = f.stat().st_size / 1e6
            print(f"     {f.name}: {size:.1f} MB")

    # Test read speed
    print("\n4. Testing read speed...")
    start = time.time()

    if partition_cols:
        test_dataset = ds.dataset(output_path, format='parquet')
        test_table = test_dataset.to_table()
    else:
        test_table = pq.read_table(str(output_path / 'data.parquet'))

    elapsed = time.time() - start
    print(f"   ✓ Read {len(test_table):,} rows in {elapsed:.1f}s")
    print(f"   Speedup estimate: {10 * 60 / elapsed:.1f}x faster (from 10 min to {elapsed:.1f}s)")

    print("\n" + "=" * 80)
    print("✓ Consolidation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate over-partitioned Parquet dataset")
    parser.add_argument("input_path", help="Path to input partitioned dataset")
    parser.add_argument("output_path", help="Path to output consolidated dataset")
    parser.add_argument(
        "--partition",
        choices=["none", "Z", "Z-A"],
        default="none",
        help="Partitioning scheme (none=single file, Z=by element, Z-A=by isotope)",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100_000,
        help="Rows per row group (default: 100,000)",
    )

    args = parser.parse_args()

    # Map partition choice to columns
    partition_map = {
        "none": None,
        "Z": ["Z"],
        "Z-A": ["Z", "A"],
    }
    partition_cols = partition_map[args.partition]

    consolidate_parquet(
        args.input_path,
        args.output_path,
        partition_cols=partition_cols,
        row_group_size=args.row_group_size,
    )
