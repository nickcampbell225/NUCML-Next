#!/bin/bash
# Find all Parquet datasets on the system and report their file counts

echo "Searching for Parquet datasets..."
echo "=================================="
echo

# Common locations to search
search_paths=(
    "$HOME"
    "/mnt"
    "/data"
    "."
)

for base_path in "${search_paths[@]}"; do
    if [ -d "$base_path" ]; then
        echo "Searching in: $base_path"

        # Find directories ending in .parquet
        find "$base_path" -type d -name "*.parquet" 2>/dev/null | while read parquet_dir; do
            # Count files and directories
            file_count=$(find "$parquet_dir" -type f -name "*.parquet" 2>/dev/null | wc -l)
            dir_count=$(find "$parquet_dir" -type d 2>/dev/null | wc -l)
            size=$(du -sh "$parquet_dir" 2>/dev/null | cut -f1)

            # Only show datasets with files
            if [ "$file_count" -gt 0 ]; then
                echo "  📊 $parquet_dir"
                echo "      Size: $size"
                echo "      Files: $file_count"
                echo "      Dirs: $dir_count"
                echo
            fi
        done
    fi
done

echo "=================================="
echo "Search complete!"
echo
echo "To consolidate a dataset, run:"
echo "  python scripts/consolidate_parquet.py <input_path> <output_path> --partition none"
