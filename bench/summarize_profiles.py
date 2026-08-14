"""Summarize profile_server JSON outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.directory.glob('*.json'):
        if path.stat().st_size:
            rows.append(json.loads(path.read_text(encoding='utf-8')))
    for row in sorted(rows, key=lambda value: value['games_per_sec'], reverse=True):
        print(f"t{row['threads']} c{row['concurrent_games']}: "
              f"{row['games_per_sec']:.3f} games/s, "
              f"{row['positions_per_sec']:.1f} samples/s, {row['seconds']:.1f}s, "
              f"batch {row['batch_mean']:.1f}, encode {row['encode_seconds']:.1f}s, "
              f"net {row['net_seconds']:.1f}s, expand {row['expand_seconds']:.1f}s, "
              f"tree {row['tree_and_output_seconds']:.1f}s")


if __name__ == '__main__':
    main()
