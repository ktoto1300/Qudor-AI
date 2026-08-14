"""Summarize JSON results produced by tools/foreign_arena.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.glob('*.json'))
    print(f'completed: {len(files)}')
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        print(f"{path.stem}: W{data['wins']} D{data['draws']} L{data['losses']} "
              f"forfeits={data['opp_forfeits']} plies={data['avg_plies']}")


if __name__ == '__main__':
    main()
