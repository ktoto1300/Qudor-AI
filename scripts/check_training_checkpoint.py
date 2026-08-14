"""Print compact metadata from an AlphaZero training checkpoint."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quoridor_ai.safe_loader import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    data = load_checkpoint(args.checkpoint)
    print(json.dumps({
        'path': str(args.checkpoint),
        'iteration': data.get('iteration'),
        'generation': data.get('generation'),
        'global_step': data.get('global_step'),
        'rules_version': data.get('rules_version'),
        'encoding': data.get('encoding'),
        'replay': len(data.get('replay', ())),
    }, indent=2))


if __name__ == '__main__':
    main()
