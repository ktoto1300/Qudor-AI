"""Extract `latest.pt` checkpoints from a training zip into per-seed files.

The old night-run archives hold one `latest.pt` per seed under a seed-named folder;
this tool walks a zip and writes every one as `{seed}_legacy.pt` for pretraining.
"""
import io
import json
import zipfile
from pathlib import Path

from .safe_loader import load_checkpoint


def inspect_and_extract(zip_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    seen_dest = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith('latest.pt'):
                continue
            seed = Path(n).parent.name or 'unknown'   # a top-level entry has no seed
            dest = out / f'{seed}_legacy.pt'
            if dest.name in seen_dest:
                raise FileExistsError(
                    f'duplicate seed {seed!r} in {zip_path.name}: {seen_dest[dest.name]} '
                    f'and {n} both map to {dest.name}')
            raw = z.read(n)
            d = load_checkpoint(io.BytesIO(raw), map_location='cpu')
            dest.write_bytes(raw)
            seen_dest[dest.name] = n
            report.append({'seed': seed, 'iteration': d.get('iteration'),
                           'config': d.get('config'),
                           'replay': len(d.get('replay', [])),
                           'file': str(dest.name)})
    (out / 'migration_report.json').write_text(json.dumps(report, indent=2))
    return report