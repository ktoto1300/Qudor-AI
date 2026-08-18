"""migrate.extract: per-seed latest.pt extraction from a training zip."""
import io
import zipfile

import pytest
import torch

from quoridor_ai import migrate
from quoridor_ai.core.engine import ACTION_SIZE

_CK = {'iteration': 10, 'model': {'w': torch.zeros(1)},
       'config': {'channels': 8, 'blocks': 1},
       'replay': [(torch.zeros(1, 9, 9), torch.zeros(ACTION_SIZE), 0.0)] * 2}


def _zip(tmp_path, entries):
    path = tmp_path / 'night.zip'
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, obj in entries:
            buf = io.BytesIO()
            torch.save(obj, buf)
            z.writestr(name, buf.getvalue())
    path.write_bytes(raw.getvalue())
    return path


def test_extracts_one_file_per_seed(tmp_path):
    z = _zip(tmp_path, [('seed_11/latest.pt', _CK), ('seed_22/latest.pt', _CK)])
    out = tmp_path / 'out'
    report = migrate.inspect_and_extract(z, out)
    assert {r['seed'] for r in report} == {'seed_11', 'seed_22'}
    assert (out / 'seed_11_legacy.pt').is_file()
    assert (out / 'seed_22_legacy.pt').is_file()
    d = torch.load(out / 'seed_11_legacy.pt', map_location='cpu', weights_only=False)
    assert d['iteration'] == 10 and len(d['replay']) == 2
    assert (out / 'migration_report.json').is_file()
    assert report[0]['replay'] == 2


def test_top_level_latest_pt_does_not_raise(tmp_path):
    z = _zip(tmp_path, [('latest.pt', _CK)])
    report = migrate.inspect_and_extract(z, tmp_path / 'out')
    assert report[0]['seed'] == 'unknown'


def test_duplicate_seed_inside_one_zip_is_refused(tmp_path):
    z = _zip(tmp_path, [('seed_11/latest.pt', _CK), ('seed_11/latest.pt', _CK)])
    with pytest.raises(FileExistsError, match='seed_11'):
        migrate.inspect_and_extract(z, tmp_path / 'out')