import numpy as np
import pytest
import torch

from quoridor_ai import legacy_pretrain as lp
from quoridor_ai.core.encoding import encode_v1
from quoridor_ai.core.engine import State, apply_unchecked

LEGACY_PLANES = 7


def _legacy(x11):
    """v1 11-plane sample -> legacy 7-plane sample (planes 5/6 swapped)."""
    x = x11.copy()[:LEGACY_PLANES]
    x[5], x[6] = x[6], x[5].copy()
    return x


def _midgame():
    s = State()
    for a in (81, 145 + 3 * 8 + 3, 9, 82, 145 + 5 * 8 + 4, 99, 18):
        s = apply_unchecked(s, a)
    return s


def test_lift_sample_round_trips_a_real_position():
    s = _midgame()
    got = lp.lift_sample(_legacy(encode_v1(s)))
    assert got is not None
    assert (got.p0, got.p1, got.walls0, got.walls1,
            got.h, got.v, got.player) == (s.p0, s.p1, s.walls0, s.walls1,
                                          s.h, s.v, s.player)


def test_lift_sample_refuses_wrong_plane_count():
    s = _midgame()
    x = _legacy(encode_v1(s))
    assert lp.lift_sample(x) is not None
    assert lp.lift_sample(x[:6]) is None
    assert lp.lift_sample(np.zeros((7, 9, 8), np.float32)) is None


def test_lift_sample_refuses_stray_wall_pixels():
    s = _midgame()
    x = _legacy(encode_v1(s))
    x[5, 0, 0] = 1.0                       # an unpaired cell in the v-wall plane
    assert lp.lift_sample(x) is None


def test_lift_sample_refuses_non_constant_count_planes():
    s = _midgame()
    x = _legacy(encode_v1(s))
    x[3, 4, 4] = 0.5
    assert lp.lift_sample(x) is None


def test_lift_sample_refuses_bad_pawn_planes():
    s = _midgame()
    x = _legacy(encode_v1(s))
    x[0, 0, 0] = 1.0                       # two pawn markers for player 0
    assert lp.lift_sample(x) is None


def test_lift_sample_parses_chained_walls():
    # Adjacent same-orientation walls (legal under the old ruleset) merge into one
    # painted run of three cells; both walls must still be recovered.
    s = State(p0=76, p1=4, walls0=8, walls1=10, h=0, v=0b11, player=0, ply=1)
    x = np.zeros((LEGACY_PLANES, 9, 9), np.float32)
    r, c = divmod(s.p0, 9); x[0, r, c] = 1
    r, c = divmod(s.p1, 9); x[1, r, c] = 1
    x[2].fill(s.player); x[3].fill(s.walls0 / 10); x[4].fill(s.walls1 / 10)
    for i in (0, 1):                       # v walls at slots (0,0) and (1,0)
        x[5, i, 0] = x[5, i + 1, 0] = 1
    got = lp.lift_sample(x)
    assert got is not None
    assert got.v == (1 | (1 << 8)) and got.h == 0   # slots (0,0) and (1,0)


def test_load_legacy_lifts_and_skips(tmp_path, capsys):
    good = _midgame()
    x = _legacy(encode_v1(good))
    bad = _legacy(encode_v1(_midgame()))
    bad[5, 0, 0] = 1.0                     # stray pixel: must be skipped
    replay = [(x, np.full(209, 1 / 209, np.float32), 0.0),
              (bad, np.full(209, 1 / 209, np.float32), 0.0)]
    torch.save({'replay': replay}, tmp_path / 'seed_1_legacy.pt')
    out = list(lp.load_legacy(tmp_path))
    assert len(out) == 1
    y, pi, z = out[0]
    want = encode_v1(good)
    assert not y[7].any()                     # ply is unrecoverable; stays zero
    assert np.array_equal(y[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10]],
                          want[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10]])
    assert np.isclose(pi, 1 / 209).all() and z == 0.0
    assert '1 usable samples kept, 1 skipped' in capsys.readouterr().out


def test_load_legacy_refuses_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match='no usable legacy replay'):
        list(lp.load_legacy(tmp_path))


@pytest.mark.integration
def test_run_trains_policy_only_when_all_z_are_zero(tmp_path, capsys):
    s = _midgame()
    x = _legacy(encode_v1(s))
    pi = np.full(209, 1 / 209, np.float32)
    torch.save({'replay': [(x, pi, 0.0)] * 6},
               tmp_path / 'seed_9_legacy.pt')
    out = tmp_path / 'out' / 'pretrained.pt'
    lp.run(tmp_path, out, channels=8, blocks=1, epochs=1, batch=4, seed=1)
    assert 'WARN every legacy value target is 0.0' in capsys.readouterr().out
    d = torch.load(out, map_location='cpu', weights_only=False)
    assert d['encoding'] == 1 and d['legacy_samples'] == 6
    assert d['channels'] == 8 and d['blocks'] == 1


@pytest.mark.integration
def test_run_keeps_the_value_term_when_z_varies(tmp_path, capsys):
    s = _midgame()
    x = _legacy(encode_v1(s))
    pi = np.full(209, 1 / 209, np.float32)
    replay = [(x, pi, 1.0)] * 3 + [(x, pi, -1.0)] * 3
    torch.save({'replay': replay}, tmp_path / 'seed_9_legacy.pt')
    out = tmp_path / 'out' / 'pretrained.pt'
    lp.run(tmp_path, out, channels=8, blocks=1, epochs=1, batch=4, seed=1)
    assert 'WARN every legacy value target is 0.0' not in capsys.readouterr().out
    assert (tmp_path / 'out' / 'pretrained.pt').is_file()