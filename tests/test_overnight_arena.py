import json
import sys

import pytest

import tools.overnight_arena as overnight


def test_laptop_dry_run_preflights_before_model_load(monkeypatch, capsys):
    seen = {}

    def preflight(opponents, **_kwargs):
        seen['opponents'] = opponents
        return {'ok': True, 'opponents': [
            {'opponent': opponent, 'command': ['python', f'{opponent}.py'],
             'ok': True, 'requirements': [], 'missing': []}
            for opponent in opponents
        ]}

    monkeypatch.setattr(overnight.fa, 'preflight', preflight)
    monkeypatch.setattr(overnight, '_add_run_requirements', lambda *args: None)
    monkeypatch.setattr(overnight, 'load_checkpoint',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError('checkpoint loaded during dry-run')))
    monkeypatch.setattr(sys, 'argv', [
        'overnight_arena.py', '--net', 'missing.pt', '--preset', 'laptop', '--dry-run'])

    assert overnight.main() == 0
    assert 'vader' not in seen['opponents']
    output = capsys.readouterr().out
    assert 'Selected opponents:' in output
    assert 'python berlioz.py' in output


def test_legacy_record_normalises_missing_fields():
    prior = {'opponent': 'cryer', 'net': 'old.pt', 'games': 4, 'wins': 3,
             'draws': 0, 'losses': 1, 'opp_forfeits': 1, 'avg_plies': 12.5}

    result = overnight._normalise(prior, 'cryer', 'new.pt', 7)

    assert result['rounds'] == []
    assert result['plies_sum'] == 50.0
    assert result['bridge_error'] == 1
    assert result['honest_game'] == 3


def test_overnight_persists_callback_games_without_double_counting(monkeypatch, tmp_path):
    ticks = iter(range(20))
    monkeypatch.setattr(overnight.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(overnight.fa, 'preflight', lambda opponents, **kwargs: {
        'ok': True, 'opponents': [
            {'opponent': opponent, 'command': ['python', 'bridge.py'], 'ok': True,
             'requirements': [], 'missing': []} for opponent in opponents]})
    monkeypatch.setattr(overnight, '_add_run_requirements', lambda *args: None)
    monkeypatch.setattr(overnight, 'resolve_device', lambda device: 'cpu')
    monkeypatch.setattr(overnight, 'configure_threads', lambda threads: None)
    monkeypatch.setattr(overnight, 'load_checkpoint', lambda *args, **kwargs: {'generation': 9})
    monkeypatch.setattr(overnight, 'net_from_checkpoint', lambda checkpoint, device: object())

    def result(games):
        return {'games': games, 'wins': games, 'draws': 0, 'losses': 0,
                'opp_forfeits': 0, 'avg_plies': 10.0, 'seed': 100000,
                'honest_game': games, 'explicit_forfeit': 0, 'illegal_move': 0,
                'timeout': 0, 'bridge_error': 0}

    def play(*args, **kwargs):
        callback = kwargs['on_game_complete']
        callback(result(1))
        callback(result(1))
        return result(2)

    monkeypatch.setattr(overnight.fa, 'play', play)
    monkeypatch.setattr(sys, 'argv', [
        'overnight_arena.py', '--net', 'net.pt', '--hours', '0.001', '--batch', '2',
        '--only', 'vader', '--include-vader', '--outdir', str(tmp_path)])

    assert overnight.main() == 0
    record = json.loads((tmp_path / 'vader.json').read_text(encoding='utf-8'))
    summary = json.loads((tmp_path / 'summary.json').read_text(encoding='utf-8'))
    assert record['games'] == 2
    assert record['wins'] == 2
    assert len(record['rounds']) == 2
    assert summary['total_games'] == 2


def test_profiles_include_requested_laptop_and_cuda_budgets():
    laptop = overnight.PROFILES['cpu-laptop']
    assert laptop['device'] == 'cpu'
    assert laptop['threads'] == 6
    assert laptop['sims'] == 16
    assert laptop['batch'] == 1
    assert 'i5-1155G7' in laptop['host']
    assert {'cpu-smoke', 'cuda-8gb', 'cuda-16gb'} <= set(overnight.PROFILES)


def test_summary_reports_honest_rate_and_wilson_interval():
    record = overnight._blank('dimi', 'net.pt', 1)
    record.update(games=10, wins=7, draws=1, losses=2, opp_forfeits=2,
                  honest_game=8, explicit_forfeit=2, win_rate=0.75)

    row = overnight._summary({'dimi': record}, 'net.pt', 1, 'later')['leaderboard'][0]

    assert row['honest_games'] == 8
    assert row['honest_win_rate'] == pytest.approx(5.5 / 8, abs=0.0001)
    low, high = row['honest_win_rate_wilson95']
    assert 0 <= low < row['honest_win_rate'] < high <= 1


def test_merge_resets_or_increments_technical_streak():
    record = overnight._blank('dimi', 'net.pt', 1)
    technical = {'games': 1, 'wins': 1, 'draws': 0, 'losses': 0,
                 'opp_forfeits': 1, 'avg_plies': 0, 'seed': 1,
                 'honest_game': 0, 'explicit_forfeit': 0, 'illegal_move': 1,
                 'timeout': 0, 'bridge_error': 0}
    honest = dict(technical, opp_forfeits=0, honest_game=1, illegal_move=0)

    overnight._merge(record, technical, 1)
    overnight._merge(record, technical, 1)
    assert record['technical_streak'] == 2
    overnight._merge(record, honest, 1)
    assert record['technical_streak'] == 0


def test_resume_rejects_results_from_a_different_checkpoint(monkeypatch, tmp_path):
    (tmp_path / 'dimi.json').write_text(json.dumps({
        'opponent': 'dimi', 'net': 'old.pt', 'generation': 1, 'rounds': []}))
    monkeypatch.setattr(overnight.fa, 'OPPONENTS', {'dimi': ['python', 'bridge.py']})
    monkeypatch.setattr(overnight.fa, 'preflight', lambda *args, **kwargs: {
        'ok': True, 'opponents': [], 'host_requirements': []})
    monkeypatch.setattr(overnight, '_add_run_requirements', lambda *args: None)
    monkeypatch.setattr(overnight, 'resolve_device', lambda device: 'cpu')
    monkeypatch.setattr(overnight, 'configure_threads', lambda threads: None)
    monkeypatch.setattr(overnight, 'load_checkpoint', lambda *args, **kwargs: {'generation': 2})
    monkeypatch.setattr(overnight, 'net_from_checkpoint', lambda *args: object())
    monkeypatch.setattr(sys, 'argv', [
        'overnight_arena.py', '--net', 'new.pt', '--only', 'dimi',
        '--outdir', str(tmp_path)])

    with pytest.raises(SystemExit):
        overnight.main()
