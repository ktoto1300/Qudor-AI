from quoridor_ai import paths

import pytest


def test_repo_root_defaults_to_the_checkout():
    assert paths.repo_root() == paths.REPO_ROOT


def test_repo_root_honours_qudor_repo(monkeypatch, tmp_path):
    monkeypatch.setenv('QUDOR_REPO', str(tmp_path))
    assert paths.repo_root() == tmp_path


def test_bots_dir_defaults_to_a_sibling_of_the_repo():
    assert paths.bots_dir() == paths.REPO_ROOT.parent / 'bots'


def test_bots_dir_honours_bots_dir(monkeypatch, tmp_path):
    monkeypatch.setenv('BOTS_DIR', str(tmp_path))
    assert paths.bots_dir() == tmp_path


def test_bots_dir_require_missing_raises(monkeypatch, tmp_path):
    missing = tmp_path / 'no-such-bots'
    monkeypatch.setenv('BOTS_DIR', str(missing))
    with pytest.raises(FileNotFoundError, match='foreign-bot directory not found'):
        paths.bots_dir(require=True)


def test_bot_repo_require_missing_names_the_engine(monkeypatch, tmp_path):
    monkeypatch.setenv('BOTS_DIR', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='no_such_engine|no-such-engine'):
        paths.bot_repo('no_such_engine', require=True)
    monkeypatch.setenv('BOTS_DIR', str(tmp_path / 'gone'))
    with pytest.raises(FileNotFoundError, match='foreign-bot directory not found'):
        paths.bot_repo('anything', require=True)


def test_bot_repo_without_require_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv('BOTS_DIR', str(tmp_path))
    assert paths.bot_repo('missing') == tmp_path / 'missing'