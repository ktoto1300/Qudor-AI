"""One entry point for training, on a GPU or on two CPU cores.

The reason this exists: the launch line used to carry the config path by hand, and a config
edited in one Drive folder while the notebook read another cost ~200 iterations of training
against settings nobody intended. So this picks the config from the hardware, resolves it to an
absolute path next to this file, and prints the settings it actually loaded before doing any
work. If the printout disagrees with what you edited, you are looking at the wrong folder - and
you find that out in the first second instead of two hundred iterations later.

    python train.py                      # detects the hardware, picks the config
    python train.py --output DIR         # somewhere other than the default runs/ dir
    python train.py --force-cpu          # CPU profile even on a machine that has a GPU
    python train.py --dry-run            # print the plan and exit, touching nothing

Everything past `--` goes to quoridor_ai.az_train untouched, so the escape hatch is intact:

    python train.py -- --config configs/az_smoke.json --output .smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# These two are a matched pair - same encoder, same 128x10 net, same Gumbel search - so a run
# can move between a GPU session and a CPU one and stay one lineage. colab_az_15gb.json is the
# older PUCT config and is deliberately not reachable from here: pointing it at this run would
# silently swap the search method halfway through.
GPU_CONFIG = ROOT / 'configs' / 'colab_az_gumbel.json'
CPU_CONFIG = ROOT / 'configs' / 'colab_az_cpu.json'
DEFAULT_OUTPUT = ROOT / 'runs' / 'Checkpoints'


def detect() -> tuple[str, str]:
    """(device, human-readable reason). Importing torch here keeps --help instant."""
    try:
        import torch
    except ImportError:
        return 'cpu', 'torch не установлен'
    if not torch.cuda.is_available():
        return 'cpu', 'видеокарта не найдена'
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return 'cuda', f'{name}, {gb:.0f} ГБ'


def describe(cfg: dict) -> list[tuple[str, str]]:
    total = cfg.get('total_steps') or cfg['iterations'] * cfg['steps']
    return [
        ('партий за раз', str(cfg['games'])),
        ('симуляций на ход', f"{cfg['sims']}" + (' (Gumbel)' if cfg.get('gumbel') else ' (PUCT)')),
        ('сеть', f"{cfg['channels']}x{cfg['blocks']}"),
        ('шагов обучения', f"{cfg['steps']} по {cfg['batch']}"),
        ('до итерации', str(cfg['iterations'])),
        ('горизонт', f'{total} шагов'),
        ('экзамен', f"раз в {cfg['gate_every']} итераций, "
                    f"{cfg['gate_games']} партий, порог {cfg['gate_threshold']}"),
    ]


def main() -> int:
    p = argparse.ArgumentParser(
        description='Запуск обучения. Сам определяет, есть ли видеокарта.',
        epilog='Аргументы после -- уходят в quoridor_ai.az_train как есть.')
    p.add_argument('--output', help=f'куда писать (по умолчанию {DEFAULT_OUTPUT})')
    p.add_argument('--force-cpu', action='store_true', help='режим CPU даже при наличии GPU')
    p.add_argument('--dry-run', action='store_true', help='показать план и выйти')
    a, extra = p.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]

    # A hand-written --config beats detection: the caller has said what they want, and silently
    # overriding it would be the exact failure this script exists to prevent.
    if '--config' in extra:
        argv = extra
        print('беру конфиг из командной строки, определение железа пропускаю', flush=True)
    else:
        device, reason = ('cpu', 'принудительно, флаг --force-cpu') if a.force_cpu else detect()
        cfg_path = GPU_CONFIG if device == 'cuda' else CPU_CONFIG
        if not cfg_path.exists():
            print(f'ОШИБКА: нет файла {cfg_path}', file=sys.stderr)
            return 2
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        out = Path(a.output).resolve() if a.output else DEFAULT_OUTPUT

        print()
        print(f'  железо   {"ВИДЕОКАРТА" if device == "cuda" else "ПРОЦЕССОР"}  ({reason})')
        print(f'  конфиг   {cfg_path}')
        print(f'  запись   {out}')
        for k, v in describe(cfg):
            print(f'  {k:16} {v}')
        print(flush=True)

        argv = ['--config', str(cfg_path), '--output', str(out), *extra]

    if a.dry_run:
        print('--dry-run: ничего не запускаю')
        return 0

    from quoridor_ai import az_train
    sys.argv = ['az_train', *argv]
    az_train.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
