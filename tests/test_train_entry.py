import json
from pathlib import Path

import pytest

import train


def test_automatic_cpu_gpu_profiles_share_one_lineage_contract():
    gpu = json.loads(Path(train.GPU_CONFIG).read_text(encoding='utf-8'))
    cpu = json.loads(Path(train.CPU_CONFIG).read_text(encoding='utf-8'))
    train._validate_matched_configs(gpu, cpu)


def test_config_pair_rejects_a_learning_rate_horizon_mismatch():
    common = dict(encoding=3, channels=8, blocks=1, se=True, gumbel=True,
                  gumbel_cap=4, sims=4, fast_sims=2, full_frac=0.5,
                  total_steps=100)
    cpu = dict(common, total_steps=99)
    with pytest.raises(ValueError, match='total_steps'):
        train._validate_matched_configs(common, cpu)
