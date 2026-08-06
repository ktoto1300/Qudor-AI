"""Policy-value network.

The body is a pre-activation-style ResNet with optional squeeze-excitation. SE matters
more here than in most board games: Quoridor's value is dominated by board-wide facts
(who wins the shortest-path race, how many walls each side has left) that a stack of 3x3
convs propagates only one cell per layer. An SE gate hands every block the pooled global
signal directly, which is why KataGo and Leela put one in the same place.

Everything about the architecture is recoverable from the weights themselves - channel
count, block count, plane count and whether SE is present - so a checkpoint never depends
on its config dict being accurate.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .core.engine import ACTION_SIZE
from .core.encoding import PLANES, PLANES_BY_VERSION


class SE(nn.Module):
    """Channel gate driven by global average pooling."""

    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(8, c // r)
        self.fc = nn.Sequential(nn.Linear(c, hidden), nn.SiLU(), nn.Linear(hidden, c))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3))))[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, c, se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        # Registered only when enabled so pre-SE checkpoints keep loading unchanged.
        self.se = SE(c) if se else None

    def forward(self, x):
        y = self.net(x)
        if self.se is not None:
            y = self.se(y)
        return F.silu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64, blocks=6, planes=PLANES, se=False):
        super().__init__()
        self.channels, self.blocks, self.planes, self.se = channels, blocks, planes, se
        self.stem = nn.Sequential(nn.Conv2d(planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.SiLU())
        self.body = nn.Sequential(*[ResBlock(channels, se) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 8, 1), nn.SiLU(), nn.Flatten(),
                                    nn.Linear(8 * 81, ACTION_SIZE))
        self.value = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.SiLU(), nn.Flatten(),
                                   nn.Linear(4 * 81, 128), nn.SiLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        z = self.body(self.stem(x))
        return self.policy(z), self.value(z).squeeze(1)


def masked_policy(logits, masks):
    return logits.masked_fill(~masks, -1e9)


def planes_of(state_dict):
    """Input-plane count a checkpoint was trained with, read off the stem conv."""
    w = state_dict.get('stem.0.weight')
    return int(w.shape[1]) if w is not None else PLANES


def arch_of(state_dict):
    """Recover (channels, blocks, planes, se) from the weights alone.

    Config dicts drift - they get copied between runs, hand-edited, or written by an older
    version - so the weights are the only trustworthy description of the shape.
    """
    w = state_dict['stem.0.weight']
    channels, planes = int(w.shape[0]), int(w.shape[1])
    blocks = 1 + max((int(k.split('.')[1]) for k in state_dict if k.startswith('body.')), default=-1)
    se = any(k.startswith('body.0.se.') for k in state_dict)
    return channels, blocks, planes, se


class IncompatibleCheckpoint(ValueError):
    """Checkpoint holds weights for an architecture this PolicyValueNet cannot represent."""


def net_from_checkpoint(d, device=None):
    """Rebuild the exact network a checkpoint holds, including its encoder version.

    Older checkpoints predate the versioned encoder and carry no 'encoding' key; their
    plane count is read from the weights, so v1 nets keep loading unchanged.

    Raises IncompatibleCheckpoint for the pre-ResBlock generation (7-plane stem, plain
    Sequential body, narrower heads) found under legacy/ - those weights have no mapping
    onto the current architecture. Their replay buffers are still usable via legacy_pretrain.
    """
    sd = d['model']
    channels, blocks, planes, se = arch_of(sd)
    if planes not in set(PLANES_BY_VERSION.values()) or not any(k.startswith('body.0.net.') for k in sd):
        raise IncompatibleCheckpoint(
            f'checkpoint uses the pre-ResBlock architecture ({planes}-plane stem); '
            'its weights cannot be loaded into PolicyValueNet. Use it as legacy_pretrain data instead.')
    net = PolicyValueNet(channels, blocks, planes, se)
    if device is not None:
        net = net.to(device)
    net.load_state_dict(sd)
    net.eval()
    return net
