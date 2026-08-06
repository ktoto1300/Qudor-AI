"""Safe PyTorch checkpoint loader with restricted pickle for numpy arrays."""
import torch
import torch.serialization as ts
import numpy as np

# Allowlist: only numpy reconstruction primitives needed for checkpoint replay buffers.
# Excludes arbitrary code execution while permitting ndarray/dtype unpickling.
try:  # numpy>=2 moved the private namespace; both spellings unpickle the same payloads.
    _reconstruct = np._core.multiarray._reconstruct
except AttributeError:
    _reconstruct = np.core.multiarray._reconstruct

_SAFE_GLOBALS = [
    _reconstruct,
    np.ndarray,
    np.dtype,
]
try:
    _SAFE_GLOBALS.append(np.dtypes.Float16DType)
    _SAFE_GLOBALS.append(np.dtypes.Float32DType)
    _SAFE_GLOBALS.append(np.dtypes.Int64DType)
except AttributeError:
    pass  # Older numpy versions


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint with weights_only=True and a restricted numpy allowlist.

    Args:
        path: Path to a .pt checkpoint file, or an open binary file-like object.
        map_location: Device to map tensors to (default: 'cpu').

    Returns:
        Checkpoint dictionary with keys: model, optimizer, config, iteration, replay, etc.

    Raises:
        pickle.UnpicklingError: If the checkpoint contains disallowed pickle operations.
    """
    with ts.safe_globals(_SAFE_GLOBALS):
        return torch.load(path, map_location=map_location, weights_only=True)
