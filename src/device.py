"""
Device selection utility.

Helpers
-------
  get_device : Return the best available PyTorch accelerator as a string.
"""

import torch


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Select the best available accelerator in priority order: CUDA, MPS, CPU.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
