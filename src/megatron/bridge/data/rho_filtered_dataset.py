"""Dataset wrapper that filters sequences based on a pre-computed rho-loss index.

The filter file is a numpy ``.npy`` containing a sorted int64 array of sample
indices to keep.  Only those samples are visible to the dataloader; all others
are skipped.  This achieves **sequence-level** selective training.

Can be combined with ``RhoMaskedGPTDataset`` (token-level masking): apply
sequence filtering first, then token masking on the remaining sequences.
"""

import os

import numpy as np
from torch.utils.data import Dataset


class RhoFilteredGPTDataset(Dataset):
    """Wraps a GPTDataset to expose only the sequences in ``filter_indices``."""

    def __init__(self, inner_dataset: Dataset, filter_path: str):
        self.inner = inner_dataset

        if not os.path.isfile(filter_path):
            raise FileNotFoundError(f"Rho filter file not found: {filter_path}")

        self._indices = np.load(filter_path).astype(np.int64)
        if len(self._indices) == 0:
            raise ValueError(f"Rho filter file is empty: {filter_path}")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self.inner[int(self._indices[idx])]

    @property
    def collate_fn(self):
        return getattr(self.inner, "collate_fn", None)
