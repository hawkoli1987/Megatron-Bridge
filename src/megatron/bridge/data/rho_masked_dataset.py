"""Dataset wrapper that adds a pre-computed rho-loss binary mask to each sample.

The mask memmap is sample-indexed: for sample ``idx`` the mask values are at
``memmap[idx * seq_length : (idx + 1) * seq_length]``.  Both annotation and
training must use identical dataset construction parameters (same data, same
``seq_length``, same ``num_samples``, same seed) so that sample ``idx`` always
maps to the same tokens.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RhoMaskedGPTDataset(Dataset):
    """Wraps a GPTDataset to augment each sample with ``rho_mask``."""

    def __init__(self, inner_dataset: Dataset, mask_path: str, seq_length: int):
        self.inner = inner_dataset
        self.seq_length = seq_length

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Rho mask file not found: {mask_path}")

        total_entries = os.path.getsize(mask_path)  # uint8 → 1 byte each
        self._mask = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(total_entries,))
        self._num_annotated_samples = total_entries // seq_length

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int):
        sample = self.inner[idx]

        if idx < self._num_annotated_samples:
            start = idx * self.seq_length
            end = start + self.seq_length
            rho_mask = torch.from_numpy(self._mask[start:end].copy()).to(torch.float32)
        else:
            # Out-of-range: keep all tokens (no masking for unannotated samples)
            rho_mask = torch.ones(self.seq_length, dtype=torch.float32)

        sample["rho_mask"] = rho_mask
        return sample

    @property
    def collate_fn(self):
        return getattr(self.inner, "collate_fn", None)
