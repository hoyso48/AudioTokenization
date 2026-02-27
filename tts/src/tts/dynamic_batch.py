from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Sequence

from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        max_samples: int,
        bucket_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1337,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")

        self.lengths = [max(1, int(x)) for x in lengths]
        self.max_tokens = int(max_tokens)
        self.max_samples = int(max_samples)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        self._cached_len = self._estimate_local_num_batches()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _ordered_indices(self) -> List[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
            return indices

        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(indices)

        ordered: List[int] = []
        for start in range(0, len(indices), self.bucket_size):
            block = indices[start : start + self.bucket_size]
            block.sort(key=lambda idx: self.lengths[idx], reverse=True)
            ordered.extend(block)
        return ordered

    def _pack_batches(self, indices: Iterable[int], shuffle_batches: bool) -> List[List[int]]:
        batches: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0

        for idx in indices:
            item_tokens = self.lengths[idx]

            if item_tokens > self.max_tokens:
                if current:
                    if not (self.drop_last and len(current) < self.max_samples):
                        batches.append(current)
                    current = []
                    current_tokens = 0
                if not self.drop_last:
                    batches.append([idx])
                continue

            next_samples = len(current) + 1
            next_tokens = current_tokens + item_tokens
            overflow = (next_samples > self.max_samples) or (next_tokens > self.max_tokens)

            if overflow and current:
                if not (self.drop_last and len(current) < self.max_samples):
                    batches.append(current)
                current = [idx]
                current_tokens = item_tokens
            else:
                current.append(idx)
                current_tokens = next_tokens

        if current and not (self.drop_last and len(current) < self.max_samples):
            batches.append(current)

        if shuffle_batches and self.shuffle:
            rng = random.Random(self.seed + self._epoch + 99991)
            rng.shuffle(batches)

        return batches

    def _estimate_local_num_batches(self) -> int:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
        global_batches = self._pack_batches(indices, shuffle_batches=False)
        return len(global_batches)

    def __iter__(self) -> Iterator[List[int]]:
        indices = self._ordered_indices()
        global_batches = self._pack_batches(indices, shuffle_batches=True)
        for batch in global_batches:
            yield batch
        self._epoch += 1

    def __len__(self) -> int:
        return self._cached_len
