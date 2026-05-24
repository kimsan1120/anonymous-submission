from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from torch.utils.data import Sampler


@dataclass(frozen=True)
class _StratumKey:
    label: int
    category: str


class _CyclingPool:
    def __init__(self, indices: Sequence[int], rng: random.Random):
        self._base = list(indices)
        self._rng = rng
        self._buffer: List[int] = []
        self._refill()

    def _refill(self) -> None:
        self._buffer = self._base.copy()
        self._rng.shuffle(self._buffer)

    def draw(self) -> int:
        if not self._buffer:
            self._refill()
        return self._buffer.pop()


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Custom batch sampler balancing (label, category) strata in every batch.

    Sampling is with replacement across an epoch boundary (via cycling pools),
    but without replacement inside each local pool refill. This keeps batches
    stable for contrastive positives while avoiding premature pool exhaustion.
    """

    def __init__(
        self,
        labels: Iterable[int],
        categories: Iterable[str],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 10,
        max_batches: int | None = None,
    ):
        super().__init__(None)
        self.labels = [int(v) for v in labels]
        self.categories = [str(v) for v in categories]
        if len(self.labels) != len(self.categories):
            raise ValueError("labels and categories length must match")
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1")

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(self.labels)

        strata: Dict[_StratumKey, List[int]] = defaultdict(list)
        for idx, (label, category) in enumerate(zip(self.labels, self.categories)):
            strata[_StratumKey(label=label, category=category)].append(idx)
        self.strata = {k: v for k, v in strata.items() if v}
        if not self.strata:
            raise ValueError("no samples available for BalancedBatchSampler")

        if max_batches is not None:
            self.num_batches = int(max_batches)
        elif self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.num_samples / self.batch_size)

        if self.num_batches <= 0:
            raise ValueError("num_batches resolved to 0; adjust batch_size/drop_last")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _alloc_quota(
        self,
        keys: List[_StratumKey],
        batch_size: int,
        rng: random.Random,
    ) -> Dict[_StratumKey, int]:
        n_keys = len(keys)
        if n_keys <= 0:
            raise ValueError("keys must not be empty")

        alloc: Dict[_StratumKey, int] = {k: 0 for k in keys}

        if n_keys >= batch_size:
            chosen = rng.sample(keys, k=batch_size)
            for key in chosen:
                alloc[key] += 1
            return alloc

        # one from each stratum first
        for key in keys:
            alloc[key] += 1
        remaining = batch_size - n_keys

        # distribute remainder evenly (plus random tie-break)
        base = remaining // n_keys
        extra = remaining % n_keys
        for key in keys:
            alloc[key] += base
        if extra > 0:
            picked = rng.sample(keys, k=extra)
            for key in picked:
                alloc[key] += 1
        return alloc

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        keys = sorted(self.strata.keys(), key=lambda x: (x.label, x.category))
        pools = {k: _CyclingPool(self.strata[k], rng=rng) for k in keys}

        for _ in range(self.num_batches):
            alloc = self._alloc_quota(keys=keys, batch_size=self.batch_size, rng=rng)
            batch: List[int] = []
            for key in keys:
                for _ in range(alloc[key]):
                    batch.append(pools[key].draw())
            rng.shuffle(batch)

            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch
