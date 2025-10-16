#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import json
import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from data.bioparse import VOCAB
from .subgraph_encoder import Subgraph2DGNNEncoder


class FragmentIndex:
    """Lightweight fragment embedding index with incremental updates."""

    def __init__(
        self,
        hidden_size: int,
        *,
        encoder: Optional[Subgraph2DGNNEncoder] = None,
        index_path: Optional[str] = None,
        base_fragments: str = "vocab",
        device: str = "cpu",
        encode_batch_size: int = 256,
        update_every: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.encoder = encoder or Subgraph2DGNNEncoder(hidden_size=hidden_size)
        self.encoder.to(self.device)
        self.index_path = index_path
        self.encode_batch_size = encode_batch_size
        self.update_every = update_every
        self._step = 0

        self.smiles_list: List[str] = []
        self._smiles_to_idx: dict[str, int] = {}
        self.embeddings = torch.empty(0, hidden_size, dtype=torch.float32, device=self.device)
        self._pending_smiles: List[str] = []

        self._file_bytes = 0
        self._index_eof = False
        initial_smiles: List[str] = []
        if base_fragments == "vocab":
            initial_smiles.extend(self._load_vocab_fragments())
        elif isinstance(base_fragments, (list, tuple)):
            initial_smiles.extend(base_fragments)

        self.add_fragments(initial_smiles)

    def _load_vocab_fragments(self) -> List[str]:
        fragments = []
        for symbol, abrv in VOCAB.idx2block:
            if abrv == 'UNK':
                continue
            idx = VOCAB.abrv_to_idx(abrv)
            if idx < len(VOCAB.aa_mask) and VOCAB.aa_mask[idx]:
                # Skip amino-acid residues when focusing on molecules
                continue
            fragments.append(abrv)
        return fragments

    def _encode_smiles(self, smiles: Sequence[str]) -> torch.Tensor:
        if not smiles:
            return torch.empty(0, self.hidden_size, device=self.device)
        embs = []
        with torch.no_grad():
            for start in range(0, len(smiles), self.encode_batch_size):
                batch = smiles[start:start + self.encode_batch_size]
                emb = self.encoder.encode_smiles_list(batch, device=self.device)
                embs.append(emb)
        emb_tensor = torch.cat(embs, dim=0)
        return F.normalize(emb_tensor, dim=-1)

    def add_fragments(self, smiles: Iterable[str]) -> None:
        smiles = list(smiles)
        new_smiles = [s for s in smiles if s and s not in self._smiles_to_idx and s not in self._pending_smiles]
        if not new_smiles:
            return
        start_idx = len(self.smiles_list) + len(self._pending_smiles)
        for i, s in enumerate(new_smiles):
            self._smiles_to_idx[s] = start_idx + i
        self._pending_smiles.extend(new_smiles)

    def maybe_refresh(self) -> None:
        if not self.index_path or not os.path.exists(self.index_path):
            return
        if self.update_every and self._step % self.update_every != 0:
            self._step += 1
            return
        if self._index_eof:
            self._step += 1
            return
        self._load_more_from_index(self.encode_batch_size)
        self._step += 1

    def _load_more_from_index(self, count: int, exclude: Optional[Sequence[str]] = None) -> None:
        if not self.index_path or not os.path.exists(self.index_path) or count <= 0:
            return
        new_smiles: List[str] = []
        try:
            with open(self.index_path, 'r', encoding='utf-8') as handle:
                handle.seek(self._file_bytes)
                while len(new_smiles) < count:
                    line = handle.readline()
                    if not line:
                        self._index_eof = True
                        break
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    frag = rec.get('fragment_smiles') or rec.get('frag')
                    if not frag:
                        continue
                    if exclude and frag in exclude:
                        continue
                    if frag in self._smiles_to_idx or frag in self._pending_smiles:
                        continue
                    new_smiles.append(frag)
                self._file_bytes = handle.tell()
        except Exception:
            return
        if new_smiles:
            self.add_fragments(new_smiles)
            self._flush_pending()

    def ensure_fragments(self, smiles: Sequence[str]) -> None:
        self.add_fragments(smiles)
        self._flush_pending()

    def get_embeddings(self, smiles: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        self.add_fragments(smiles)
        self._flush_pending()
        indices = [self._smiles_to_idx[s] for s in smiles if s in self._smiles_to_idx]
        if not indices:
            return torch.empty(0, self.hidden_size, device=device or self.device)
        emb = self.embeddings[indices]
        if device is not None and emb.device != device:
            emb = emb.to(device)
        return emb

    def search(self, query: torch.Tensor, topk: int = 1) -> Tuple[torch.Tensor, List[List[str]]]:
        self._flush_pending()
        if len(self.embeddings) == 0:
            raise RuntimeError('FragmentIndex is empty')
        query = F.normalize(query, dim=-1)
        sims = torch.matmul(query, self.embeddings.t())
        topk = min(topk, sims.shape[-1])
        scores, indices = torch.topk(sims, k=topk, dim=-1)
        smiles = [[self.smiles_list[idx] for idx in row.tolist()] for row in indices.cpu()]
        return scores, smiles

    def sample_negative_embeddings(
        self,
        k: int,
        *,
        exclude: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        self._flush_pending()
        available = [s for s in self.smiles_list if exclude is None or s not in exclude]
        if (not available or len(available) < k) and self.index_path:
            needed = max(0, k - len(available))
            self._load_more_from_index(max(needed, self.encode_batch_size), exclude)
            available = [s for s in self.smiles_list if exclude is None or s not in exclude]
        if not available:
            return torch.empty(0, self.hidden_size, device=device or self.device), []
        if k >= len(available):
            chosen = available
        else:
            chosen = random.sample(available, k)
        emb = self.get_embeddings(chosen, device=device)
        return emb, chosen

    def sample_random_smiles(self, k: int) -> List[str]:
        import random

        if k <= 0 or not self.smiles_list:
            return []
        if k >= len(self.smiles_list):
            return random.sample(self.smiles_list, len(self.smiles_list))
        return random.sample(self.smiles_list, k)

    def sample_random_smiles(self, k: int) -> List[str]:
        import random

        if not self.smiles_list or k <= 0:
            return []
        if k >= len(self.smiles_list):
            return random.sample(self.smiles_list, len(self.smiles_list))
        return random.sample(self.smiles_list, k)

    def _flush_pending(self) -> None:
        if not self._pending_smiles:
            return
        embeddings = self._encode_smiles(self._pending_smiles)
        start_idx = len(self.smiles_list)
        self.smiles_list.extend(self._pending_smiles)
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self._pending_smiles = []
