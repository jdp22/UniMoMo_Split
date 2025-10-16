import json
import os
import random
from typing import Callable, Optional, Sequence, List
from collections import deque
from tqdm import tqdm

import numpy as np
import torch
import sympy

from utils import register as R
from data.bioparse import VOCAB
from data.bioparse.const import aa_smiles, aas

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from rdkit import Chem
from rdkit.Chem.rdchem import KekulizeException

from scripts.chem.partition_demo import (
    random_subgraph_partition,
    fragment_smiles_for_partitions,
)
from utils.subgraph_storage import _heavy_atom_count, SubgraphStorage


class _DynamicFragmentSampler:
    def __init__(self, opt: Optional[dict]):
        if opt is None:
            self.enabled = False
            return
        self.enabled = True
        self.opt = opt
        self.random = random.Random(opt.get('seed', None))
        self.num_parts = opt.get('num_parts')
        self.min_parts = opt.get('min_parts', 1)
        self.max_parts = opt.get('max_parts', max(self.min_parts, 1))
        self.storage_path = opt.get('storage_path')
        self.min_heavy_atoms = opt.get('min_heavy_atoms', 0)
        self.store_metadata = opt.get('store_metadata', True)
        self.log_samples = opt.get('log_samples', False)
        self.log_every = max(1, opt.get('log_every', 1))
        # Optional: record each sampled partition so we can audit fragment coverage later.
        self.partition_log_path = opt.get('log_path')
        self._sample_counter = 0
        self._seen_fragments: set[str] = set()
        self._storage_initialized = False
        self._stored_bytes = 0
        if self.storage_path is not None and os.path.exists(self.storage_path):
            self._warmup_storage()

    def _warmup_storage(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    frag = rec.get('fragment_smiles')
                    if frag is not None:
                        self._seen_fragments.add(frag)
            self._stored_bytes = os.path.getsize(self.storage_path)
        except FileNotFoundError:
            self._stored_bytes = 0
        self._storage_initialized = True

    def _choose_num_parts(self, mol: Chem.Mol) -> int:
        if isinstance(self.num_parts, int):
            return max(1, self.num_parts)
        if isinstance(self.num_parts, (list, tuple)) and len(self.num_parts):
            return max(1, self.random.choice(self.num_parts))
        if self.min_parts >= self.max_parts:
            return max(1, self.min_parts)
        return max(1, self.random.randint(self.min_parts, self.max_parts))

    def _store_fragments(self, mol: Chem.Mol, partitions, fragments, metadata):
        if self.storage_path is None:
            return
        if fcntl is None:
            # On non-posix platforms fall back to in-memory dedup only
            lock = None
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        records = []
        for atom_indices, frag in zip(partitions, fragments):
            if frag in self._seen_fragments:
                continue
            heavy = _heavy_atom_count(mol, atom_indices)
            if heavy < self.min_heavy_atoms:
                continue
            rec = {
                'source_smiles': Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                'fragment_smiles': frag,
                'atom_indices': list(map(int, atom_indices)),
                'heavy_atom_count': heavy,
            }
            if self.store_metadata and metadata is not None:
                rec['metadata'] = metadata
            records.append(rec)
        if not records:
            return
        if not self._storage_initialized:
            self._warmup_storage()
        with open(self.storage_path, 'a+', encoding='utf-8') as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(self._stored_bytes)
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    frag = rec.get('fragment_smiles')
                    if frag is not None:
                        self._seen_fragments.add(frag)
                self._stored_bytes = handle.tell()
                handle.seek(0, os.SEEK_END)
                for rec in records:
                    frag = rec['fragment_smiles']
                    if frag in self._seen_fragments:
                        continue
                    handle.write(json.dumps(rec, ensure_ascii=False))
                    handle.write('\n')
                    self._seen_fragments.add(frag)
                self._stored_bytes = handle.tell()
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _log_partitions(self, canonical: str, partitions, fragments, metadata):
        if self.partition_log_path is None:
            return
        try:
            os.makedirs(os.path.dirname(self.partition_log_path), exist_ok=True)
        except Exception:
            pass
        handle = None
        record = {
            'id': metadata.get('id') if isinstance(metadata, dict) else None,
            'source_smiles': canonical,
            'fragments': fragments,
            # Keep atom indices to verify fragments tile the ligand.
            'partitions': [list(map(int, part)) for part in partitions] if partitions else []
        }
        try:
            handle = open(self.partition_log_path, 'a', encoding='utf-8')
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            pass
        finally:
            if handle is not None:
                if fcntl is not None:
                    try:
                        handle.flush()
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    except Exception:
                        pass
                handle.close()

    def sample(self, smiles: Optional[str], metadata):
        if not self.enabled or not smiles:
            return None, []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, []
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except KekulizeException:
            try:
                Chem.SetAromaticity(mol)
            except Exception:
                return None, []
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        attempts = 0
        partitions = None
        while attempts < 5:
            attempts += 1
            num_parts = self._choose_num_parts(mol)
            try:
                partitions = random_subgraph_partition(
                    mol,
                    num_parts,
                    seed=self.random.randint(0, 10**9),
                )
                break
            except ValueError:
                # fall back by reducing num_parts
                if num_parts > 1:
                    self.max_parts = max(1, num_parts - 1)
                    continue
        if partitions is None or not partitions:
            partitions = [list(range(mol.GetNumAtoms()))]
        fragments = fragment_smiles_for_partitions(mol, partitions)
        if not fragments:
            fragments = [canonical] if canonical else []
        if not fragments:
            return None, []
        # Logging now happens after filtering inside DynamicBlockWrapper to avoid
        # duplicating records for pre-filtered candidate partitions.
        self._store_fragments(mol, partitions, fragments, metadata)
        self._sample_counter += 1
        if self.log_samples and self._sample_counter % self.log_every == 0:
            print(f"[FragmentSampler] SMILES={smiles} canonical={canonical} fragments={fragments}")
        return canonical, fragments


@R.register('MixDatasetWrapper')
class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, datasets, collate_fn: Callable=None, weights=None) -> None:
        super().__init__()
        self.datasets = [R.recur_construct(dataset) for dataset in datasets]
        self.cum_len = []
        self.total_len = 0
        for dataset in self.datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        # Mixed datasets may yield heterogenous sample structures (e.g., dict vs tuple).
        # Use a robust mixed collate that dispatches to the appropriate sub‑dataset
        # collate_fn based on minibatch shape, then merges results.
        self.collate_fn = self._mixed_collate if collate_fn is None else collate_fn
        if weights is not None: assert len(weights) == len(datasets)
        self.weights = weights
        self.dynamic_idx = []
        self.update_epoch()

    def _get_dataset_and_idx(self, idx: int):
        assert idx < self.total_len
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i], idx - last_cum_len
            last_cum_len = cum_len
        return None, None  # this is not possible

    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()
        if self.weights is None:
            self.dynamic_idx = [i for i in range(self.total_len)]
            np.random.shuffle(self.dynamic_idx)
        else:
            self.dynamic_idx = []
            start_idx = 0
            for i, (w, dataset) in enumerate(zip(self.weights, self.datasets)):
                add_len, end_idx = int(len(dataset) * w), self.cum_len[i]
                self.dynamic_idx.extend(np.random.choice(
                    list(range(start_idx, end_idx)),
                    size=add_len, replace=True # maybe weight > 1.0
                ))
                start_idx = end_idx

    def get_len(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset.get_len(idx)

    def __len__(self):
        return len(self.dynamic_idx)
    
    def __getitem__(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset[idx]

    def _merge_batches(self, a: dict, b: dict) -> dict:
        import torch
        out = dict(a)
        for k, vb in b.items():
            if k not in out:
                out[k] = vb
                continue
            va = out[k]
            try:
                if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                    if va.numel() == 0:
                        out[k] = vb
                    elif vb.numel() == 0:
                        out[k] = va
                    else:
                        out[k] = torch.cat([va, vb], dim=0)
                elif isinstance(va, list) and isinstance(vb, list):
                    out[k] = va + vb
                elif isinstance(va, tuple) and isinstance(vb, tuple):
                    out[k] = tuple(list(va) + list(vb))
                else:
                    # Fallback: keep the first if incompatible types
                    out[k] = va if va is not None else vb
            except Exception:
                out[k] = va if va is not None else vb
        return out

    def _mixed_collate(self, batched_batch):
        """Collate a batch coming from a MixDatasetWrapper.

        Items originate from different sub‑datasets that may return different
        sample structures. We detect the minibatch shape and dispatch to the
        corresponding sub‑dataset collate_fn, then merge the resulting dicts.
        """
        if not batched_batch:
            # Defer to the first dataset's collate as a no‑op fallback
            return self.datasets[0].collate_fn(batched_batch)

        # Heuristic: a DynamicBatchWrapper around ResidueFragmentWrapper yields
        # minibatches as List[dict]; a DynamicBatchWrapper around RAGBatchWrapper
        # yields List[Tuple[...]]. We support mixing by collating each subgroup
        # separately and then concatenating.
        group0, group1 = [], []  # 0 -> like dict, 1 -> like tuple/others
        for minibatch in batched_batch:
            head = None
            try:
                if isinstance(minibatch, (list, tuple)) and len(minibatch) > 0:
                    head = minibatch[0]
            except Exception:
                head = None
            if isinstance(head, dict):
                group0.append(minibatch)
            else:
                group1.append(minibatch)

        results = []
        # Safeguard: if datasets are fewer than expected, fall back to first
        if group0:
            coll0 = self.datasets[0].collate_fn if hasattr(self.datasets[0], 'collate_fn') else None
            if coll0 is None:
                results.append(group0)
            else:
                results.append(coll0(group0))
        if group1:
            idx1 = 1 if len(self.datasets) > 1 else 0
            coll1 = self.datasets[idx1].collate_fn if hasattr(self.datasets[idx1], 'collate_fn') else None
            if coll1 is None:
                results.append(group1)
            else:
                results.append(coll1(group1))

        if not results:
            return self.datasets[0].collate_fn(batched_batch)
        if len(results) == 1:
            return results[0]
        return self._merge_batches(results[0], results[1])


class StandardIterator:
    def __init__(self, indexes):
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        return self.indexes[i]
    
    def prefecth(self, i):
        return self.__getitem__(i)

    def done_batch(self):
        pass
    

class PackIterator(StandardIterator):
    def __init__(self, indexes, lengths):
        super().__init__(indexes)
        
        self.ordered_indexes = sorted(self.indexes, key=lambda i: lengths[i], reverse=True)
        self.idx_to_sorted = { i: sorted_i for sorted_i, i in enumerate(self.ordered_indexes) }

        # for recording (dynamically change during iteration)
        self.not_visited = { idx: True for idx in self.indexes }
        self.last_visited = None
        self.within_batch = False

        # TODO: prefetch, and local batch bias

    def done_batch(self):
        self.with_in_batch = False

    def __getitem__(self, i, prefetch=False):
        if self.within_batch:
            rank = self.idx_to_sorted[self.last_visited]
            offset = 1
            while True:
                left, right = rank - offset, rank + offset
                idx = None
                if left >=0 and self.ordered_indexes[left] in self.not_visited:
                    idx = self.ordered_indexes[left]
                elif right < len(self.ordered_indexes) and self.ordered_indexes[right] in self.not_visited:
                    idx = self.ordered_indexes[right]
                offset += 1
                if idx is not None: break
        else: # start a new batch
            assert len(self.not_visited)
            for idx in self.not_visited:
                break
            if not prefetch:
                self.within_batch = True
        if not prefetch:
            self.last_visited = idx
            self.not_visited.pop(idx)
        return idx
    
    def prefecth(self, i):
        return self.__getitem__(i, prefetch=True)


@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch, n_use_max_in_batch=False, pack_similar_len=False) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.indexes = [i for i in range(len(self.dataset))]
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.n_use_max_in_dataset = n_use_max_in_batch
        self.pack_similar_len = pack_similar_len
        if self.pack_similar_len: # put items with similar lengths together
            assert n_use_max_in_batch, 'Pack_similar_len enabled, but not in the mode n_use_max_in_batch' # otherwise the packing algorithm is not necessary
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_complexity = 0
        batch = []

        # if self.pack_similar_len:
        #     iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
        # else:
        #     iterator = StandardIterator(self.indexes)
        if self.pack_similar_len:
            batch_max_n = 0
            iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
            for idx in tqdm(range(len(iterator)), ascii=True):
                i = iterator.prefecth(idx)
                n = self.dataset.get_len(i)
                if self.eval_func(n) > self.ubound_per_batch:
                    i = iterator[idx] # record visited
                    continue
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    iterator.done_batch()
                    i = iterator[idx] # get a new one for a new batch
                    n = self.dataset.get_len(i)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch

        elif self.n_use_max_in_dataset:
            batch_max_n = 0
            for i in tqdm(self.indexes, ascii=True):
                n = self.dataset.get_len(i)
                if self.eval_func(n) > self.ubound_per_batch:
                    continue
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch
        else:
            for i in tqdm(self.indexes, ascii=True):
                item_len = self.eval_func(self.dataset.get_len(i))
                if item_len > self.ubound_per_batch:
                    continue
                cur_complexity += item_len
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = item_len
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)


@R.register('RAGBatchWrapper')
class RAGBatchWrapper(torch.utils.data.Dataset):
    """Inject RAG fields (source SMILES and positive fragment SMILES) into each batch.

    Expects a JSONL file where each line is a dict with keys:
      - id: dataset item id (matches `dataset.get_id(i)`)
      - source_smiles: canonical molecule SMILES
      - pos_frag_smiles: list[str] of fragment SMILES as positives
    """

    def __init__(self, dataset, rag_map_path: Optional[str] = None, dynamic_partition: Optional[dict] = None) -> None:
        super().__init__()
        import json
        self.dataset = R.recur_construct(dataset)
        self.dynamic_sampler = _DynamicFragmentSampler(dynamic_partition)
        if hasattr(self.dataset, 'set_partition_log_path') and self.dynamic_sampler.partition_log_path:
            try:
                self.dataset.set_partition_log_path(self.dynamic_sampler.partition_log_path)
            except Exception:
                pass
        self.id2entry = {}
        if rag_map_path is not None:
            with open(rag_map_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    _id = rec.get("id")
                    if not _id:
                        continue
                    self.id2entry[_id] = {
                        "source_smiles": rec.get("source_smiles"),
                        "pos_frag_smiles": rec.get("pos_frag_smiles", []),
                    }
        if not self.id2entry and not self.dynamic_sampler.enabled:
            raise ValueError("RAGBatchWrapper requires either rag_map_path or dynamic_partition")

    def __len__(self):
        return len(self.dataset)

    def get_id(self, idx):
        if hasattr(self.dataset, 'get_id'):
            return self.dataset.get_id(idx)
        raise AttributeError('DynamicBlockWrapper underlying dataset lacks get_id')

    def get_summary(self, idx):
        if hasattr(self.dataset, 'get_summary'):
            return self.dataset.get_summary(idx)
        return None

    def get_len(self, idx):
        if hasattr(self.dataset, 'get_len'):
            return self.dataset.get_len(idx)
        raise AttributeError('RAGBatchWrapper underlying dataset lacks get_len')

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item_id = self.dataset.get_id(idx)
        dynamic_entry = None
        if self.dynamic_sampler.enabled:
            summary = self.dataset.get_summary(idx) if hasattr(self.dataset, 'get_summary') else None
            smiles = getattr(summary, 'ref_seq', None)
            dynamic_entry = self.dynamic_sampler.sample(smiles, metadata={'id': item_id})
        return data, item_id, dynamic_entry

    def collate_fn(self, batch):
        if len(batch[0]) == 3:
            data_list, id_list, dyn_list = zip(*batch)
        else:
            data_list, id_list = zip(*batch)
            dyn_list = None
        base = self.dataset.collate_fn(list(data_list))
        if self.dynamic_sampler.enabled and dyn_list is not None:
            rag_src = []
            for entry in dyn_list:
                if entry is None:
                    rag_src.append(None)
                else:
                    src, frags = entry
                    rag_src.append(src)
            base["rag_source_smiles"] = rag_src
        else:
            rag_src = []
            for _id in id_list:
                rec = self.id2entry.get(_id, None)
                if rec is None:
                    rag_src.append(None)
                else:
                    rag_src.append(rec["source_smiles"])
            base["rag_source_smiles"] = rag_src
        base['sample_ids'] = list(id_list)
        return base


AA_ABRV_TO_ONE = {abrv: symbol for symbol, abrv in aas}


@R.register('DynamicBlockWrapper')
class DynamicBlockWrapper(torch.utils.data.Dataset):
    """Wrap a dataset and regenerate ligand blocks via random partitions."""

    def __init__(
            self,
            dataset,
            num_parts=(2, 3, 4),
            seed: Optional[int] = None,
            max_attempts: int = 5,
            storage_path: Optional[str] = None,
            storage_min_heavy_atoms: int = 1,
            storage_metadata: bool = True,
            min_heavy_atoms: Optional[Sequence[int]] = None,
        ) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.num_parts = num_parts
        self.max_attempts = max(1, max_attempts)
        self.random = random.Random(seed)
        self.block_dummy_idx = VOCAB.get_block_dummy_idx()
        self.storage = None
        self.storage_metadata = storage_metadata
        self.partition_log_path: Optional[str] = None
        # Allow staged thresholds (e.g., [8, 4]) so that we can relax the minimum
        # heavy-atom requirement instead of returning an empty fragment set. Values
        # are clamped to non-negative integers to avoid unexpected behaviour.
        if isinstance(min_heavy_atoms, (list, tuple)):
            seq = [max(0, int(v)) for v in min_heavy_atoms if v is not None]
        elif min_heavy_atoms is None:
            seq = []
        else:
            seq = [max(0, int(min_heavy_atoms))]
        self.min_heavy_atom_thresholds = seq if seq else [0]
        if storage_path is not None:
            self.storage = SubgraphStorage(storage_path, min_heavy_atoms=storage_min_heavy_atoms)

    def __len__(self):
        return len(self.dataset)

    def get_len(self, idx):
        if hasattr(self.dataset, 'get_len'):
            return self.dataset.get_len(idx)
        raise AttributeError('DynamicBlockWrapper underlying dataset lacks get_len')

    def get_id(self, idx):
        if hasattr(self.dataset, 'get_id'):
            return self.dataset.get_id(idx)
        raise AttributeError('DynamicBlockWrapper underlying dataset lacks get_id')

    def get_summary(self, idx):
        if hasattr(self.dataset, 'get_summary'):
            return self.dataset.get_summary(idx)
        return None

    def set_partition_log_path(self, path: Optional[str]) -> None:
        self.partition_log_path = path

    def _choose_num_parts(self, ligand_atoms: int) -> int:
        if isinstance(self.num_parts, int):
            return max(1, min(int(self.num_parts), ligand_atoms))
        if isinstance(self.num_parts, (list, tuple)) and len(self.num_parts):
            choice = int(self.random.choice(self.num_parts))
            return max(1, min(choice, ligand_atoms))
        return 1

    def _log_partitions(
            self,
            mol: Optional[Chem.Mol],
            raw_partitions: Sequence[Sequence[int]],
            raw_fragments: Optional[Sequence[Optional[str]]],
            final_partitions: Optional[Sequence[Sequence[int]]],
            final_fragments: Optional[Sequence[Optional[str]]],
            summary,
            sample_type: Optional[str] = None,
        ) -> None:
        if not self.partition_log_path or not raw_partitions or mol is None:
            return
        try:
            os.makedirs(os.path.dirname(self.partition_log_path), exist_ok=True)
        except Exception:
            pass
        try:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            canonical = None
        if (not raw_fragments) or len(raw_fragments) != len(raw_partitions):
            try:
                raw_fragments = fragment_smiles_for_partitions(mol, list(raw_partitions))
            except Exception:
                raw_fragments = [None] * len(raw_partitions)
        if final_partitions is None:
            final_partitions = []
        if final_fragments is None or len(final_fragments) != len(final_partitions):
            try:
                final_fragments = fragment_smiles_for_partitions(mol, list(final_partitions)) if final_partitions else []
            except Exception:
                final_fragments = [None] * len(final_partitions)
        metadata_id = getattr(summary, 'id', None) if summary is not None else None
        if not sample_type:
            sample_type = 'molecule' if canonical else 'peptide'
        record = {
            'type': sample_type,
            'id': metadata_id,
            'source_smiles': canonical,
            'raw_partitions': [list(map(int, part)) for part in raw_partitions],
            'raw_fragments': list(raw_fragments) if raw_fragments is not None else None,
            'final_partitions': [list(map(int, part)) for part in final_partitions],
            'final_fragments': list(final_fragments) if final_fragments is not None else None,
            # keep backward-compatible keys for downstream tooling
            'partitions': [list(map(int, part)) for part in final_partitions],
            'fragments': list(final_fragments) if final_fragments is not None else None,
        }
        handle = None
        try:
            handle = open(self.partition_log_path, 'a', encoding='utf-8')
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            pass
        finally:
            if handle is not None:
                if fcntl is not None:
                    try:
                        handle.flush()
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    except Exception:
                        pass
                handle.close()

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        summary = self.dataset.get_summary(idx) if hasattr(self.dataset, 'get_summary') else None
        dyn_info = self._build_dynamic_blocks(data, summary)
        if isinstance(data, dict):
            data = dict(data)
            data['_dyn_info'] = dyn_info
        else:
            # fall back: wrap into dict-like structure
            data = {'_base_data': data, '_dyn_info': dyn_info}
        return data

    def _build_dynamic_blocks(self, data: dict, summary) -> dict:
        block_lengths = data['block_lengths'].long()
        block_types = data['S'].long()
        generate_mask = data['generate_mask'].bool()
        chain_ids = data['chain_ids'].long()
        position_ids = data['position_ids'].long()
        is_aa = data['is_aa'].bool()

        num_blocks = int(block_lengths.shape[0])
        block_offsets = torch.repeat_interleave(torch.arange(num_blocks, dtype=torch.long), block_lengths)

        context_block_indices = torch.nonzero(~generate_mask, as_tuple=False).view(-1)
        ligand_block_indices = torch.nonzero(generate_mask, as_tuple=False).view(-1)

        ligand_mask = torch.zeros_like(block_offsets, dtype=torch.bool)
        for blk in ligand_block_indices.tolist():
            ligand_mask |= (block_offsets == blk)
        ligand_atom_indices = torch.nonzero(ligand_mask, as_tuple=False).view(-1)

        partitions: Optional[List[List[int]]] = None
        ligand_contains_aa = bool(is_aa[ligand_block_indices].any().item()) if ligand_block_indices.numel() else False
        sample_type = 'peptide' if ligand_contains_aa else 'molecule'
        ligand_smiles = getattr(summary, 'ref_seq', None) if summary is not None else None
        num_ligand_atoms = int(ligand_atom_indices.numel())
        frag_smiles_list: Optional[List[str]] = None
        mol = None
        if ligand_smiles and num_ligand_atoms > 0:
            try:
                mol = Chem.MolFromSmiles(ligand_smiles)
            except Exception:
                mol = None
        if mol is not None and num_ligand_atoms > 0:
            candidate_parts: List[int]
            if isinstance(self.num_parts, int):
                candidate_parts = [self.num_parts]
            elif isinstance(self.num_parts, (list, tuple)):
                candidate_parts = sorted({int(max(1, p)) for p in self.num_parts if isinstance(p, (int, float))}, reverse=True)
            else:
                candidate_parts = [2]
            if 2 not in candidate_parts:
                candidate_parts.append(2)
            candidate_parts = [min(int(p), num_ligand_atoms) for p in candidate_parts if int(p) > 0]
            candidate_parts = [p for p in candidate_parts if p >= 1]
            if not candidate_parts:
                candidate_parts = [1]
            attempts = 0
            max_trials = max(self.max_attempts, 1)
            for requested in candidate_parts:
                if requested <= 0:
                    continue
                trials = 0
                while trials < max_trials:
                    trials += 1
                    seed = self.random.randint(0, 2**32 - 1)
                    try:
                        partitions = random_subgraph_partition(mol, num_partitions=requested, seed=seed)
                    except ValueError:
                        partitions = None
                    if partitions and len(partitions) > 1:
                        try:
                            frag_smiles_list = fragment_smiles_for_partitions(mol, partitions)
                        except Exception:
                            frag_smiles_list = None
                        if frag_smiles_list and len(frag_smiles_list) == len(partitions):
                            break
                        partitions = None
                        frag_smiles_list = None
                if partitions and len(partitions) > 1:
                    break
        if mol is not None and num_ligand_atoms > 0:
            if not partitions or len(partitions) <= 1 or not frag_smiles_list or len(frag_smiles_list) < len(partitions):
                fallback = self._fallback_partitions_from_blocks(
                    ligand_block_indices,
                    block_offsets,
                    ligand_atom_indices
                )
                if fallback and len(fallback) > 1:
                    partitions = fallback
                    try:
                        frag_smiles_list = fragment_smiles_for_partitions(mol, partitions)
                    except Exception:
                        frag_smiles_list = None
        if mol is not None and num_ligand_atoms > 0:
            if (not partitions or len(partitions) <= 1) and num_ligand_atoms > 1:
                connected = self._connected_partition(mol, min(2, num_ligand_atoms), self.random.randint(0, 2**32 - 1))
                if connected and len(connected) > 1:
                    partitions = connected
                    try:
                        frag_smiles_list = fragment_smiles_for_partitions(mol, partitions)
                    except Exception:
                        frag_smiles_list = None
        if not partitions:
            fallback_blocks = self._fallback_partitions_from_blocks(
                ligand_block_indices,
                block_offsets,
                ligand_atom_indices
            ) if num_ligand_atoms > 0 else []
            if fallback_blocks:
                partitions = fallback_blocks
            else:
                partitions = [list(range(num_ligand_atoms))]
            frag_smiles_list = None

        if partitions:
            original_partitions = list(partitions)
            base_smiles = frag_smiles_list if frag_smiles_list is not None else [None] * len(partitions)
            original_smiles = list(base_smiles)
            for threshold in self.min_heavy_atom_thresholds:
                if threshold <= 0:
                    filtered_partitions = original_partitions
                    filtered_smiles = original_smiles
                else:
                    filtered_partitions = []
                    filtered_smiles = []
                    for part, smi in zip(original_partitions, original_smiles):
                        if len(part) >= threshold:
                            filtered_partitions.append(part)
                            filtered_smiles.append(smi)
                if filtered_partitions:
                    partitions = filtered_partitions
                    frag_smiles_list = filtered_smiles
                    break
            else:
                partitions = original_partitions
                frag_smiles_list = original_smiles

        if mol is not None and 'original_partitions' in locals():
            self._log_partitions(
                mol,
                original_partitions,
                original_smiles,
                partitions,
                frag_smiles_list,
                summary,
                sample_type,
            )

        if self.storage is not None and ligand_smiles and partitions:
            try:
                metadata = {'id': getattr(summary, 'id', None)} if self.storage_metadata and summary is not None else None
                self.storage.store_partitions(ligand_smiles, partitions, metadata=metadata)
            except Exception:
                pass

        ligand_atom_indices_list = ligand_atom_indices.tolist()
        mapped_partitions: List[List[int]] = []
        for part in partitions:
            mapped = [ligand_atom_indices_list[idx] for idx in part if idx < len(ligand_atom_indices_list)]
            if mapped:
                mapped_partitions.append(sorted(mapped))
        if not mapped_partitions:
            mapped_partitions = [ligand_atom_indices_list]

        dyn_block_lengths: List[int] = []
        dyn_block_types: List[int] = []
        dyn_generate_mask: List[bool] = []
        dyn_chain_ids: List[int] = []
        dyn_position_ids: List[int] = []
        dyn_is_aa: List[bool] = []
        dyn_block_ids = torch.full_like(block_offsets, fill_value=-1, dtype=torch.long)

        # preserve context blocks
        for new_idx, orig_idx in enumerate(context_block_indices.tolist()):
            mask = (block_offsets == orig_idx)
            dyn_block_ids[mask] = new_idx
            dyn_block_lengths.append(int(block_lengths[orig_idx].item()))
            dyn_block_types.append(int(block_types[orig_idx].item()))
            dyn_generate_mask.append(False)
            dyn_chain_ids.append(int(chain_ids[orig_idx].item()))
            dyn_position_ids.append(int(position_ids[orig_idx].item()))
            dyn_is_aa.append(bool(is_aa[orig_idx].item()))

        current_idx = len(dyn_block_lengths)
        dyn_fragment_smiles: List[Optional[str]] = [None] * current_idx
        ligand_chain_id = int(chain_ids[ligand_block_indices[0]].item()) if ligand_block_indices.numel() else 0
        for part_idx, partition in enumerate(mapped_partitions):
            if not partition:
                continue
            for atom_idx in partition:
                dyn_block_ids[atom_idx] = current_idx
            dyn_block_lengths.append(len(partition))
            dyn_block_types.append(self.block_dummy_idx)
            dyn_generate_mask.append(True)
            dyn_chain_ids.append(ligand_chain_id)
            dyn_position_ids.append(0)
            dyn_is_aa.append(False)
            if frag_smiles_list is not None and part_idx < len(frag_smiles_list):
                dyn_fragment_smiles.append(frag_smiles_list[part_idx])
            else:
                dyn_fragment_smiles.append(None)
            current_idx += 1

        remaining_atoms = torch.nonzero(dyn_block_ids < 0, as_tuple=False).view(-1)
        for atom_idx in remaining_atoms.tolist():
            dyn_block_ids[atom_idx] = current_idx
            dyn_block_lengths.append(1)
            dyn_block_types.append(self.block_dummy_idx)
            dyn_generate_mask.append(True)
            dyn_chain_ids.append(ligand_chain_id)
            dyn_position_ids.append(0)
            dyn_is_aa.append(False)
            dyn_fragment_smiles.append(None)
            current_idx += 1

        dyn = {
            'dyn_block_lengths': torch.tensor(dyn_block_lengths, dtype=torch.long),
            'dyn_block_types': torch.tensor(dyn_block_types, dtype=torch.long),
            'dyn_generate_mask': torch.tensor(dyn_generate_mask, dtype=torch.bool),
            'dyn_block_ids': dyn_block_ids,
            'dyn_chain_ids': torch.tensor(dyn_chain_ids, dtype=torch.long),
            'dyn_position_ids': torch.tensor(dyn_position_ids, dtype=torch.long),
            'dyn_is_aa': torch.tensor(dyn_is_aa, dtype=torch.bool),
            'dyn_fragment_smiles': dyn_fragment_smiles,
            'dyn_num_blocks': torch.tensor([len(dyn_block_lengths)], dtype=torch.long),
        }
        return dyn

    def collate_fn(self, batch):
        base_batch = []
        dyn_list = []
        for sample in batch:
            if isinstance(sample, dict) and '_dyn_info' in sample:
                dyn_list.append(sample.pop('_dyn_info'))
                base_batch.append(sample)
            elif isinstance(sample, dict):
                dyn_list.append({})
                base_batch.append(sample)
            else:
                dyn_list.append(sample.get('_dyn_info', {}))
                base_batch.append(sample.get('_base_data', sample))
        base = self.dataset.collate_fn(base_batch)

        lengths_cat = []
        types_cat = []
        mask_cat = []
        ids_cat = []
        chain_cat = []
        pos_cat = []
        is_aa_cat = []
        num_blocks_cat = []
        fragment_cat: List[Optional[str]] = []

        block_offset = 0
        for dyn in dyn_list:
            if not dyn:
                continue
            lengths = dyn['dyn_block_lengths']
            types = dyn['dyn_block_types']
            mask = dyn['dyn_generate_mask']
            ids = dyn['dyn_block_ids'] + block_offset

            lengths_cat.append(lengths)
            types_cat.append(types)
            mask_cat.append(mask)
            ids_cat.append(ids)
            chain_cat.append(dyn['dyn_chain_ids'])
            pos_cat.append(dyn['dyn_position_ids'])
            is_aa_cat.append(dyn['dyn_is_aa'])
            num_blocks_cat.append(torch.tensor([lengths.shape[0]], dtype=torch.long))
            fragment_cat.extend(dyn.get('dyn_fragment_smiles', [None] * lengths.shape[0]))

            block_offset += lengths.shape[0]

        base['dyn_block_lengths'] = torch.cat(lengths_cat, dim=0) if lengths_cat else torch.empty(0, dtype=torch.long)
        base['dyn_block_types'] = torch.cat(types_cat, dim=0) if types_cat else torch.empty(0, dtype=torch.long)
        base['dyn_generate_mask'] = torch.cat(mask_cat, dim=0) if mask_cat else torch.empty(0, dtype=torch.bool)
        base['dyn_block_ids'] = torch.cat(ids_cat, dim=0) if ids_cat else torch.empty(0, dtype=torch.long)
        base['dyn_chain_ids'] = torch.cat(chain_cat, dim=0) if chain_cat else torch.empty(0, dtype=torch.long)
        base['dyn_position_ids'] = torch.cat(pos_cat, dim=0) if pos_cat else torch.empty(0, dtype=torch.long)
        base['dyn_is_aa'] = torch.cat(is_aa_cat, dim=0) if is_aa_cat else torch.empty(0, dtype=torch.bool)
        base['dyn_num_blocks'] = torch.cat(num_blocks_cat, dim=0) if num_blocks_cat else torch.empty(0, dtype=torch.long)
        base['dyn_fragment_smiles'] = fragment_cat
        return base

    def _fallback_partitions_from_blocks(self, ligand_block_indices, block_offsets, ligand_atom_indices):
        if ligand_block_indices.numel() == 0:
            return []
        ligand_atom_indices_list = ligand_atom_indices.tolist()
        local_map = {idx: local for local, idx in enumerate(ligand_atom_indices_list)}
        partitions: List[List[int]] = []
        for blk in ligand_block_indices.tolist():
            atom_indices = torch.nonzero(block_offsets == blk, as_tuple=False).view(-1)
            if atom_indices.numel() == 0:
                continue
            local_indices = [local_map.get(int(atom_idx.item())) for atom_idx in atom_indices if int(atom_idx.item()) in local_map]
            if not local_indices:
                continue
            part = sorted(set(local_indices))
            if part:
                partitions.append(part)
        unique_partitions = []
        seen = set()
        for part in partitions:
            key = tuple(part)
            if key not in seen:
                seen.add(key)
                unique_partitions.append(part)
        return unique_partitions

    def _connected_partition(self, mol: Chem.Mol, num_parts: int, seed: Optional[int]) -> List[List[int]]:
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return []
        num_parts = max(1, min(num_parts, num_atoms))
        if num_parts == 1:
            return [list(range(num_atoms))]
        rng = random.Random(seed)
        atoms = list(range(num_atoms))
        seeds = rng.sample(atoms, k=num_parts)
        labels = {}
        partitions: dict[int, List[int]] = {}
        for part_id, atom_idx in enumerate(seeds):
            labels[atom_idx] = part_id
            partitions[part_id] = [atom_idx]
        queue = deque(seeds)
        while queue:
            atom_idx = queue.popleft()
            part_id = labels[atom_idx]
            neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
            rng.shuffle(neighbors)
            for nbr_idx in neighbors:
                if nbr_idx in labels:
                    continue
                labels[nbr_idx] = part_id
                partitions.setdefault(part_id, []).append(nbr_idx)
                queue.append(nbr_idx)
        if len(labels) < num_atoms:
            for atom_idx in atoms:
                if atom_idx in labels:
                    continue
                neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetIdx() in labels]
                if neighbors:
                    part_id = labels[neighbors[0]]
                else:
                    part_id = 0
                labels[atom_idx] = part_id
                partitions.setdefault(part_id, []).append(atom_idx)
        ordered = [sorted(set(group)) for _, group in sorted(partitions.items()) if group]
        return ordered


@R.register('GuideBatchWrapper')
class GuideBatchWrapper(torch.utils.data.Dataset):
    """Attach training-time guidance fragments to batch for LDM.

    JSONL format (one per line):
      {"id": "<dataset_item_id>", "guide_frag_smiles": ["frag1", "frag2", ...]}
    """

    def __init__(self, dataset, guide_map_path: Optional[str] = None, dynamic_partition: Optional[dict] = None) -> None:
        super().__init__()
        import json
        self.dataset = R.recur_construct(dataset)
        self.dynamic_sampler = _DynamicFragmentSampler(dynamic_partition)
        self.id2frags = {}
        if guide_map_path is not None:
            with open(guide_map_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    _id = rec.get("id")
                    if not _id:
                        continue
                    self.id2frags[_id] = rec.get("guide_frag_smiles", [])
        if not self.id2frags and not self.dynamic_sampler.enabled:
            raise ValueError("GuideBatchWrapper requires either guide_map_path or dynamic_partition")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item_id = self.dataset.get_id(idx)
        dynamic_entry = None
        if self.dynamic_sampler.enabled:
            summary = self.dataset.get_summary(idx) if hasattr(self.dataset, 'get_summary') else None
            smiles = getattr(summary, 'ref_seq', None)
            dynamic_entry = self.dynamic_sampler.sample(smiles, metadata={'id': item_id})
        return data, item_id, dynamic_entry

    def collate_fn(self, batch):
        if len(batch[0]) == 3:
            data_list, id_list, dyn_list = zip(*batch)
        else:
            data_list, id_list = zip(*batch)
            dyn_list = None
        base = self.dataset.collate_fn(list(data_list))
        if self.dynamic_sampler.enabled and dyn_list is not None:
            guide_list = []
            for entry in dyn_list:
                if entry is None:
                    guide_list.append([])
                else:
                    _, frags = entry
                    guide_list.append(frags)
            base["guide_frag_smiles"] = guide_list
        else:
            guide_list = []
            for _id in id_list:
                guide_list.append(self.id2frags.get(_id, []))
            base["guide_frag_smiles"] = guide_list
        return base


@R.register('ResidueFragmentWrapper')
class ResidueFragmentWrapper(torch.utils.data.Dataset):
    """Attach per-residue fragment SMILES (and optional storage) for peptide/antibody datasets."""

    def __init__(
            self,
            dataset,
            storage_path: Optional[str] = None,
            storage_min_heavy_atoms: int = 1,
            storage_metadata: bool = True,
        ) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.storage = None
        self.storage_metadata = storage_metadata
        if storage_path is not None:
            self.storage = SubgraphStorage(storage_path, min_heavy_atoms=storage_min_heavy_atoms)

    def __len__(self):
        return len(self.dataset)

    def get_id(self, idx):
        if hasattr(self.dataset, 'get_id'):
            return self.dataset.get_id(idx)
        raise AttributeError('ResidueFragmentWrapper underlying dataset lacks get_id')

    def get_summary(self, idx):
        if hasattr(self.dataset, 'get_summary'):
            return self.dataset.get_summary(idx)
        return None

    def get_len(self, idx):
        if hasattr(self.dataset, 'get_len'):
            return self.dataset.get_len(idx)
        raise AttributeError('ResidueFragmentWrapper underlying dataset lacks get_len')

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        if not isinstance(data, dict):
            raise TypeError('ResidueFragmentWrapper expects underlying dataset to return dict samples')
        summary = self.get_summary(idx)
        fragments = self._build_residue_fragments(data, summary)
        sample = dict(data)
        sample['_residue_fragments'] = fragments
        sample['_sample_id'] = getattr(summary, 'id', None) if summary is not None else None
        return sample

    def _build_residue_fragments(self, data: dict, summary) -> List[Optional[str]]:
        block_types = data['S'].long()
        generate_mask = data.get('generate_mask', torch.ones_like(block_types, dtype=torch.bool)).bool()
        fragments: List[Optional[str]] = []
        sample_id = getattr(summary, 'id', None) if summary is not None else None
        for block_idx, block_type in enumerate(block_types.tolist()):
            abrv = VOCAB.idx_to_abrv(block_type)
            one_letter = AA_ABRV_TO_ONE.get(abrv)
            smiles = None
            if generate_mask[block_idx].item() and one_letter in aa_smiles:
                smiles = aa_smiles[one_letter]
                if self.storage is not None and smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            partitions = [list(range(mol.GetNumAtoms()))]
                            metadata = None
                            if self.storage_metadata:
                                metadata = {
                                    'id': sample_id,
                                    'block_index': block_idx,
                                    'residue': abrv
                                }
                            self.storage.store_partitions(smiles, partitions, metadata=metadata)
                    except Exception:
                        pass
            fragments.append(smiles)
        return fragments

    def collate_fn(self, batch):
        frag_lists = [sample.pop('_residue_fragments', None) for sample in batch]
        sample_ids = [sample.pop('_sample_id', None) for sample in batch]
        base = self.dataset.collate_fn(batch)
        if frag_lists and any(frag is not None for frag in frag_lists):
            fragment_cat: List[Optional[str]] = []
            for fragments in frag_lists:
                if fragments is None:
                    continue
                fragment_cat.extend(fragments)
            if fragment_cat:
                base['dyn_fragment_smiles'] = fragment_cat
                # mirror base block tensors so peptides can participate in dynamic RAG
                if 'block_lengths' in base and isinstance(base['block_lengths'], torch.Tensor):
                    base['dyn_block_lengths'] = base['block_lengths'].clone()
                if 'S' in base and isinstance(base['S'], torch.Tensor):
                    base['dyn_block_types'] = base['S'].clone()
                if 'generate_mask' in base and isinstance(base['generate_mask'], torch.Tensor):
                    base['dyn_generate_mask'] = base['generate_mask'].clone()
                if 'chain_ids' in base and isinstance(base['chain_ids'], torch.Tensor):
                    base['dyn_chain_ids'] = base['chain_ids'].clone()
                if 'position_ids' in base and isinstance(base['position_ids'], torch.Tensor):
                    base['dyn_position_ids'] = base['position_ids'].clone()
                if 'is_aa' in base and isinstance(base['is_aa'], torch.Tensor):
                    base['dyn_is_aa'] = base['is_aa'].clone()
                if 'lengths' in base and isinstance(base['lengths'], torch.Tensor):
                    base['dyn_num_blocks'] = base['lengths'].clone()
            additional = len(frag_lists)
            if 'rag_source_smiles' in base:
                base['rag_source_smiles'].extend([None] * additional)
            else:
                base['rag_source_smiles'] = [None] * additional
        if sample_ids and any(s is not None for s in sample_ids):
            if 'sample_ids' in base:
                existing = list(base['sample_ids'])
                existing.extend(sample_ids)
                base['sample_ids'] = existing
            else:
                base['sample_ids'] = list(sample_ids)
        return base
