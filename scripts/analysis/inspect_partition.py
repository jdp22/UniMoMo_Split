#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Inspect DynamicBlockWrapper partitions for custom SMILES.

Usage examples:
  python scripts/analysis/inspect_partition.py --smiles "NS(=O)(=O)c1ccccc1"
  python scripts/analysis/inspect_partition.py --smiles-file smiles.txt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

from data.dataset_wrapper import DynamicBlockWrapper


def _init_wrapper(num_parts: Sequence[int], max_attempts: int, seed: int, min_fragment_atoms: Optional[int]) -> DynamicBlockWrapper:
    wrapper = DynamicBlockWrapper.__new__(DynamicBlockWrapper)  # type: ignore[misc]
    wrapper.random = random.Random(seed)
    wrapper._partition_seeds = (7801, 1, 12, 123, 1234, 12345)
    wrapper.max_attempts = max(1, int(max_attempts))
    wrapper.num_parts = tuple(int(p) for p in num_parts)
    wrapper.block_dummy_idx = 0
    wrapper.storage = None
    wrapper.storage_metadata = True
    wrapper.min_heavy_atom_thresholds = [0]
    wrapper.partition_log_path = None
    if min_fragment_atoms is not None:
        wrapper._min_fragment_atoms_value = max(2, int(min_fragment_atoms))
    else:
        wrapper._min_fragment_atoms_value = None
    return wrapper


def _tensor_indices(num_atoms: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ligand_block_indices = torch.arange(num_atoms, dtype=torch.long)
    block_offsets = torch.arange(num_atoms, dtype=torch.long)
    ligand_atom_indices = torch.arange(num_atoms, dtype=torch.long)
    return ligand_block_indices, block_offsets, ligand_atom_indices


def inspect_smiles(
        wrapper: DynamicBlockWrapper,
        smiles: str,
        partitions: Optional[List[List[int]]] = None,
        fragments: Optional[List[Optional[str]]] = None,
) -> Tuple[List[List[int]], List[Optional[str]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    ligand_block_indices, block_offsets, ligand_atom_indices = _tensor_indices(mol.GetNumAtoms())
    part, frag = wrapper._ensure_partition_quality(
        mol,
        partitions,
        fragments,
        ligand_block_indices,
        block_offsets,
        ligand_atom_indices,
    )
    return part, frag


def main():
    parser = argparse.ArgumentParser(description="Inspect DynamicBlockWrapper fragment partitions for SMILES input.")
    parser.add_argument("--smiles", nargs="*", default=[], help="Explicit SMILES strings to inspect.")
    parser.add_argument("--smiles-file", type=str, default=None, help="Optional file containing SMILES (one per line).")
    parser.add_argument("--num-parts", type=int, nargs="*", default=[2, 3, 4], help="Candidate number of partitions.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum attempts for random partitioning.")
    parser.add_argument("--seed", type=int, default=7801, help="Random seed for reproducibility.")
    parser.add_argument("--min-fragment-atoms", type=int, default=None, help="Minimum atom count per fragment.")

    args = parser.parse_args()

    smiles_list: List[str] = list(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, "r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if s:
                    smiles_list.append(s)

    if not smiles_list:
        raise SystemExit("No SMILES provided. Use --smiles or --smiles-file.")

    wrapper = _init_wrapper(args.num_parts, args.max_attempts, args.seed, args.min_fragment_atoms)

    for idx, smi in enumerate(smiles_list, start=1):
        try:
            partitions, fragments = inspect_smiles(wrapper, smi)
        except Exception as exc:  # pragma: no cover
            print(f"[{idx}] SMILES: {smi}")
            print(f"  ERROR: {exc}")
            continue
        covered_atoms = sorted({atom for part in partitions for atom in part})
        print(f"[{idx}] SMILES: {smi}")
        print(f"  Num atoms     : {len(covered_atoms)}")
        print(f"  Num partitions: {len(partitions)}")
        for part_id, (part, frag) in enumerate(zip(partitions, fragments)):
            print(f"    Part {part_id}: atoms={part}; fragment={frag}")
        print()


if __name__ == "__main__":
    main()
