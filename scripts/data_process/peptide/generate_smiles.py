#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Generate canonical peptide SMILES strings from the pepbench mmap dataset.

The script reconstructs an RDKit molecule for every ligand chain contained in
the peptide dataset and writes the resulting SMILES into a JSONL file.  Each
output line has the following structure:

    {
        "id": "<complex id>",
        "chain_id": "<ligand chain id>",
        "sequence": "<ligand amino-acid sequence>",
        "smiles": "<canonical smiles>",
        "num_atoms": <heavy atom count>
    }

Entries that fail the conversion are optionally recorded in a separate log.

python scripts/data_process/peptide/generate_smiles.py \\
  --dataset-root ./datasets/peptide/pepbench/processed \\
  --index train_index.txt valid_index.txt non_standard_index.txt \\
  --output ./datasets/peptide/pepbench/peptide_smiles.jsonl \\
  --fail-log ./logs/peptide_smiles_fail.jsonl \\
  --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm

# script resides in <repo>/scripts/data_process/peptide/
# move up to project root so that ``data.*`` imports succeed when run directly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.peptide import PeptideDataset
from data.bioparse.hierarchy import Complex
from data.bioparse.utils import bond_type_to_rdkit

Task = Tuple[int, str, str]  # (raw_idx, chain_id, sequence)
Result = Tuple[bool, str, str, str, str, Optional[int]]

_WORKER_DATASET: Optional[PeptideDataset] = None
_WORKER_INDEXES: Optional[Sequence[Tuple[str, ...]]] = None

# Map non-standard element symbols to RDKit-recognised ones (e.g. D -> H).
ELEMENT_ALIASES: Dict[str, str] = {
    'D': 'H',  # Deuterium
    'T': 'H',  # Tritium
}


def _build_chain_mol(cplx: Complex, chain_id: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Construct an RDKit molecule for a specific ligand chain."""

    ligand_idx = cplx.id2idx.get(chain_id)
    if ligand_idx is None:
        return None, f'chain_id {chain_id} not found in complex'

    rw_mol = Chem.RWMol()
    atom_map: Dict[Tuple[int, int], int] = {}

    periodic_table = rdchem.GetPeriodicTable()

    for block_idx, block in enumerate(cplx[ligand_idx]):
        for atom_idx, atom in enumerate(block):
            element = atom.get_element()
            if not element:
                return None, 'missing_element_symbol'
            raw_symbol = element.strip()
            if not raw_symbol:
                return None, 'empty_element_symbol'
            if len(raw_symbol) > 1:
                symbol = raw_symbol[0].upper() + raw_symbol[1:].lower()
            else:
                symbol = raw_symbol.upper()
            symbol = ELEMENT_ALIASES.get(symbol.upper(), symbol)
            atomic_num = periodic_table.GetAtomicNumber(symbol)
            if atomic_num <= 0:
                return None, f'unknown_element:{element}'
            rd_atom = Chem.Atom(symbol)
            formal_charge = atom.get_property('formal_charge', 0)
            try:
                rd_atom.SetFormalCharge(int(formal_charge))
            except Exception:
                rd_atom.SetFormalCharge(0)
            atom_map[(block_idx, atom_idx)] = rw_mol.AddAtom(rd_atom)

    for bond in cplx.bonds:
        idx1, idx2 = bond.index1, bond.index2
        if idx1[0] != ligand_idx or idx2[0] != ligand_idx:
            continue
        mapped1 = atom_map.get((idx1[1], idx1[2]))
        mapped2 = atom_map.get((idx2[1], idx2[2]))
        if mapped1 is None or mapped2 is None:
            continue
        bond_type = bond_type_to_rdkit(bond.bond_type)
        if bond_type is None:
            continue
        if rw_mol.GetBondBetweenAtoms(mapped1, mapped2) is None:
            rw_mol.AddBond(mapped1, mapped2, bond_type)

    mol = rw_mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, f'sanitize_failed: {exc}'
    return mol, None


def _finalise_record(complex_id: str, complex_obj: Complex, chain_id: str, sequence: str) -> Result:
    mol, error = _build_chain_mol(complex_obj, chain_id)
    if mol is None or error is not None:
        return False, complex_id, chain_id, sequence, error or 'unknown_error', None
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return True, complex_id, chain_id, sequence, smiles, mol.GetNumAtoms()


def _process_task_local(dataset: PeptideDataset, task: Task) -> Result:
    raw_idx, chain_id, sequence = task
    complex_id = dataset._indexes[raw_idx][0]
    complex_obj = dataset.get_raw_data(raw_idx)
    return _finalise_record(complex_id, complex_obj, chain_id, sequence)


def _worker_init(dataset_root: str, index_path: str) -> None:
    global _WORKER_DATASET, _WORKER_INDEXES
    _WORKER_DATASET = PeptideDataset(dataset_root, specify_index=index_path)
    _WORKER_INDEXES = _WORKER_DATASET._indexes


def _worker_process(task: Task) -> Result:
    if _WORKER_DATASET is None or _WORKER_INDEXES is None:
        raise RuntimeError("Worker dataset not initialised")
    raw_idx, chain_id, sequence = task
    complex_id = _WORKER_INDEXES[raw_idx][0]
    complex_obj = _WORKER_DATASET.get_raw_data(raw_idx)
    return _finalise_record(complex_id, complex_obj, chain_id, sequence)


def _gather_tasks(dataset: PeptideDataset, limit: Optional[int], seen_keys: set[Tuple[str, str]]) -> List[Task]:
    tasks: List[Task] = []
    for raw_idx, (complex_id, _, _) in enumerate(dataset._indexes):
        if limit is not None and raw_idx + 1 > limit:
            break
        props = dataset._properties[raw_idx]
        chain_ids = props.get('ligand_chain_ids', [])
        sequences = props.get('ligand_sequences', [])
        for chain_id, sequence in zip(chain_ids, sequences):
            key = (complex_id, chain_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            tasks.append((raw_idx, chain_id, sequence))
    return tasks


def _handle_result(result: Result,
                   out_file,
                   fail_handle,
                   totals: Tuple[int, int]) -> Tuple[int, int]:
    total_success, total_failed = totals
    success, complex_id, chain_id, sequence, payload, num_atoms = result
    if success:
        out_file.write(json.dumps({
            'id': complex_id,
            'chain_id': chain_id,
            'sequence': sequence,
            'smiles': payload,
            'num_atoms': num_atoms
        }, ensure_ascii=False) + '\n')
        total_success += 1
    else:
        total_failed += 1
        if fail_handle is not None:
            fail_handle.write(json.dumps({
                'id': complex_id,
                'chain_id': chain_id,
                'sequence': sequence,
                'error': payload
            }, ensure_ascii=False) + '\n')
    return total_success, total_failed


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate peptide SMILES from pepbench dataset.')
    parser.add_argument('--dataset-root', type=Path, default=Path('./datasets/peptide/pepbench/processed'),
                        help='Root directory of the pepbench mmap dataset.')
    parser.add_argument('--index', type=Path, nargs='+',
                        default=[Path('train_index.txt'), Path('valid_index.txt')],
                        help='One or more index files to process (relative or absolute).')
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination JSONL file for the generated SMILES.')
    parser.add_argument('--fail-log', type=Path, default=None,
                        help='Optional path to record entries that failed to convert.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional cap on the number of entries processed per index file.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes to use per index file (default: 1).')
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_handle = None
    if args.fail_log is not None:
        args.fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_handle = args.fail_log.open('w', encoding='utf-8')

    seen_keys: set[Tuple[str, str]] = set()
    total_success = 0
    total_failed = 0

    ctx = mp.get_context('spawn') if args.num_workers > 1 else None

    with output_path.open('w', encoding='utf-8') as out_file:
        for index_arg in args.index:
            index_path = index_arg if index_arg.is_absolute() else args.dataset_root / index_arg
            dataset = PeptideDataset(str(args.dataset_root), specify_index=str(index_path))
            tasks = _gather_tasks(dataset, args.limit, seen_keys)
            if not tasks:
                continue

            desc = f'Processing {index_path.name}'
            if args.num_workers > 1 and len(tasks) > 1:
                chunksize = max(1, len(tasks) // (args.num_workers * 4) or 1)
                assert ctx is not None
                with ctx.Pool(args.num_workers,
                              initializer=_worker_init,
                              initargs=(str(args.dataset_root), str(index_path))) as pool:
                    iterator = pool.imap_unordered(_worker_process, tasks, chunksize=chunksize)
                    for result in tqdm(iterator, total=len(tasks), ascii=True, desc=desc):
                        total_success, total_failed = _handle_result(
                            result, out_file, fail_handle, (total_success, total_failed))
            else:
                for task in tqdm(tasks, ascii=True, desc=desc):
                    result = _process_task_local(dataset, task)
                    total_success, total_failed = _handle_result(
                        result, out_file, fail_handle, (total_success, total_failed))
            del dataset

    if fail_handle is not None:
        fail_handle.close()

    print(f'Completed. Success: {total_success}, Failed: {total_failed}')


if __name__ == '__main__':
    main()
