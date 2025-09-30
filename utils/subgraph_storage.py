"""Utilities for persisting molecular subgraphs for retrieval workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from rdkit import Chem


AtomIndices = Sequence[int]


def _ensure_mol(mol_or_smiles: Union[str, Chem.Mol]) -> Chem.Mol:
    """Convert a SMILES string into an ``rdkit.Chem.Mol`` if needed."""

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles!r}")
        return mol
    return mol_or_smiles


def _fragment_smiles(mol: Chem.Mol, partitions: Sequence[AtomIndices]) -> Sequence[str]:
    """Return canonical fragment SMILES for every atom index group."""

    if not partitions:
        return []

    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    return [
        Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(group),
            canonical=True,
            isomericSmiles=True,
            allHsExplicit=has_explicit_h,
        )
        for group in partitions
    ]


def _heavy_atom_count(mol: Chem.Mol, atom_indices: Iterable[int]) -> int:
    """Count the number of atoms excluding hydrogens for a partition."""

    return sum(1 for idx in atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)


class SubgraphStorage:
    """JSONL-based persistence for molecular subgraphs.

    The storage records each fragment with its canonical SMILES string, the
    originating molecule, atom indices, and heavy atom count. Entries can be
    filtered by providing ``min_heavy_atoms`` to skip tiny fragments such as
    isolated hydrogens.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        *,
        min_heavy_atoms: int = 0,
    ) -> None:
        if min_heavy_atoms < 0:
            raise ValueError("min_heavy_atoms must be non-negative")

        self.output_path = Path(output_path)
        if self.output_path.parent and not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_heavy_atoms = min_heavy_atoms

    def store_partitions(
        self,
        mol_or_smiles: Union[str, Chem.Mol],
        partitions: Sequence[AtomIndices],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Persist qualifying fragments for a molecule.

        Args:
            mol_or_smiles: Source molecule or SMILES string.
            partitions: Atom index partitions describing each fragment.
            metadata: Optional mapping with contextual data that will be stored
                alongside each fragment entry.

        Returns:
            The number of fragments written to ``output_path``.
        """

        mol = _ensure_mol(mol_or_smiles)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        fragment_smiles = _fragment_smiles(mol, partitions)
        threshold = self.min_heavy_atoms

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping if provided")
        metadata_dict = dict(metadata) if metadata is not None else None

        stored = 0
        with self.output_path.open("a", encoding="utf-8") as handle:
            for index, (atom_indices, frag_smi) in enumerate(zip(partitions, fragment_smiles)):
                heavy_atoms = _heavy_atom_count(mol, atom_indices)
                if threshold > 0 and heavy_atoms <= threshold:
                    continue

                record = {
                    "source_smiles": canonical_smiles,
                    "fragment_smiles": frag_smi,
                    "atom_indices": list(map(int, atom_indices)),
                    "heavy_atom_count": heavy_atoms,
                    "partition_index": index,
                }
                if metadata_dict:
                    record["metadata"] = metadata_dict

                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                stored += 1

        return stored
