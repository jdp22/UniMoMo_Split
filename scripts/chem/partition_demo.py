"""Command-line demo for ``random_subgraph_partition``."""

import argparse
import json
import random
import sys
from typing import List, Optional, Set, Union

from rdkit import Chem


def _ensure_mol(mol_or_smiles: Union[str, Chem.Mol]) -> Chem.Mol:
    """Convert SMILES to ``Chem.Mol`` if necessary."""

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles!r}")
        return mol
    return mol_or_smiles


def random_subgraph_partition(
    mol_or_smiles: Union[str, Chem.Mol],
    num_partitions: int,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Randomly partition a molecule into connected subgraphs.

    Args:
        mol_or_smiles: The input molecule or SMILES string.
        num_partitions: Number of partitions to generate. Must be in
            ``[1, mol.GetNumAtoms()]``.
        seed: Optional random seed to make the partitioning deterministic.

    Returns:
        A list of atom index groups. Each group contains atom indices that are
        connected within the input molecule.

    Raises:
        ValueError: If the number of partitions is invalid or the SMILES string
            cannot be parsed.
    """

    mol = _ensure_mol(mol_or_smiles)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return []

    if num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer")
    if num_partitions > num_atoms:
        raise ValueError(
            f"num_partitions ({num_partitions}) cannot exceed number of atoms ({num_atoms})"
        )

    rng = random.Random(seed)
    atom_indices = list(range(num_atoms))
    seeds = rng.sample(atom_indices, num_partitions)

    partitions: List[List[int]] = [[seed_idx] for seed_idx in seeds]
    assigned = {seed_idx: part_idx for part_idx, seed_idx in enumerate(seeds)}

    neighbor_map = {
        atom_idx: [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
        for atom_idx in atom_indices
    }

    frontier: List[Set[int]] = [set(neighbor_map[idx]) for idx in seeds]

    while len(assigned) < num_atoms:
        progress_made = False
        for part_idx in range(num_partitions):
            if len(assigned) == num_atoms:
                break

            candidates = [idx for idx in frontier[part_idx] if idx not in assigned]
            if not candidates:
                candidates = [
                    nbr
                    for atom_idx in partitions[part_idx]
                    for nbr in neighbor_map[atom_idx]
                    if nbr not in assigned
                ]

            if not candidates:
                continue

            chosen = rng.choice(candidates)
            partitions[part_idx].append(chosen)
            assigned[chosen] = part_idx
            frontier[part_idx].update(neighbor_map[chosen])
            progress_made = True

        if progress_made:
            continue

        remaining = sorted(idx for idx in atom_indices if idx not in assigned)
        for atom_idx in remaining:
            part_idx = min(range(num_partitions), key=lambda i: len(partitions[i]))
            partitions[part_idx].append(atom_idx)
            assigned[atom_idx] = part_idx
        break

    normalized_partitions = [sorted(group) for group in partitions if group]
    normalized_partitions.sort(key=lambda group: (group[0], len(group)))
    return normalized_partitions


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition a molecule into random connected subgraphs."
    )
    parser.add_argument(
        "smiles",
        help="Input SMILES string to be partitioned.",
    )
    parser.add_argument(
        "-k",
        "--partitions",
        type=int,
        default=2,
        help="Number of partitions to generate (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {args.smiles}", file=sys.stderr)
        return 1

    try:
        partitions = random_subgraph_partition(mol, args.partitions, seed=args.seed)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(partitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

