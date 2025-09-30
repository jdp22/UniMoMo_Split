"""Command-line demo for ``random_subgraph_partition``."""

import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

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

    The implementation respects aromatic ring systems, ensuring that every atom
    belonging to a fused ring set is assigned to the same partition. This keeps
    chemically intuitive fragments (e.g., benzene rings) intact instead of
    arbitrarily splitting the ring across multiple fragments.

    Args:
        mol_or_smiles: The input molecule or SMILES string.
        num_partitions: Number of partitions to generate. Must be in
            ``[1, mol.GetNumAtoms()]`` and cannot exceed the number of available
            ring systems plus the number of non-ring atoms.
        seed: Optional random seed to make the partitioning deterministic.

    Returns:
        A list of atom index groups. Each group contains atom indices that are
        connected within the input molecule.

    Raises:
        ValueError: If the number of partitions is invalid, violates the ring
            preservation constraint, or the SMILES string cannot be parsed.
    """

    # Example: ``random_subgraph_partition("Cc1ccccc1", 3, seed=7)`` returns
    # ``[[0], [1, 2, 3, 4, 5, 6], [7]]``—the two methyl carbons are separated
    # while the aromatic ring remains intact as a single fused system.

    mol = _ensure_mol(mol_or_smiles)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return []

    if num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer")

    ring_systems = _ring_systems(mol)
    ring_atom_ids = {atom_idx for system in ring_systems for atom_idx in system}
    non_ring_atoms = [idx for idx in range(num_atoms) if idx not in ring_atom_ids]

    max_partitions = len(non_ring_atoms) + len(ring_systems)
    if num_partitions > max_partitions:
        raise ValueError(
            "num_partitions cannot exceed the number of ring systems plus non-ring atoms"
        )
    if num_partitions > num_atoms:
        raise ValueError(
            f"num_partitions ({num_partitions}) cannot exceed number of atoms ({num_atoms})"
        )

    unit_partitions = _random_partition_units(
        mol,
        ring_systems=ring_systems,
        non_ring_atoms=non_ring_atoms,
        num_partitions=num_partitions,
        seed=seed,
    )

    normalized_partitions = [sorted(group) for group in unit_partitions if group]
    normalized_partitions.sort(key=lambda group: (group[0], len(group)))
    return normalized_partitions


def _ring_systems(mol: Chem.Mol) -> List[Set[int]]:
    """Return fused ring systems as disjoint atom index sets."""

    ring_info = Chem.GetSymmSSSR(mol)
    if not ring_info:
        return []

    parent = list(range(len(ring_info)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    rings = [set(map(int, ring)) for ring in ring_info]
    for i, ring_i in enumerate(rings):
        for j in range(i + 1, len(rings)):
            if ring_i & rings[j]:
                union(i, j)

    systems: Dict[int, Set[int]] = defaultdict(set)
    for idx, ring in enumerate(rings):
        systems[find(idx)].update(ring)

    return list(systems.values())


def _random_partition_units(
    mol: Chem.Mol,
    ring_systems: Sequence[Set[int]],
    non_ring_atoms: Sequence[int],
    num_partitions: int,
    seed: Optional[int],
) -> List[List[int]]:
    """Partition a molecule while keeping each ring system intact."""

    units: List[Set[int]] = [set(system) for system in ring_systems]
    units.extend([{atom_idx} for atom_idx in non_ring_atoms])

    if not units:
        return []

    atom_to_unit: Dict[int, int] = {}
    for unit_idx, atom_set in enumerate(units):
        for atom in atom_set:
            atom_to_unit[atom] = unit_idx

    neighbor_map: Dict[int, Set[int]] = {idx: set() for idx in range(len(units))}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        unit_a = atom_to_unit[begin]
        unit_b = atom_to_unit[end]
        if unit_a == unit_b:
            continue
        neighbor_map[unit_a].add(unit_b)
        neighbor_map[unit_b].add(unit_a)

    unit_partitions = _random_partition_graph(
        neighbor_map={idx: sorted(neighbors) for idx, neighbors in neighbor_map.items()},
        num_partitions=num_partitions,
        seed=seed,
    )

    return [
        sorted({atom for unit_idx in group for atom in units[unit_idx]})
        for group in unit_partitions
    ]


def _random_partition_graph(
    neighbor_map: Dict[int, Iterable[int]], num_partitions: int, seed: Optional[int]
) -> List[List[int]]:
    """Partition a generic graph described by ``neighbor_map``."""

    node_indices = sorted(neighbor_map)
    if not node_indices:
        return []

    num_nodes = len(node_indices)
    if num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer")
    if num_partitions > num_nodes:
        raise ValueError(
            f"num_partitions ({num_partitions}) cannot exceed number of nodes ({num_nodes})"
        )

    rng = random.Random(seed)

    adjacency: Dict[int, List[int]] = {
        node: sorted(set(neighbor_map[node])) for node in node_indices
    }

    components = _connected_components(adjacency)
    if num_partitions < len(components):
        raise ValueError(
            "num_partitions must be at least the number of connected components"
        )

    seeds: List[int] = []
    for component in components:
        seeds.append(rng.choice(sorted(component)))

    remaining_nodes = [node for node in node_indices if node not in seeds]
    additional = num_partitions - len(seeds)
    if additional > 0:
        seeds.extend(rng.sample(remaining_nodes, additional))

    partitions: List[List[int]] = [[seed_node] for seed_node in seeds]
    assigned = {seed_node: part_idx for part_idx, seed_node in enumerate(seeds)}

    frontier: List[Set[int]] = [set(adjacency[idx]) for idx in seeds]

    while len(assigned) < num_nodes:
        progress_made = False
        for part_idx in range(num_partitions):
            if len(assigned) == num_nodes:
                break

            candidates = [idx for idx in frontier[part_idx] if idx not in assigned]
            if not candidates:
                candidates = [
                    nbr
                    for node_idx in partitions[part_idx]
                    for nbr in adjacency[node_idx]
                    if nbr not in assigned
                ]

            if not candidates:
                continue

            chosen = rng.choice(candidates)
            partitions[part_idx].append(chosen)
            assigned[chosen] = part_idx
            frontier[part_idx].update(adjacency[chosen])
            progress_made = True

        if progress_made:
            continue

        remaining = sorted(node for node in node_indices if node not in assigned)
        if not remaining:
            break

        for node_idx in remaining:
            adjacent_parts = {
                assigned[neighbor]
                for neighbor in adjacency[node_idx]
                if neighbor in assigned
            }
            if not adjacent_parts:
                raise ValueError(
                    "Unable to assign node while preserving connectivity."
                )
            part_idx = min(adjacent_parts, key=lambda i: len(partitions[i]))
            partitions[part_idx].append(node_idx)
            assigned[node_idx] = part_idx
            frontier[part_idx].update(adjacency[node_idx])

    normalized_partitions = [sorted(group) for group in partitions if group]
    normalized_partitions.sort(key=lambda group: (group[0], len(group)))
    return normalized_partitions


def _connected_components(adjacency: Dict[int, Sequence[int]]) -> List[Set[int]]:
    """Return connected components from an adjacency mapping."""

    visited: Set[int] = set()
    components: List[Set[int]] = []
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def fragment_smiles_for_partitions(mol: Chem.Mol, partitions: List[List[int]]) -> List[str]:
    """Generate canonical fragment SMILES for each partition."""

    if not partitions:
        return []

    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    return [
        Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=group,
            canonical=True,
            isomericSmiles=True,
            allHsExplicit=has_explicit_h,
        )
        for group in partitions
    ]


def fragment_smiles_with_attachment_points(
    mol: Chem.Mol, partitions: List[List[int]]
) -> List[str]:
    """Generate fragment SMILES that expose inter-fragment attachment points.

    Each returned SMILES string mirrors the canonical fragment representation but
    replaces bonds that cross partition boundaries with wildcards (``*``).
    This mirrors the common convention for denoting open valences in fragment
    libraries.
    """

    if not partitions:
        return []

    smiles_with_markers: List[str] = []
    for group in partitions:
        atoms_in_group = set(group)
        boundary_bonds: List[int] = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            in_begin = begin in atoms_in_group
            in_end = end in atoms_in_group
            if in_begin ^ in_end:
                boundary_bonds.append(bond.GetIdx())

        if boundary_bonds:
            working_mol = Chem.FragmentOnBonds(mol, boundary_bonds, addDummies=True)
        else:
            working_mol = mol

        atoms_to_use: Set[int] = set(group)

        for atom in working_mol.GetAtoms():
            if (
                boundary_bonds
                and atom.GetAtomicNum() == 0
                and any(neighbor.GetIdx() in atoms_in_group for neighbor in atom.GetNeighbors())
            ):
                atoms_to_use.add(atom.GetIdx())
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")

        smiles = Chem.MolFragmentToSmiles(
            working_mol,
            atomsToUse=sorted(atoms_to_use),
            canonical=True,
            isomericSmiles=True,
        )
        smiles_with_markers.append(smiles.replace("[*]", "*"))

    return smiles_with_markers


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
    parser.add_argument(
        "--show-smiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include fragment SMILES in the JSON output (default: True).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
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

    payload = {"partitions": partitions}
    if args.show_smiles:
        payload["fragments"] = fragment_smiles_for_partitions(mol, partitions)
        payload["fragments_with_attachment_points"] = fragment_smiles_with_attachment_points(
            mol, partitions
        )

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

