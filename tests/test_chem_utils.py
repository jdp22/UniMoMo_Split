"""Tests for the subgraph partition demo script."""

import json

import pytest
from rdkit import Chem

from scripts.chem.partition_demo import (
    fragment_smiles_for_partitions,
    random_subgraph_partition,
)
from utils.subgraph_storage import SubgraphStorage


def _expected_fragment_smiles(mol: Chem.Mol, partitions):
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


def test_random_subgraph_partition_linear_chain():
    mol = Chem.MolFromSmiles("CCCC")
    partitions = random_subgraph_partition(mol, 2, seed=0)
    assert partitions == [[0, 1], [2, 3]]
    fragments = fragment_smiles_for_partitions(mol, partitions)
    assert fragments == _expected_fragment_smiles(mol, partitions)


def test_random_subgraph_partition_ring_preserved():
    mol = Chem.MolFromSmiles("c1ccccc1")
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 2, seed=13)

    partitions = random_subgraph_partition(mol, 1, seed=13)
    assert partitions == [[0, 1, 2, 3, 4, 5]]
    fragments = fragment_smiles_for_partitions(mol, partitions)
    assert fragments == _expected_fragment_smiles(mol, partitions)


def test_random_subgraph_partition_with_explicit_hydrogens():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    partitions = random_subgraph_partition(mol, 2, seed=5)
    assert partitions == [[0, 1, 3, 4], [2]]
    fragments = fragment_smiles_for_partitions(mol, partitions)
    assert fragments == _expected_fragment_smiles(mol, partitions)


def test_random_subgraph_partition_toluene_fragments():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 3, seed=2)

    partitions = random_subgraph_partition(mol, 2, seed=2)
    assert partitions == [[0], [1, 2, 3, 4, 5, 6]]
    fragments = fragment_smiles_for_partitions(mol, partitions)
    assert fragments == _expected_fragment_smiles(mol, partitions)


def test_random_subgraph_partition_multiple_components():
    mol = Chem.MolFromSmiles("CC.CC")
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 1, seed=0)

    partitions = random_subgraph_partition(mol, 2, seed=0)
    assert partitions == [[0, 1], [2, 3]]
    fragments = fragment_smiles_for_partitions(mol, partitions)
    assert fragments == _expected_fragment_smiles(mol, partitions)


def test_random_subgraph_partition_invalid_num_parts():
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 3)
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 0)


def test_subgraph_storage_persists_fragments(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    partitions = [[0, 1], [2]]

    storage_path = tmp_path / "fragments.jsonl"
    storage = SubgraphStorage(storage_path, min_heavy_atoms=0)
    stored = storage.store_partitions(
        mol,
        partitions,
        metadata={"tag": "unit-test"},
    )

    assert stored == 2
    lines = storage_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]

    expected_smiles = fragment_smiles_for_partitions(mol, partitions)
    assert {record["fragment_smiles"] for record in records} == set(expected_smiles)
    assert all(record["source_smiles"] == Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for record in records)
    assert all(record["metadata"] == {"tag": "unit-test"} for record in records)


def test_subgraph_storage_filters_by_heavy_atoms(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    partitions = [[0, 1], [2]]

    storage_path = tmp_path / "filtered.jsonl"
    storage = SubgraphStorage(storage_path, min_heavy_atoms=1)
    stored = storage.store_partitions(mol, partitions)

    assert stored == 1
    lines = storage_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["heavy_atom_count"] == 2
    assert record["fragment_smiles"] in fragment_smiles_for_partitions(mol, partitions)
