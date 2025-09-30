"""Tests for the subgraph partition demo script."""

import pytest
from rdkit import Chem

from scripts.chem.partition_demo import random_subgraph_partition


def test_random_subgraph_partition_linear_chain():
    mol = Chem.MolFromSmiles("CCCC")
    partitions = random_subgraph_partition(mol, 2, seed=0)
    assert partitions == [[0, 1], [2, 3]]


def test_random_subgraph_partition_ring():
    mol = Chem.MolFromSmiles("c1ccccc1")
    partitions = random_subgraph_partition(mol, 3, seed=13)
    assert partitions == [[0, 5], [1], [2, 3, 4]]


def test_random_subgraph_partition_with_explicit_hydrogens():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    partitions = random_subgraph_partition(mol, 2, seed=5)
    assert partitions == [[0, 1, 3, 4], [2]]


def test_random_subgraph_partition_invalid_num_parts():
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 3)
    with pytest.raises(ValueError):
        random_subgraph_partition(mol, 0)
