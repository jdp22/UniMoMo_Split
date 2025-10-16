#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from .nn import GINEConv


def _rdkit_bond_type_to_id(b):
    # Map RDKit bond types to {0: NONE, 1: SINGLE, 2: DOUBLE, 3: TRIPLE, 4: AROMATIC}
    if b is None:
        return 0
    t = b.GetBondType()
    if t == Chem.rdchem.BondType.SINGLE:
        return 1
    elif t == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif t == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif t == Chem.rdchem.BondType.AROMATIC:
        return 4
    return 0


def _mol_to_graph(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an RDKit molecule to tensors.

    Returns:
        atom_types: [N]
        edges: [2, E] (bidirectional)
        edge_types: [E]
    """
    N = mol.GetNumAtoms()
    atom_types = []
    for i in range(N):
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        atom_types.append(z)
    rows, cols, etypes = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _rdkit_bond_type_to_id(b)
        rows += [i, j]
        cols += [j, i]
        etypes += [t, t]
    atom_types = torch.tensor(atom_types, dtype=torch.long)
    edges = torch.tensor([rows, cols], dtype=torch.long) if len(rows) else torch.zeros(2, 0, dtype=torch.long)
    edge_types = torch.tensor(etypes, dtype=torch.long) if len(etypes) else torch.zeros(0, dtype=torch.long)
    return atom_types, edges, edge_types


def _smiles_to_mol_relaxed(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES without forcing Kekulization to avoid RDKit warnings on fragments."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception:
        return None
    return mol


class Subgraph2DGNNEncoder(nn.Module):
    """Lightweight 2D GNN encoder for molecular subgraphs.

    Builds small graphs via RDKit and encodes them using GINEConv layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 3,
        atom_vocab_size: int = 120,
        bond_vocab_size: int = 5,
        readout: str = 'mean',
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_size)
        self.bond_embed = nn.Embedding(bond_vocab_size, hidden_size)
        self.convs = nn.ModuleList([
            GINEConv(hidden_size, hidden_size, hidden_size, hidden_size, n_layers=2)
            for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.readout = readout

    def encode_mol(self, mol: Chem.Mol, device: Optional[torch.device] = None) -> torch.Tensor:
        atom_types, edges, edge_types = _mol_to_graph(mol)
        if device is not None:
            atom_types = atom_types.to(device)
            edges = edges.to(device)
            edge_types = edge_types.to(device)
        H = self.atom_embed(torch.clamp(atom_types, max=self.atom_embed.num_embeddings - 1))
        E = edges
        A = self.bond_embed(torch.clamp(edge_types, max=self.bond_embed.num_embeddings - 1))
        for conv in self.convs:
            H = conv(H, E, A)
        if H.shape[0] == 0:
            H = torch.zeros(1, self.hidden_size, device=H.device)
        if self.readout == 'mean':
            g = H.mean(dim=0, keepdim=True)
        else:
            g = H.max(dim=0, keepdim=True).values
        return self.proj(g).squeeze(0)

    def encode_smiles_list(self, smiles_list: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        embs = []
        for smi in smiles_list:
            mol = _smiles_to_mol_relaxed(smi)
            if mol is None:
                embs.append(torch.zeros(self.hidden_size, device=device))
                continue
            embs.append(self.encode_mol(mol, device=device))
        return torch.stack(embs, dim=0)
