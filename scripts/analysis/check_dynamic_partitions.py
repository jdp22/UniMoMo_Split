#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect dynamic ligand partition quality on a dataset split."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import importlib.machinery
import importlib.util
import types

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_module(name: str, path: Path, *, aliases: tuple[str, ...] = ()):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    for alias in aliases:
        sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        for alias in aliases:
            sys.modules.pop(alias, None)
        raise
    return module

register_module = _load_module('analysis_register', REPO_ROOT / 'utils' / 'register.py')
logger_module = _load_module('analysis_logger', REPO_ROOT / 'utils' / 'logger.py')

stub_subgraph_module = types.ModuleType('analysis_subgraph_storage')

class _StubSubgraphStorage:
    def __init__(self, *args, **kwargs):
        self.enabled = False

    def store_partitions(self, *args, **kwargs):
        return None

def _stub_heavy_atom_count(*args, **kwargs):
    return 0

stub_subgraph_module.SubgraphStorage = _StubSubgraphStorage
stub_subgraph_module._heavy_atom_count = _stub_heavy_atom_count

try:
    chem_module = _load_module('analysis_chem_utils', REPO_ROOT / 'utils' / 'chem_utils.py')
except Exception as exc:  # pragma: no cover - fallback keeps helpful error message
    stub_chem_module = types.ModuleType('analysis_chem_utils')

    def _missing_chem_fn(*_args, **_kwargs):
        raise RuntimeError(
            'RDKit chem utilities unavailable while running analysis script. '
            'Install rdkit or use an environment with full chemistry dependencies.'
        ) from exc

    stub_chem_module.smi2mol = _missing_chem_fn
    stub_chem_module.mol2smi = _missing_chem_fn
    stub_chem_module.get_submol = _missing_chem_fn
    stub_chem_module.get_submol_atom_map = _missing_chem_fn
    chem_module = stub_chem_module

utils_pkg = types.ModuleType('utils')
utils_pkg.register = register_module
utils_pkg.subgraph_storage = stub_subgraph_module
utils_pkg.chem_utils = chem_module
utils_pkg.logger = logger_module
singleton_module = _load_module('analysis_singleton', REPO_ROOT / 'utils' / 'singleton.py')
utils_pkg.singleton = singleton_module.singleton
sys.modules['utils'] = utils_pkg
sys.modules['utils.register'] = register_module
sys.modules['utils.subgraph_storage'] = stub_subgraph_module
sys.modules['utils.chem_utils'] = chem_module
sys.modules['utils.logger'] = logger_module
sys.modules['utils.singleton'] = singleton_module

data_pkg = types.ModuleType('data')
data_pkg.__path__ = [str(REPO_ROOT / 'data')]
data_pkg.__package__ = 'data'
data_pkg.__spec__ = importlib.machinery.ModuleSpec('data', loader=None, is_package=True)
sys.modules['data'] = data_pkg

bioparse_module = _load_module(
    'data.bioparse',
    REPO_ROOT / 'data' / 'bioparse' / '__init__.py',
    aliases=('analysis_data_bioparse',),
)
data_pkg.bioparse = bioparse_module

peptide_module = _load_module(
    'data.peptide',
    REPO_ROOT / 'data' / 'peptide.py',
    aliases=('analysis_data_peptide',),
)
data_pkg.peptide = peptide_module

molecule_module = _load_module(
    'data.molecule',
    REPO_ROOT / 'data' / 'molecule.py',
    aliases=('analysis_data_molecule',),
)
data_pkg.molecule = molecule_module

R = utils_pkg.register

dataset_module = _load_module(
    'analysis_dataset_wrapper',
    REPO_ROOT / 'data' / 'dataset_wrapper.py',
    aliases=('data.dataset_wrapper',),
)
data_pkg.dataset_wrapper = dataset_module


def _load_dataset(config_path: Path, split: str, index: int):
    cfg = yaml.safe_load(config_path.read_text())
    split_cfg = cfg['dataset'].get(split)
    if split_cfg is None:
        raise ValueError(f"Split '{split}' not found in config {config_path}")
    node = split_cfg
    if isinstance(node, list):
        if index >= len(node):
            raise IndexError(f"Split '{split}' has only {len(node)} entries (index {index} requested)")
        node = node[index]
    elif index != 0:
        raise IndexError(f"Split '{split}' is not a list; index must be 0")

    while isinstance(node, dict) and node.get('class') == 'DynamicBatchWrapper' and 'dataset' in node:
        node = node['dataset']
    if not isinstance(node, dict) or 'class' not in node:
        raise ValueError('Malformed dataset specification; missing class field')
    dataset = R.construct(node)
    return dataset


def _fragment_statistics(sample: Any, *, collect_details: bool = False) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if isinstance(sample, tuple):
        data = sample[0]
        sample_id = sample[1]
    else:
        data = sample
        sample_id = data.get('sample_ids', [None])[0] if isinstance(data, dict) else None
    dyn = None
    if isinstance(data, dict):
        dyn = data.get('_dyn_info') or data
    if not dyn:
        info = {'id': sample_id, 'total_frag': 0, 'valid_frag': 0}
        details = {'id': sample_id, 'fragments': []} if collect_details else None
        return info, details

    gen_mask = dyn.get('dyn_generate_mask')
    if gen_mask is None:
        info = {'id': sample_id, 'total_frag': 0, 'valid_frag': 0}
        details = {'id': sample_id, 'fragments': []} if collect_details else None
        return info, details
    gen_mask = torch.as_tensor(gen_mask, dtype=torch.bool)
    smiles_list = dyn.get('dyn_fragment_smiles', [])
    graphs_list = dyn.get('dyn_fragment_graphs', [])

    total = int(gen_mask.long().sum().item())
    valid = 0
    fragments = [] if collect_details else None
    for idx, is_gen in enumerate(gen_mask.tolist()):
        if not is_gen:
            continue
        smi_ok = False
        graph_ok = False
        graph_atom_count = 0
        if idx < len(smiles_list):
            smi = smiles_list[idx]
            smi_ok = bool(smi)
        if idx < len(graphs_list) and graphs_list[idx] is not None:
            graph = graphs_list[idx]
            atom_types = graph.get('atom_types')
            if isinstance(atom_types, torch.Tensor):
                graph_atom_count = int(atom_types.numel())
                graph_ok = graph_atom_count > 0
            else:
                try:
                    graph_atom_count = len(atom_types)
                except TypeError:
                    graph_atom_count = 0
                graph_ok = bool(atom_types)
        if smi_ok or graph_ok:
            valid += 1
        if collect_details:
            fragments.append(
                {
                    'index': idx,
                    'smiles': smiles_list[idx] if idx < len(smiles_list) else None,
                    'smiles_ok': smi_ok,
                    'graph_present': graph_ok,
                    'graph_atom_count': graph_atom_count,
                }
            )
    info = {'id': sample_id, 'total_frag': total, 'valid_frag': valid}
    details = None
    if collect_details:
        fragments = fragments or []
        details = {
            'id': sample_id,
            'total_slots': int(gen_mask.numel()),
            'fragments': fragments,
        }
    return info, details


def main():
    parser = argparse.ArgumentParser(description='Check dynamic partitions on dataset split')
    parser.add_argument('--config', required=True, type=Path, help='Training config path')
    parser.add_argument('--split', default='train', help='Dataset split name (train/valid/test)')
    parser.add_argument('--index', type=int, default=0, help='Index within split list (for train list entries)')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of samples to inspect')
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON report path')
    parser.add_argument('--dump-fragments', type=Path, default=None, help='Optional per-sample fragment dump path')
    args = parser.parse_args()

    dataset = _load_dataset(args.config, args.split, args.index)
    length = len(dataset)
    limit = min(length, args.limit) if args.limit else length

    stats = []
    zero_valid_ids = []
    single_valid_ids = []
    collect_details = args.dump_fragments is not None
    fragment_dump = [] if collect_details else None

    for idx in tqdm(range(limit), desc='Inspecting partitions'):
        sample = dataset[idx]
        info, details = _fragment_statistics(sample, collect_details=collect_details)
        stats.append(info)
        if info['valid_frag'] == 0:
            zero_valid_ids.append(info['id'])
        elif info['valid_frag'] == 1:
            single_valid_ids.append(info['id'])
        if details is not None:
            fragment_dump.append(details)

    total = len(stats)
    multi_valid = sum(1 for s in stats if s['valid_frag'] > 1)

    summary = {
        'total_samples': total,
        'multi_valid_fragments': multi_valid,
        'single_valid_fragment': len(single_valid_ids),
        'zero_valid_fragment': len(zero_valid_ids),
        'ratio_multi_valid': multi_valid / total if total else 0.0,
    }

    print('\nSummary:')
    for key, val in summary.items():
        print(f'  {key}: {val}')
    if zero_valid_ids:
        print(f'  Zero-valid sample examples (first 10): {zero_valid_ids[:10]}')
    if single_valid_ids:
        print(f'  Single-valid sample examples (first 10): {single_valid_ids[:10]}')

    if args.output:
        report = {
            'summary': summary,
            'zero_valid_ids': zero_valid_ids,
            'single_valid_ids': single_valid_ids,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    if args.dump_fragments and fragment_dump is not None:
        args.dump_fragments.parent.mkdir(parents=True, exist_ok=True)
        args.dump_fragments.write_text(json.dumps(fragment_dump, indent=2))


if __name__ == '__main__':
    main()
