#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Evaluate DynamicBlockWrapper partition success rates for a dataset config.

Examples
--------
python scripts/analysis/check_partition_success.py \
    --config configs/IterAE/train_rag_dynamic_mol.yaml \
    --section dataset --entry train --index 1

python scripts/analysis/check_partition_success.py \
    --config configs/test/test_mol_dynamic.yaml \
    --section dataset --entry test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from utils import register as R  # noqa: E402
from data.dataset_wrapper import FragmentPartitionError  # noqa: E402


def _resolve_cfg(cfg: dict, section: str, entry_key, list_index: Optional[int]):
    if section not in cfg:
        raise KeyError(f"Section '{section}' not found in config.")
    node = cfg[section]
    if isinstance(node, list):
        if entry_key is None:
            raise ValueError("Section is list; --entry must specify an integer index.")
        if not isinstance(entry_key, int):
            raise ValueError("Section is list; --entry must be integer index.")
        if not (0 <= entry_key < len(node)):
            raise IndexError(f"Entry index {entry_key} out of range for section '{section}'.")
        node = node[entry_key]
    elif isinstance(node, dict) and entry_key is not None:
        if entry_key not in node:
            raise KeyError(f"Entry '{entry_key}' not found under section '{section}'.")
        node = node[entry_key]

    if isinstance(node, list):
        if list_index is None:
            raise ValueError("Section->entry resolved to list; please provide --index.")
        if not (0 <= list_index < len(node)):
            raise IndexError(f"--index {list_index} out of range (size {len(node)}).")
        node = node[list_index]

    cfg_node = node
    while isinstance(cfg_node, dict):
        cls = cfg_node.get('class')
        if cls == 'DynamicBlockWrapper':
            return cfg_node
        dataset = cfg_node.get('dataset')
        if dataset is None:
            break
        cfg_node = dataset
    raise ValueError('Could not locate a DynamicBlockWrapper in the provided config path.')


def main():
    parser = argparse.ArgumentParser(description='Check DynamicBlockWrapper partition success rate.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--section', default='dataset', help='Top-level key to inspect (default: dataset).')
    parser.add_argument('--entry', help='When section is dict/list, specify key or index to select.')
    parser.add_argument('--index', type=int, default=None, help='If section->entry resolves to a list, pick which element.')
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of samples to inspect.')
    parser.add_argument('--success-log', type=str, default=None, help='Optional JSONL path to store successful partition details.')
    parser.add_argument('--failure-log', type=str, default=None, help='Optional JSONL path to store failed partition details.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)

    entry_key = None
    if args.entry is not None:
        try:
            entry_key = int(args.entry)
        except ValueError:
            entry_key = args.entry

    cfg = yaml.safe_load(open(args.config, 'r'))
    block_cfg = _resolve_cfg(cfg, args.section, entry_key, args.index)
    dataset = R.construct(block_cfg)

    def _summary_dict(summary):
        if summary is None:
            return None
        return {
            'id': summary.id,
            'ref_seq': summary.ref_seq,
            'target_chain_ids': summary.target_chain_ids,
            'ligand_chain_ids': summary.ligand_chain_ids,
            'select_indexes': list(summary.select_indexes),
            'generate_mask': list(summary.generate_mask),
            'center_mask': list(summary.center_mask),
        }

    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if hasattr(value, 'tolist'):
            try:
                return value.tolist()
            except Exception:
                pass
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _prepare_dyn_info(dyn_info: dict):
        if not dyn_info:
            return {}
        return {
            key: _convert(dyn_info.get(key))
            for key in (
                'dyn_fragment_smiles',
                'dyn_block_lengths',
                'dyn_block_types',
                'dyn_generate_mask',
                'dyn_block_ids',
                'dyn_chain_ids',
                'dyn_position_ids',
                'dyn_is_aa',
                'dyn_num_blocks',
            )
            if key in dyn_info
        }

    success_log = args.success_log
    failure_log = args.failure_log
    success_handle = None
    failure_handle = None
    try:
        if success_log:
            Path(success_log).parent.mkdir(parents=True, exist_ok=True)
            success_handle = open(success_log, 'a', encoding='utf-8')
        if failure_log:
            Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
            failure_handle = open(failure_log, 'a', encoding='utf-8')

        total = len(dataset)
        limit = min(total, args.limit) if args.limit else total
        success = 0
        failures: list[dict] = []

        current_idx = 0
        processed = 0
        while processed < limit and current_idx < len(dataset):
            summary = dataset.get_summary(current_idx) if hasattr(dataset, 'get_summary') else None
            summary_dict = _summary_dict(summary)
            item_id = summary_dict['id'] if summary_dict else None
            original_smiles = summary_dict['ref_seq'] if summary_dict else None
            try:
                sample = dataset[current_idx]
                dyn_info = sample.get('_dyn_info', {}) if isinstance(sample, dict) else {}
                record = {
                    'index': current_idx,
                    'summary': summary_dict,
                    'fragments': _prepare_dyn_info(dyn_info),
                }
                if success_handle:
                    success_handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                success += 1
                current_idx += 1
            except FragmentPartitionError as exc:
                failure_record = {
                    'index': current_idx,
                    'item_id': exc.item_id or item_id,
                    'smiles': exc.smiles or original_smiles,
                    'message': str(exc),
                    'summary': summary_dict,
                }
                failures.append(failure_record)
                if failure_handle:
                    failure_handle.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
            processed += 1

        checked = success + len(failures)
        summary_lines = [
            f'Checked {checked} samples (total requested: {limit}, available: {total})',
            f'  success: {success}',
            f'  failed : {len(failures)}',
        ]
        if failures:
            print('First few failures:')
            for record in failures[:10]:
                print(' ', json.dumps(record, ensure_ascii=False))
        for line in summary_lines:
            print(line)
    finally:
        if success_handle:
            success_handle.close()
        if failure_handle:
            failure_handle.close()


if __name__ == '__main__':
    main()
