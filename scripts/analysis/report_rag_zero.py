#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarise RAG zero-positive samples using stored fragment JSONL."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_zero_ids(path: Path) -> Counter:
    counter: Counter = Counter()
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get('id')
            if sid:
                counter[sid] += 1
    return counter


def _scan_fragments(path: Path, interest: set[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'count': 0,
        'unique_smiles': set(),
        'heavy_atoms': [],
        'examples': [],
    })
    if not interest:
        return {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = rec.get('metadata') or {}
            sid = meta.get('id')
            if sid not in interest:
                continue
            frag = rec.get('fragment_smiles')
            atom_cnt = rec.get('heavy_atom_count')
            info = stats[sid]
            info['count'] += 1
            if frag:
                info['unique_smiles'].add(frag)
                if len(info['examples']) < 5:
                    info['examples'].append(frag)
            if isinstance(atom_cnt, int):
                info['heavy_atoms'].append(atom_cnt)
    for sid, info in stats.items():
        info['unique_smiles'] = sorted(info['unique_smiles'])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect RAG zero-positive cases.')
    parser.add_argument('--zero-log', type=Path, required=True, help='Path to rag_zero_positive.jsonl')
    parser.add_argument('--fragments', type=Path, required=True, help='Path to dynamic fragment JSONL store')
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON output path')
    parser.add_argument('--top', type=int, default=50, help='Print first N detailed entries')
    args = parser.parse_args()

    zero_counter = _read_zero_ids(args.zero_log)
    if not zero_counter:
        print('No zero-positive entries found.')
        return

    stats = _scan_fragments(args.fragments, set(zero_counter.keys()))

    global_stats = {
        'total_zero_samples': len(zero_counter),
        'total_zero_events': sum(zero_counter.values()),
        'with_fragment_records': sum(1 for sid in zero_counter if sid in stats),
        'missing_fragment_records': [sid for sid in zero_counter if sid not in stats],
    }

    print('Summary:')
    for key, val in global_stats.items():
        if key == 'missing_fragment_records':
            print(f'  {key}: {len(val)}')
        else:
            print(f'  {key}: {val}')

    detailed: List[Dict[str, Any]] = []
    for sid, freq in zero_counter.most_common():
        info = stats.get(sid, {})
        heavy_atoms = info.get('heavy_atoms') or []
        small_frags = sum(1 for h in heavy_atoms if h <= 2)
        record = {
            'id': sid,
            'zero_events': freq,
            'stored_fragment_count': info.get('count', 0),
            'unique_fragment_count': len(info.get('unique_smiles', [])),
            'examples': info.get('examples', []),
            'num_small_fragments(<=2 heavy atoms)': small_frags,
            'heavy_atom_counts': heavy_atoms[:20],
        }
        detailed.append(record)

    limit = max(1, args.top)
    print(f'\nTop {limit} zero-positive entries:')
    for rec in detailed[:limit]:
        print('-' * 60)
        print(json.dumps(rec, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'summary': global_stats,
            'details': detailed,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
