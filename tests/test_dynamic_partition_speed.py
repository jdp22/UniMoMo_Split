#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset_wrapper import _DynamicFragmentSampler


def test_dynamic_fragment_sampler_speed():
    smiles_list = [
        "c1ccccc1",          # benzene
        "CCO",               # ethanol
        "CC(=O)O",           # acetic acid
        "CCN(CC)CC",         # triethylamine
        "CCC(C)Cl",          # isopropyl chloride
        "OC(=O)C1=CC=CC=C1", # salicylic acid
        "CC(C)C(=O)O",       # isobutyric acid
        "CCCOC(=O)C",        # ethyl acetate
        "CC1=CC=CC=C1",      # ethylbenzene
        "CC(C)OC(=O)N",      # acetamide derivative
    ]

    sampler = _DynamicFragmentSampler({
        'num_parts': [2, 3],
        'min_heavy_atoms': 1,
        'storage_path': None,
        'seed': 123
    })

    tic = time.perf_counter()
    results = [sampler.sample(smi, metadata={'id': idx}) for idx, smi in enumerate(smiles_list)]
    duration = time.perf_counter() - tic

    # Ensure every SMILES returns at least one fragment (fallback included)
    for input_smi, (canonical, fragments) in zip(smiles_list, results):
        assert canonical is not None
        assert isinstance(fragments, list)
        assert len(fragments) >= 1
        print(f"Input: {input_smi:20s} -> canonical: {canonical} | fragments: {fragments}")

    # Cutting 10 small molecules should finish quickly (< 2 seconds on normal CPUs)
    assert duration < 2.0, f"Dynamic partitioning took too long: {duration:.2f}s"
    print(f"Total time for {len(smiles_list)} molecules: {duration:.2f}s")

if __name__ == "__main__":
    test_dynamic_fragment_sampler_speed()