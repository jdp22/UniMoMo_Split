#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from copy import deepcopy
from typing import List

import yaml
import torch
from rdkit import Chem
import numpy as np

import models
from utils.config_utils import overwrite_values
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse import Complex, Block, Atom, VOCAB, BondType
from data.base import Summary, transform_data
from data import create_dataloader, create_dataset
from utils.logger import print_log
from utils.random_seed import setup_seed
from models.LDM.data_utils import Recorder, OverwriteTask, _get_item


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    # Recursively move any tensors (or collections of tensors) onto the
    # desired device, keeping the original data structure intact.
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    # some models (e.g. diffab) will output very large coordinates (absolute value >1000) which will corrupt the pdb file
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def generate_wrapper(model, sample_opt=None):
    if isinstance(model, models.CondIterAutoEncoder):
        def wrapper(batch):
            opt = dict(sample_opt) if sample_opt else {}
            mode = opt.pop('mode', 'codesign')
            fixseq = (mode == 'fixseq')
            n_iter = int(opt.pop('n_iter', 10))
            return_x_only = bool(opt.pop('return_x_only', False))
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = model.generate(
                fixseq=fixseq,
                n_iter=n_iter,
                return_x_only=return_x_only,
                **batch,
                **opt
            )
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    elif isinstance(model, models.LDMMolDesign):# or isinstance(model, models.LFMMolDesign):
        def wrapper(batch):
            # Sampling APIs may return different tuple lengths depending on
            # whether intra-bond predictions are supported, so we normalize the
            # outputs to a consistent 6-tuple.
            res_tuple = model.sample(sample_opt=sample_opt, **batch)
            if len(res_tuple) == 6:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = res_tuple
            else:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds = res_tuple
                batch_intra_bonds = []
                for s in batch_S:
                    batch_intra_bonds.append([None for _ in s])
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    else:
        raise NotImplementedError(f'Wrapper for {type(model)} not implemented')
    return wrapper


def overwrite(cplx: Complex, summary: Summary, S: list, X: list, A: list, ll: list, bonds: tuple, intra_bonds: list, out_path: str, check_validity: bool=True, expect_atom_num=None, fragment_smiles=None, ground_truth_fragments=None):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    task = OverwriteTask(
        cplx = cplx,
        select_indexes = summary.select_indexes,
        generate_mask = summary.generate_mask,
        target_chain_ids = summary.target_chain_ids,
        ligand_chain_ids = summary.ligand_chain_ids,
        S = S,
        X = X,
        A = A,
        ll = ll,
        inter_bonds = bonds,
        intra_bonds = intra_bonds,
        out_path = out_path,
        fragment_smiles = fragment_smiles
    )

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        expect_atom_num = expect_atom_num
    )

    if cplx is None or gen_mol is None:
        return None

    retrieved_fragments = task.get_generated_seq()
    if isinstance(fragment_smiles, (list, tuple)):
        retrieved_fragments = list(fragment_smiles)
    elif isinstance(retrieved_fragments, tuple):
        retrieved_fragments = list(retrieved_fragments)
    elif isinstance(retrieved_fragments, list):
        retrieved_fragments = list(retrieved_fragments)
    result = {
        'id': summary.id,
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': retrieved_fragments,
        'fragment_smiles': retrieved_fragments,
        'target_chains_ids': summary.target_chain_ids,
        'ligand_chains_ids': summary.ligand_chain_ids,
        'gen_block_idx': overwrite_indexes, # TODO: in pdb, (1, '0') will be saved as (1, 'A')
        'gen_pdb': os.path.abspath(out_path),
        'ref_pdb': os.path.abspath(summary.ref_pdb),
    }
    if ground_truth_fragments is not None:
        gt_list = list(ground_truth_fragments)
        result['ground_truth_fragment_smiles'] = gt_list
        result['gen_seq'] = gt_list
    else:
        result['ground_truth_fragment_smiles'] = None
    result['ground_truth_smiles'] = getattr(summary, 'ref_seq', None)
    return result


def format_id(summary: Summary):
    # format saving id for cross dock
    # e.g. BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb|BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf
    if '|' in summary.id:
        summary.id = summary.id.split('|')[0].strip('.pdb')


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    mode = config.get('sample_opt', {}).get('mode', 'codesign')
    struct_only = mode == 'fixseq'
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    
    # save path
    if args.save_dir is None:
        save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir
    ref_save_dir = os.path.join(save_dir, 'references')
    cand_save_dir = os.path.join(save_dir, 'candidates')
    tmp_cand_save_dir = os.path.join(save_dir, 'tmp_candidates')
    for directory in [ref_save_dir, cand_save_dir, tmp_cand_save_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    

    n_samples = config.get('n_samples', 1)
    n_cycles = config.get('n_cycles', 0)

    recorder = Recorder(test_set, n_samples, save_dir)
    
    batch_size = config['dataloader']['batch_size']

    total_blocks = 0
    total_hits = 0

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = [test_set[i] for i, _ in batch_list]
        batch = test_set.collate_fn(batch)
        batch = to_device(batch, device)
        
        with torch.no_grad():
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)
        fragment_payload = getattr(model, 'last_fragment_smiles', None)

        vae_batch_list = []
        for sample_idx, (S, X, A, ll, bonds, intra_bonds, (item_idx, n)) in enumerate(zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list)):
            fragment_smiles = None
            ground_truth_fragments = None
            if isinstance(fragment_payload, list) and sample_idx < len(fragment_payload):
                payload_entry = fragment_payload[sample_idx]
                if isinstance(payload_entry, dict):
                    fragment_smiles = payload_entry.get('retrieved')
                    ground_truth_fragments = payload_entry.get('ground_truth')
                else:
                    fragment_smiles = payload_entry
            if isinstance(fragment_smiles, (list, tuple)):
                fragment_smiles = list(fragment_smiles)
            if isinstance(ground_truth_fragments, tuple):
                ground_truth_fragments = list(ground_truth_fragments)
            elif isinstance(ground_truth_fragments, list):
                ground_truth_fragments = list(ground_truth_fragments)
            cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
            summary: Summary = deepcopy(test_set.get_summary(item_idx))
            # revise id
            format_id(summary)
            summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
            if n == 0: # the first round
                os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
                complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
                os.makedirs(os.path.join(cand_save_dir, summary.id), exist_ok=True)
                os.makedirs(os.path.join(tmp_cand_save_dir, summary.id), exist_ok=True)
                complex_to_pdb(cplx, os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'), summary.target_chain_ids)
            if n_cycles == 0: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
            else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}.pdb')
            log = overwrite(
                cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path,
                check_validity=False, fragment_smiles=fragment_smiles, ground_truth_fragments=ground_truth_fragments
            )
            stats = getattr(model, 'last_retrieval_stats', None)
            if stats:
                total_blocks += stats.get('evaluated_blocks', 0)
                total_hits += stats.get('correct_selections', 0)
                if log is not None:
                    log.setdefault('retrieval_stats', {
                        'evaluated_blocks': 0,
                        'correct_selections': 0,
                        'accuracy': 0.0
                    })
                    log['retrieval_stats'] = {
                        'evaluated_blocks': stats.get('evaluated_blocks', 0),
                        'correct_selections': stats.get('correct_selections', 0),
                        'accuracy': stats.get('accuracy', 0.0)
                    }
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n, struct_only)
            else:
                vae_batch_list.append(
                    _get_item(
                        os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
                        save_path.rstrip('.pdb') + '.sdf',
                        summary.target_chain_ids
                    )
                )

        for cyc_i in range(n_cycles):
            print_log(f'Cycle: {cyc_i}', level='DEBUG')
            final_cycle = cyc_i == n_cycles - 1
            batch = test_set.collate_fn(vae_batch_list)
            batch = to_device(batch, device)
            vae_batch_list = []
            model_autoencoder = getattr(model, 'autoencoder', model)
            with torch.no_grad():
                if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(model_autoencoder, deepcopy(config.get('sample_opt', {})))(batch)
            fragment_payload = getattr(model_autoencoder, 'last_fragment_smiles', None)
            for sample_idx, (S, X, A, ll, bonds, intra_bonds, (item_idx, n)) in enumerate(zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list)):
                fragment_smiles = None
                ground_truth_fragments = None
                if isinstance(fragment_payload, list) and sample_idx < len(fragment_payload):
                    payload_entry = fragment_payload[sample_idx]
                    if isinstance(payload_entry, dict):
                        fragment_smiles = payload_entry.get('retrieved')
                        ground_truth_fragments = payload_entry.get('ground_truth')
                    else:
                        fragment_smiles = payload_entry
                if isinstance(fragment_smiles, (list, tuple)):
                    fragment_smiles = list(fragment_smiles)
                if isinstance(ground_truth_fragments, tuple):
                    ground_truth_fragments = list(ground_truth_fragments)
                elif isinstance(ground_truth_fragments, list):
                    ground_truth_fragments = list(ground_truth_fragments)
                cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
                summary: Summary = deepcopy(test_set.get_summary(item_idx))
                # revise id
                format_id(summary)
                summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
                if final_cycle: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
                else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}_cyc{cyc_i}.pdb')
                # get expect atom number
                if hasattr(test_set, 'get_expected_atom_num'):
                    expect_atom_num = test_set.get_expected_atom_num(item_idx)
                else: expect_atom_num = None
                log = overwrite(
                    cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path,
                    check_validity=final_cycle, expect_atom_num=expect_atom_num,
                    fragment_smiles=fragment_smiles, ground_truth_fragments=ground_truth_fragments
                )
                stats = getattr(model_autoencoder, 'last_retrieval_stats', None)
                if stats:
                    total_blocks += stats.get('evaluated_blocks', 0)
                    total_hits += stats.get('correct_selections', 0)
                    if log is not None:
                        log.setdefault('retrieval_stats', {
                            'evaluated_blocks': 0,
                            'correct_selections': 0,
                            'accuracy': 0.0
                        })
                        log['retrieval_stats'] = {
                            'evaluated_blocks': stats.get('evaluated_blocks', 0),
                            'correct_selections': stats.get('correct_selections', 0),
                            'accuracy': stats.get('accuracy', 0.0)
                        }
                if final_cycle: recorder.check_and_save(log, item_idx, n, struct_only)
                else:
                    vae_batch_list.append(
                        _get_item(
                            os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
                            save_path.rstrip('.pdb') + '.sdf',
                            summary.target_chain_ids
                        )
                    )

        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    if total_blocks:
        accuracy = total_hits / total_blocks if total_blocks else 0.0
        print_log(f'Retrieval stats - total blocks: {total_blocks}, correct: {total_hits}, accuracy: {accuracy:.4f}', level='INFO')
    return    


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
