#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Evaluate IterVAE reconstruction quality against ground-truth coordinates and optionally export reconstructed PDB files.

Example:
    python scripts/analysis/eval_vae_reconstruction.py \
        --config configs/IterAE/train_rag_dynamic_mol.yaml \
        --checkpoint ckpts/unimomo_rag_dynamic/best.ckpt \
        --split valid \
        --pdb-dir outputs/recon_pdb \
        --save-predictions recon_metrics.jsonl
"""

import argparse
import heapq
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import yaml
try:
    from torch_scatter import scatter_max, scatter_sum
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_scatter is required for eval_vae_reconstruction.py. "
        "Install it via `pip install torch-scatter` matching your torch version."
    ) from exc
from tqdm import tqdm

from data import create_dataset, create_dataloader
from data.bioparse.utils import recur_index
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from utils import register as R
from utils.config_utils import overwrite_values
from utils.gnn_utils import length_to_batch_id
from utils.logger import print_log


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Run VAE reconstruction and compare with ground truth."
    )
    parser.add_argument("--config", required=True, help="Path to training/eval YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint. Falls back to config.load_ckpt when omitted.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        default=device_default,
        help=f"Computation device (default: {device_default}).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Number of iterative refinement steps (defaults to model.default_num_steps).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of ligand samples evaluated.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Optional path to write per-sample metrics as JSONL.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        default=None,
        help="Directory to export reconstructed PDB files (one per sample).",
    )
    parser.add_argument(
        "--write-ground-truth",
        action="store_true",
        help="When exporting PDBs, also write the ground-truth structures for comparison.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Report the K worst ligand RMSDs (set 0 to disable).",
    )
    parser.add_argument(
        "--predict-blocks",
        action="store_true",
        help="Decode block types as well (fixseq=False). Default uses ground-truth block types.",
    )
    args, opt_args = parser.parse_known_args()
    return args, opt_args


def move_to_device(obj: Any, device: Union[torch.device, str]) -> Any:
    """Recursively move tensors to target device while leaving other types untouched."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(config: Dict[str, Any], ckpt_path: Optional[str], device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    model: torch.nn.Module = R.construct(model_cfg)
    weights_path = ckpt_path or config.get("load_ckpt") or ""
    if weights_path:
        print_log(f"Loading checkpoint from {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif hasattr(state, "state_dict"):
            state_dict = state.state_dict()
        else:
            state_dict = state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print_log(f"Missing keys: {missing}", level="WARN")
        if unexpected:
            print_log(f"Unexpected keys: {unexpected}", level="WARN")
    model.to(device)
    model.eval()
    return model


def ensure_tensor(data: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.to(device)
    else:
        tensor = torch.as_tensor(data, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def aggregate_scope(
    dist: torch.Tensor,
    dist_sq: torch.Tensor,
    atom_batch_ids: torch.Tensor,
    mask_atoms: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate per-atom distances into per-sample sums."""
    if mask_atoms is not None:
        if mask_atoms.dtype != torch.bool:
            mask_atoms = mask_atoms.bool()
        valid = mask_atoms & torch.isfinite(dist_sq)
        if not torch.any(valid):
            zeros = torch.zeros(batch_size, dtype=dist.dtype, device=dist.device)
            neg_inf = torch.full((batch_size,), float("-inf"), dtype=dist.dtype, device=dist.device)
            return zeros, zeros, zeros, neg_inf
        dist = dist[valid]
        dist_sq = dist_sq[valid]
        atom_batch_ids = atom_batch_ids[valid]
    if dist.numel() == 0:
        zeros = torch.zeros(batch_size, dtype=dist.dtype, device=dist.device)
        neg_inf = torch.full((batch_size,), float("-inf"), dtype=dist.dtype, device=dist.device)
        return zeros, zeros, zeros, neg_inf
    ones = torch.ones_like(dist, dtype=dist.dtype)
    sum_dist = scatter_sum(dist, atom_batch_ids, dim=0, dim_size=batch_size)
    sum_sq = scatter_sum(dist_sq, atom_batch_ids, dim=0, dim_size=batch_size)
    counts = scatter_sum(ones, atom_batch_ids, dim=0, dim_size=batch_size)
    max_dist, _ = scatter_max(dist, atom_batch_ids, dim=0, dim_size=batch_size)
    return sum_dist, sum_sq, counts, max_dist


def write_record(handle, record: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def update_topk(
    heaps: Dict[str, List[Tuple[float, Dict[str, Any]]]],
    scope: str,
    record: Dict[str, Any],
    k: int,
) -> None:
    if k <= 0:
        return
    entry = (record["rmsd"], record)
    heap = heaps.setdefault(scope, [])
    if len(heap) < k:
        heapq.heappush(heap, entry)
        return
    if entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


class ReconstructionPDBWriter:
    """Map sample IDs back to raw complexes and emit predicted PDB files."""

    def __init__(self, dataset, out_dir: Union[str, Path], write_gt: bool = False) -> None:
        self.dataset = dataset
        self.out_dir = Path(out_dir)
        self.pred_dir = self.out_dir / "pred"
        self.gt_dir = self.out_dir / "gt" if write_gt else None
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        if self.gt_dir is not None:
            self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.write_gt = write_gt
        self.id_lookup = self._build_lookup(dataset)

    def _build_lookup(self, dataset) -> Dict[str, Tuple[Any, int]]:
        lookup: Dict[str, Tuple[Any, int]] = {}
        visited: Set[int] = set()

        def recurse(ds):
            if ds is None:
                return
            obj_id = id(ds)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if hasattr(ds, "datasets"):
                for sub in getattr(ds, "datasets"):
                    recurse(sub)
                return
            if hasattr(ds, "dataset"):
                recurse(getattr(ds, "dataset"))
                return
            if hasattr(ds, "_indexes"):
                for idx, entry in enumerate(ds._indexes):
                    sample_id = entry[0]
                    lookup.setdefault(sample_id, (ds, idx))
            elif hasattr(ds, "get_id"):
                try:
                    total = len(ds)
                except Exception:
                    total = 0
                for idx in range(total):
                    try:
                        sample_id = ds.get_id(idx)
                    except Exception:
                        continue
                    lookup.setdefault(sample_id, (ds, idx))

        recurse(dataset)
        return lookup

    @staticmethod
    def _sanitize(sample_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)

    def _collect_atoms(self, cplx, select_indexes: List[tuple]) -> List[Any]:
        atoms: List[Any] = []
        for block_id in select_indexes:
            block = recur_index(cplx, block_id)
            for atom in block:
                element = getattr(atom, "element", None)
                if element == "H":
                    continue
                atoms.append(atom)
        return atoms

    def _assign_coords(self, atoms: List[Any], coords: torch.Tensor) -> bool:
        if len(atoms) != coords.shape[0]:
            return False
        coords_list = coords.detach().cpu().tolist()
        for atom_obj, coord in zip(atoms, coords_list):
            atom_obj.coordinate = [float(coord[0]), float(coord[1]), float(coord[2])]
        return True

    def write(self, sample_id: str, pred_coords: torch.Tensor, true_coords: Optional[torch.Tensor] = None) -> None:
        entry = self.id_lookup.get(sample_id)
        if entry is None:
            print_log(f"[PDB] sample id '{sample_id}' not found in dataset.", level="WARN")
            return
        dataset, base_idx = entry
        try:
            cplx = deepcopy(dataset.get_raw_data(base_idx))
            summary = dataset.get_summary(base_idx)
        except Exception as exc:
            print_log(f"[PDB] failed to fetch raw data for '{sample_id}': {exc}", level="WARN")
            return
        select_indexes = getattr(summary, "select_indexes", None)
        if not select_indexes:
            print_log(f"[PDB] missing select indexes for '{sample_id}', skip.", level="WARN")
            return
        atoms = self._collect_atoms(cplx, select_indexes)
        if not self._assign_coords(atoms, pred_coords):
            print_log(
                f"[PDB] atom count mismatch for '{sample_id}' "
                f"(atoms={len(atoms)}, coords={pred_coords.shape[0]}), skip.",
                level="WARN",
            )
            return
        chain_ids = list(getattr(summary, "target_chain_ids", [])) + list(getattr(summary, "ligand_chain_ids", []))
        pred_path = self.pred_dir / f"{self._sanitize(sample_id)}.pdb"
        try:
            complex_to_pdb(cplx, str(pred_path), selected_chains=chain_ids)
        except Exception as exc:
            print_log(f"[PDB] failed to write prediction PDB for '{sample_id}': {exc}", level="WARN")
        if self.write_gt and true_coords is not None:
            try:
                cplx_gt = deepcopy(dataset.get_raw_data(base_idx))
                atoms_gt = self._collect_atoms(cplx_gt, select_indexes)
                if self._assign_coords(atoms_gt, true_coords):
                    gt_path = self.gt_dir / f"{self._sanitize(sample_id)}.pdb"
                    complex_to_pdb(cplx_gt, str(gt_path), selected_chains=chain_ids)
            except Exception as exc:
                print_log(f"[PDB] failed to write ground-truth PDB for '{sample_id}': {exc}", level="WARN")


def evaluate(args: argparse.Namespace, opt_args: Optional[List[str]] = None) -> None:
    device = torch.device(args.device)
    config = load_config(args.config)
    if opt_args:
        config = overwrite_values(config, opt_args)
    model = load_model(config, args.checkpoint, device)

    dataset_splits = create_dataset(config["dataset"])
    split_map = {"train": dataset_splits[0], "valid": dataset_splits[1], "test": dataset_splits[2]}
    dataset = split_map.get(args.split)
    if dataset is None:
        raise ValueError(f"Dataset split '{args.split}' is not available in config.")

    dl_cfg_all = config.get("dataloader", {})
    dataloader_cfg = dl_cfg_all.get(args.split, dl_cfg_all)
    dataloader = create_dataloader(dataset, dataloader_cfg, n_gpu=1)

    pdb_writer: Optional[ReconstructionPDBWriter] = None
    if args.pdb_dir:
        try:
            pdb_writer = ReconstructionPDBWriter(dataset, args.pdb_dir, write_gt=args.write_ground_truth)
        except Exception as exc:
            print_log(f"Failed to initialise PDB writer: {exc}", level="WARN")
            pdb_writer = None

    n_iter = args.num_iters or getattr(model, "default_num_steps", 10)
    fixseq = not args.predict_blocks

    summary: Dict[str, Dict[str, float]] = {
        "ligand": {"sum_sq": 0.0, "sum_dist": 0.0, "total_atoms": 0.0, "max_dist": 0.0, "num_samples": 0},
        "all": {"sum_sq": 0.0, "sum_dist": 0.0, "total_atoms": 0.0, "max_dist": 0.0, "num_samples": 0},
    }
    topk_heaps: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {scope: [] for scope in summary}

    save_path = Path(args.save_predictions) if args.save_predictions else None
    if save_path is not None and save_path.parent:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open(save_path, "w", encoding="utf-8") if save_path else None

    global_sample_counter = 0
    evaluated_samples = 0

    try:
        iterable: Iterable = enumerate(tqdm(dataloader, desc="Evaluating", dynamic_ncols=True))
    except TypeError:
        iterable = enumerate(dataloader)

    for batch_idx, batch in iterable:
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        batch = move_to_device(batch, device)
        sample_ids = batch.get("sample_ids")
        lengths_tensor = ensure_tensor(batch["lengths"], device, torch.long)
        if sample_ids is None:
            batch_size = int(lengths_tensor.shape[0])
            sample_ids = [f"{args.split}_{global_sample_counter + i}" for i in range(batch_size)]
        else:
            sample_ids = [
                str(s) if s is not None else f"{args.split}_{global_sample_counter + i}"
                for i, s in enumerate(sample_ids)
            ]
            batch_size = len(sample_ids)

        with torch.no_grad():
            recon_x = model.generate(
                n_iter=n_iter,
                fixseq=fixseq,
                return_x_only=True,
                **batch,
            )

        true_x = batch["X"]
        dist_vec = torch.norm(recon_x - true_x, dim=-1)  # [Natom]
        dist_sq = dist_vec ** 2

        block_lengths = ensure_tensor(batch["block_lengths"], device, torch.long)
        block_ids = length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths_tensor)
        atom_batch_ids = batch_ids[block_ids]
        generate_mask = ensure_tensor(batch["generate_mask"], device, torch.bool)
        ligand_atom_mask = generate_mask[block_ids]

        lengths_list = lengths_tensor.cpu().tolist()
        block_lengths_list = block_lengths.cpu().tolist()
        sample_slices: List[Tuple[int, int, int, int]] = []
        block_ptr = 0
        atom_ptr = 0
        for n_blocks in lengths_list:
            block_end = block_ptr + n_blocks
            atom_end = atom_ptr + sum(block_lengths_list[block_ptr:block_end])
            sample_slices.append((block_ptr, block_end, atom_ptr, atom_end))
            block_ptr = block_end
            atom_ptr = atom_end

        scopes = {
            "ligand": ligand_atom_mask,
            "all": None,
        }

        for scope_name, mask in scopes.items():
            sum_dist, sum_sq, counts, max_dist = aggregate_scope(
                dist_vec, dist_sq, atom_batch_ids, mask, batch_size
            )
            sum_dist_cpu = sum_dist.cpu()
            sum_sq_cpu = sum_sq.cpu()
            counts_cpu = counts.cpu().long()
            max_dist_cpu = max_dist.cpu()
            valid_mask = counts_cpu > 0
            if torch.any(valid_mask):
                summary_scope = summary[scope_name]
                summary_scope["sum_sq"] += float(sum_sq_cpu[valid_mask].sum().item())
                summary_scope["sum_dist"] += float(sum_dist_cpu[valid_mask].sum().item())
                summary_scope["total_atoms"] += float(counts_cpu[valid_mask].sum().item())
                summary_scope["num_samples"] += int(valid_mask.sum().item())
                max_val = float(max_dist_cpu[valid_mask].max().item())
                summary_scope["max_dist"] = max(summary_scope["max_dist"], max_val)

            for idx in torch.nonzero(valid_mask, as_tuple=False).view(-1).tolist():
                atoms = int(counts_cpu[idx].item())
                if atoms <= 0:
                    continue
                rmsd = math.sqrt(sum_sq_cpu[idx].item() / atoms)
                mean_dist = sum_dist_cpu[idx].item() / atoms
                max_dist_val = float(max_dist_cpu[idx].item())
                record = {
                    "scope": scope_name,
                    "sample_id": sample_ids[idx],
                    "sample_index": global_sample_counter + idx,
                    "batch_index": batch_idx,
                    "atoms": atoms,
                    "rmsd": rmsd,
                    "mean_distance": mean_dist,
                    "max_distance": max_dist_val,
                }
                if scope_name == "ligand":
                    evaluated_samples += 1
                write_record(writer, record)
                update_topk(topk_heaps, scope_name, record, args.top_k)

        if pdb_writer is not None:
            for local_idx, sample_id in enumerate(sample_ids):
                _, _, atom_start, atom_end = sample_slices[local_idx]
                pred_coords = recon_x[atom_start:atom_end]
                gt_coords = true_x[atom_start:atom_end] if args.write_ground_truth else None
                try:
                    pdb_writer.write(sample_id, pred_coords, gt_coords)
                except Exception as exc:
                    print_log(f"Failed to write PDB for '{sample_id}': {exc}", level="WARN")

        global_sample_counter += batch_size
        if args.max_samples is not None and evaluated_samples >= args.max_samples:
            break

    if writer is not None:
        writer.close()

    print_log("Evaluation summary:")
    for scope_name, stats in summary.items():
        total_atoms = stats["total_atoms"]
        if total_atoms <= 0:
            print_log(f"  {scope_name}: no atoms to evaluate.", level="WARN")
            continue
        global_rmsd = math.sqrt(stats["sum_sq"] / total_atoms)
        mean_dist = stats["sum_dist"] / total_atoms
        print_log(
            f"  {scope_name}: RMSD={global_rmsd:.4f} Å | mean distance={mean_dist:.4f} Å | "
            f"atoms={int(total_atoms)} | samples={stats['num_samples']} | "
            f"max distance={stats['max_dist']:.4f} Å"
        )

    if args.top_k > 0:
        ligand_heap = topk_heaps.get("ligand", [])
        if ligand_heap:
            worst = sorted((entry[1] for entry in ligand_heap), key=lambda rec: rec["rmsd"], reverse=True)
            print_log(f"Worst {len(worst)} ligand RMSDs:")
            for rank, rec in enumerate(worst, start=1):
                print_log(
                    f"  #{rank}: RMSD={rec['rmsd']:.4f} Å | atoms={rec['atoms']} | "
                    f"mean={rec['mean_distance']:.4f} Å | max={rec['max_distance']:.4f} Å | "
                    f"id={rec['sample_id']} (batch={rec['batch_index']})"
                )


if __name__ == "__main__":
    parsed_args, overrides = parse_args()
    evaluate(parsed_args, overrides)
