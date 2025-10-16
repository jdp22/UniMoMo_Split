#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from rdkit import Chem

from torch_scatter import scatter_mean, scatter_sum, scatter_min

from data.bioparse import VOCAB, const
from utils.nn_utils import SinusoidalPositionEmbedding, expand_like, SinusoidalTimeEmbeddings, graph_to_batch_nx
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean, scatter_sort
import utils.register as R
from utils.oom_decorator import oom_decorator

from .map import block_to_atom_map
from .tools import _avoid_clash

from ..modules.GET.tools import fully_connect_edges, knn_edges
from ..modules.nn import BlockEmbedding, MLP
from ..modules.create_net import create_net
from ..modules.metrics import batch_accu


def _random_mask(batch_ids, sigma=0.15):
    '''
        Get random mask position, with mask ratio 68% within 1 sigma, 95% within 2 sigma
    '''
    w = torch.empty(batch_ids.max() + 1, device=batch_ids.device) # [batch_size]
    # 68% within 1sigma (0.15), 95% within 2sigma (0.30)
    if sigma < 1e-5: # eps, near zero
        mask_ratio = w * 0.0
    else:
        mask_ratio = torch.abs(nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-1.0, b=1.0))
    mask = torch.rand_like(batch_ids, dtype=torch.float) < mask_ratio[batch_ids]
    return mask


def _contrastive_loss(
    x_repr,
    y_repr,
    reduction='none',
    temperature=1.0,
    negative_x=None,
    negative_y=None,
):
    '''
        Args:
            x_repr: [bs, hidden_size]
            y_repr: [bs, hidden_size]
            negative_x: [n_neg, hidden_size] negatives for the y->x direction
            negative_y: [n_neg, hidden_size] negatives for the x->y direction
        Returns:
            loss: [bs] if reduction == 'none' else scalar (sum, mean, ...)
    '''
    if x_repr.shape[0] == 0 or y_repr.shape[0] == 0:
        device = x_repr.device if x_repr.shape[0] else y_repr.device
        dtype = x_repr.dtype if x_repr.shape[0] else y_repr.dtype
        return torch.tensor(0.0, device=device, dtype=dtype)

    x_repr = F.normalize(x_repr, dim=-1)
    y_repr = F.normalize(y_repr, dim=-1)

    logits_xy = torch.matmul(x_repr, y_repr.T)
    if negative_y is not None and negative_y.numel():
        neg_y = F.normalize(negative_y.to(device=x_repr.device, dtype=x_repr.dtype), dim=-1)
        neg_logits = torch.matmul(x_repr, neg_y.T)
        logits_xy = torch.cat([logits_xy, neg_logits], dim=1)
    logits_xy = logits_xy / max(temperature, 1e-6)
    label_xy = torch.arange(x_repr.shape[0], device=x_repr.device)
    loss_xy = F.cross_entropy(input=logits_xy, target=label_xy, reduction=reduction)

    logits_yx = torch.matmul(y_repr, x_repr.T)
    if negative_x is not None and negative_x.numel():
        neg_x = F.normalize(negative_x.to(device=y_repr.device, dtype=y_repr.dtype), dim=-1)
        neg_logits = torch.matmul(y_repr, neg_x.T)
        logits_yx = torch.cat([logits_yx, neg_logits], dim=1)
    logits_yx = logits_yx / max(temperature, 1e-6)
    label_yx = torch.arange(y_repr.shape[0], device=y_repr.device)
    loss_yx = F.cross_entropy(input=logits_yx, target=label_yx, reduction=reduction)

    return 0.5 * (loss_xy + loss_yx)


def _contrastive_accu(x_repr, y_repr):
    if x_repr.shape[0] == 0: return 1.0, 1.0
    x2y = torch.matmul(x_repr, y_repr.T) # [bs, bs], from x to y
    label = torch.arange(x_repr.shape[0], device=x_repr.device)
    x2y_accu = (torch.argmax(x2y, dim=-1) == label).long().sum() / len(label)
    y2x_accu = (torch.argmax(x2y.T, dim=-1) == label).long().sum() / len(label)
    return x2y_accu, y2x_accu


def _local_distance_loss(X_gt, X_t, t, batch_ids, block_ids, generate_mask, dist_th=6.0, t_th=0.25):
    with torch.no_grad():
        row, col = fully_connect_edges(batch_ids[block_ids])

        # at least one end is in generation part, and don't repeat the same pair
        select_mask = (generate_mask[block_ids[row]] | generate_mask[block_ids[col]]) & (row < col)
        row, col = row[select_mask], col[select_mask]

        # get distance within 6.0 A
        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1) # [E]
        select_mask = (dist < dist_th)

        row, col, dist = row[select_mask], col[select_mask], dist[select_mask]

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    return loss.mean() if len(loss) else 0


def _bond_length_loss(X_gt, X_t, t, bonds, block_ids, generate_mask, t_th=0.25):
    with torch.no_grad():
        generate_mask = generate_mask[block_ids]
        bond_mask = generate_mask[bonds[:, 0]] & generate_mask[bonds[:, 1]]
        bonds = bonds[bond_mask]
        row, col = bonds[:, 0], bonds[:, 1]
        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1)

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    return loss.mean() if len(loss) else 0


# conditional iterative autoencoder
@R.register('CondIterAutoEncoder')
class CondIterAutoEncoder(nn.Module):
    """Conditional iterative auto-encoder used throughout UniMoMo.

    The module couples a graph encoder/decoder pair with several optional
    objectives (contrastive, retrieval-aware, 2D fragment RAG) used during
    training.  The constructor wires together all shared components
    (embeddings, latent heads, losses) while the forward pass coordinates
    dynamic ligand partitions and retrieval side information supplied via
    ``kwargs``.  Comments below highlight where each sub-system is initialised
    or consumed so the subsequent methods read more clearly."""
    def __init__(
            self,
            encoder_type: str,
            decoder_type: str,
            embed_size,
            hidden_size,
            latent_size,
            edge_size,
            num_block_type = VOCAB.get_num_block_type(),
            num_atom_type = VOCAB.get_num_atom_type(),
            num_bond_type = 5, # [None, single, double, triple, aromatic]
            max_num_atoms = const.aa_max_n_atoms,
            k_neighbors = 9,
            encoder_opt = {},
            decoder_opt = {},
            loss_weights = {
                'Zh_kl_loss': 0.3,
                'Zx_kl_loss': 0.5,
                'atom_coord_loss': 1.0,
                'block_type_loss': 1.0,
                'contrastive_loss': 0.3,
                'local_distance_loss': 0.5,
                'bond_loss': 0.5,
                'bond_length_loss': 0.0,
                # optional: RAG-style contrastive with 2D subgraphs
                'rag_contrastive_loss': 0.0,
                # optional: retrieval loss for dynamic fragment vocabulary
                'retrieval_loss': 0.0
            },
            # auxiliary parameters
            prior_coord_std=1.0,
            coord_noise_scale=0.0,
            pocket_mask_ratio=0.05,     # cannot be zero when kl_on_pocket=False, otherwise the latent state of the pocket will explode
            decode_mask_ratio=0.0,
            kl_on_pocket=False,         # whether to exert kl regularization also on pocket nodes
            discrete_timestep=False,
            default_num_steps=10,
            # RAG options
            rag_opt=None,
            # Retrieval options
            retrieval_opt=None,
            # Contrastive options
            contrastive_opt=None,
        ) -> None:
        super().__init__()
        self.encoder = create_net(encoder_type, hidden_size, edge_size, encoder_opt)
        self.decoder = create_net(decoder_type, hidden_size, edge_size, decoder_opt)

        # --- Core parameterisations shared by encoder/decoder ---
        self.embedding = BlockEmbedding(num_block_type, num_atom_type, embed_size)
        self.ctx_embedding = nn.Embedding(2, embed_size) # [context, generation]
        self.edge_embedding = nn.Embedding(3, edge_size) # [intra, inter, topo]
        self.atom_edge_embedding = nn.Embedding(5, edge_size) # [None, single, double, triple, aromatic]
        self.enc_embed2hidden = nn.Linear(embed_size, hidden_size)
        self.dec_atom_embedding = nn.Embedding(num_atom_type, hidden_size)
        self.dec_pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.dec_latent2hidden = nn.Linear(latent_size, hidden_size)
        self.dec_time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        self.dec_input_linear = nn.Linear(hidden_size * 3 + latent_size, hidden_size) # atom, time, position, latent

        self.mask_embedding = nn.Parameter(torch.randn(latent_size), requires_grad=True)

        # --- Latent heads & variance parametrisations ---
        self.Wh_mu = nn.Linear(hidden_size, latent_size)
        self.Wh_log_var = nn.Linear(hidden_size, latent_size)
        self.Wx_log_var = nn.Linear(hidden_size, 1) # has to be isotropic gaussian to maintain equivariance

        # --- Downstream prediction heads (block types / bond types) ---
        self.block_type_mlp = MLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=num_block_type,
            n_layers=3,
            dropout=0.1
        )
        self.bond_type_mlp = MLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=num_bond_type,
            n_layers=3,
            dropout=0.1
        )

        self.max_num_atoms = max_num_atoms
        self.k_neighbors = k_neighbors
        self.loss_weights = loss_weights
        self.prior_coord_std = prior_coord_std
        self.coord_noise_scale = coord_noise_scale
        self.pocket_mask_ratio = pocket_mask_ratio
        self.decode_mask_ratio = decode_mask_ratio
        self.kl_on_pocket = kl_on_pocket
        self.discrete_timestep = discrete_timestep
        self.default_num_steps = default_num_steps

        # --- Retrieval-aware fragment (RAG) configuration ---
        self.rag_enabled = False
        self.rag_temperature = 0.07
        self.rag_negatives_per_pos = 8
        self.rag_use_mean_pool = True
        self.rag_debug = False
        self._rag_error_logged = False
        self.rag_zero_log_path = None
        self.rag_zero_detail_log_path = None
        self.rag_success_detail_log_path = None
        self.rag_cross_index = None
        self.rag_cross_modal_neg_ratio = 0.0
        self.rag_strict_dynamic_pos = False

        if rag_opt is not None and isinstance(rag_opt, dict):
            # Configure the optional 2D fragment encoder and negative sampling indexes.
            # The try/except ensures training keeps running even if RDKit/subgraph
            # dependencies are missing on the host machine.
            try:
                from ..modules.subgraph_encoder import Subgraph2DGNNEncoder
                self.subgraph_encoder = Subgraph2DGNNEncoder(
                    hidden_size=rag_opt.get('hidden_size', hidden_size),
                    num_layers=rag_opt.get('num_layers', 3),
                    atom_vocab_size=rag_opt.get('atom_vocab_size', 120),
                    bond_vocab_size=rag_opt.get('bond_vocab_size', 5),
                    readout=rag_opt.get('readout', 'mean')
                )
                # optional negatives index
                self.rag_index = None
                index_path = rag_opt.get('subgraph_index', None)
                if index_path:
                    try:
                        from utils.subgraph_storage import SubgraphIndex
                        self.rag_index = SubgraphIndex(index_path)
                    except Exception:
                        self.rag_index = None
                self.rag_temperature = rag_opt.get('temperature', 0.07)
                self.rag_negatives_per_pos = rag_opt.get('negatives_per_pos', 8)
                self.rag_use_mean_pool = rag_opt.get('use_mean_pool', True)
                self.rag_debug = rag_opt.get('debug', False)
                self.rag_enabled = True
                self.rag_zero_log_path = rag_opt.get('zero_positive_log', None)
                self.rag_zero_detail_log_path = rag_opt.get('zero_detail_log', None)
                self.rag_success_detail_log_path = rag_opt.get('success_detail_log', None)
                self.rag_cross_modal_neg_ratio = float(rag_opt.get('cross_modal_neg_ratio', 0.0) or 0.0)
                self.rag_strict_dynamic_pos = bool(rag_opt.get('strict_dynamic_pos', False))
                cross_index_path = rag_opt.get('cross_modal_index')
                if cross_index_path:
                    try:
                        from utils.subgraph_storage import SubgraphIndex
                        self.rag_cross_index = SubgraphIndex(cross_index_path)
                    except Exception:
                        self.rag_cross_index = None
                # logging cadence for success detail
                try:
                    self.rag_success_log_every = int(rag_opt.get('success_log_every', 1) or 1)
                except Exception:
                    self.rag_success_log_every = 1
                self._rag_success_counter = 0
                # logging cadence for zero-positive detail
                try:
                    self.rag_zero_log_every = int(rag_opt.get('zero_log_every', self.rag_success_log_every) or self.rag_success_log_every)
                except Exception:
                    self.rag_zero_log_every = self.rag_success_log_every
                self._rag_zero_counter = 0
            except Exception as exc:
                # fail-safe: keep training without RAG if dependency fails
                from utils.logger import print_log
                print_log(f"[RAG] disabled due to exception: {exc}", level='WARN')
                self.rag_enabled = False

        # --- Dynamic fragment retrieval index (separate from RAG) ---
        self.retrieval_enabled = False
        self.retrieval_temperature = 0.07
        self.retrieval_negatives = 16
        self.retrieval_topk = 5
        self.retrieval_device = torch.device('cpu')
        self.retrieval_index = None
        self.retrieval_query_proj: Optional[nn.Module] = None

        contrastive_opt = contrastive_opt or {}
        self.contrastive_temperature = float(contrastive_opt.get('temperature', 0.07))
        self.contrastive_queue_size = int(contrastive_opt.get('queue_size', 256))
        self.use_contrastive_queue = self.contrastive_queue_size > 0
        self.register_buffer('contrastive_queue_x', torch.empty(0, hidden_size))
        self.register_buffer('contrastive_queue_y', torch.empty(0, hidden_size))

        aa_mask_tensor = torch.tensor(VOCAB.aa_mask, dtype=torch.bool) if hasattr(VOCAB, 'aa_mask') else None
        self.register_buffer('aa_mask_tensor', aa_mask_tensor)

        self.rag_zero_pos_molecule_ids: set = set()
        self.rag_zero_pos_peptide_ids: set = set()

    @property
    def latent_size(self):
        return self.mask_embedding.shape[0]

    def _update_contrastive_queue(self, pocket_repr, ligand_repr):
        if not self.use_contrastive_queue:
            return
        if pocket_repr.shape[0] == 0 or ligand_repr.shape[0] == 0:
            return
        with torch.no_grad():
            pocket_norm = F.normalize(pocket_repr.detach(), dim=-1)
            ligand_norm = F.normalize(ligand_repr.detach(), dim=-1)
            queue_x = self.contrastive_queue_x.to(device=pocket_norm.device, dtype=pocket_norm.dtype)
            queue_y = self.contrastive_queue_y.to(device=ligand_norm.device, dtype=ligand_norm.dtype)
            queue_x = torch.cat([queue_x, pocket_norm], dim=0)
            queue_y = torch.cat([queue_y, ligand_norm], dim=0)
            if queue_x.shape[0] > self.contrastive_queue_size:
                queue_x = queue_x[-self.contrastive_queue_size:]
                queue_y = queue_y[-self.contrastive_queue_size:]
            self._buffers['contrastive_queue_x'] = queue_x.detach()
            self._buffers['contrastive_queue_y'] = queue_y.detach()

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3) (single-directional)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            warmup_progress=1.0, # increasing KL from 0% to 100% for training stability
            **kwargs
        ):
        """Run one training/evaluation step.

        The method ingests both the static batch (pocket + ligand) and any
        dynamic ligand partitions produced by ``DynamicBlockWrapper`` via
        ``kwargs``.  Downstream it orchestrates three phases: (1) swap-in the
        dynamic metadata, (2) encode/decode with iterative denoising, and
        (3) assemble the different objectives (reconstruction, contrastive,
        retrieval).  Only the first part is shown here; the rest of the method
        follows the same structure with additional inline comments."""
        dyn_block_lengths = kwargs.pop('dyn_block_lengths', None)
        dyn_block_types = kwargs.pop('dyn_block_types', None)
        dyn_generate_mask = kwargs.pop('dyn_generate_mask', None)
        dyn_block_ids = kwargs.pop('dyn_block_ids', None)
        dyn_chain_ids = kwargs.pop('dyn_chain_ids', None)
        dyn_position_ids = kwargs.pop('dyn_position_ids', None)
        dyn_is_aa = kwargs.pop('dyn_is_aa', None)
        dyn_num_blocks = kwargs.pop('dyn_num_blocks', None)
        dyn_fragment_smiles = kwargs.pop('dyn_fragment_smiles', None)
        if dyn_fragment_smiles is not None:
            dyn_fragment_smiles = list(dyn_fragment_smiles)

        block_lengths_device, block_lengths_dtype = block_lengths.device, block_lengths.dtype
        chain_ids_device, chain_ids_dtype = chain_ids.device, chain_ids.dtype
        position_ids_device, position_ids_dtype = position_ids.device, position_ids.dtype
        generate_mask_device = generate_mask.device
        S_device, S_dtype = S.device, S.dtype
        is_aa_device, is_aa_dtype = is_aa.device, is_aa.dtype

        block_ids_override = None
        if dyn_block_ids is not None and dyn_block_ids.numel():
            block_ids_override = dyn_block_ids.to(device=X.device, dtype=torch.long)

        if dyn_block_lengths is not None and dyn_block_lengths.numel():
            block_lengths = dyn_block_lengths.to(device=block_lengths_device, dtype=block_lengths_dtype)
            if dyn_block_types is not None and dyn_block_types.numel():
                S = dyn_block_types.to(device=S_device, dtype=S_dtype)
            if dyn_generate_mask is not None and dyn_generate_mask.numel():
                generate_mask = dyn_generate_mask.to(device=generate_mask_device)
            if dyn_chain_ids is not None and dyn_chain_ids.numel():
                chain_ids = dyn_chain_ids.to(device=chain_ids_device, dtype=chain_ids_dtype)
            if dyn_position_ids is not None and dyn_position_ids.numel():
                position_ids = dyn_position_ids.to(device=position_ids_device, dtype=position_ids_dtype)
            if dyn_is_aa is not None and dyn_is_aa.numel():
                is_aa = dyn_is_aa.to(device=is_aa_device, dtype=is_aa_dtype)
            if dyn_num_blocks is not None and dyn_num_blocks.numel():
                lengths = dyn_num_blocks.to(device=lengths.device, dtype=lengths.dtype)

        # backup ground truth
        X_gt, S_gt, A_gt = X.clone(), S.clone(), A.clone()
        block_ids = block_ids_override if block_ids_override is not None else length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths)
        batch_size = lengths.shape[0]

        sample_ids_input = kwargs.get('sample_ids', None)
        if sample_ids_input is None:
            sample_ids = [None] * batch_size
        else:
            if isinstance(sample_ids_input, torch.Tensor):
                sample_ids = sample_ids_input.detach().cpu().tolist()
            else:
                sample_ids = list(sample_ids_input)
            if len(sample_ids) < batch_size:
                sample_ids.extend([None] * (batch_size - len(sample_ids)))
            elif len(sample_ids) > batch_size:
                sample_ids = sample_ids[:batch_size]

        positive_mask = torch.zeros(batch_size, dtype=torch.bool, device=X.device)
        zero_molecule_ids: List[Optional[str]] = []
        zero_peptide_ids: List[Optional[str]] = []
        rag_source_list: List[Optional[str]] = [None] * batch_size

        # sample binding site mask prediction (0% to 30%)
        binding_site_gen_mask = _random_mask(batch_ids, sigma=self.pocket_mask_ratio) & (~generate_mask)

        # encoding
        Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, bind_site_repr, ligand_repr = self.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, binding_site_gen_mask=binding_site_gen_mask, block_ids_override=block_ids_override
        ) # [Nblock, d_latent], [Nblock, 3], [1], [1]

        if self.training: # add noise on Zx to improve robustness
            noise = torch.randn_like(Zx) * self.coord_noise_scale
            noise = torch.where(expand_like(generate_mask, Zx), noise, torch.zeros_like(noise))
            Zx = Zx + noise

        # random mask some Zh for contextual prediction
        Zh = self._random_mask(Zh, generate_mask, batch_ids)

        # decode block types from latent graph
        pred_block_logits, block_hidden = self.decode_block_type(Zh, Zx, chain_ids, lengths, return_hidden=True)

        # sample a timestep and decode structure
        if self.discrete_timestep:
            t = torch.randint(1, self.default_num_steps + 1, size=(batch_size,), device=X.device)[batch_ids][block_ids]
            t = t.float() / self.default_num_steps
        else:
            t = torch.rand(batch_size, device=X.device)[batch_ids][block_ids]
        # sample an initial state
        X_t, vector = self._sample_from_prior(X, Zx, block_ids, generate_mask | binding_site_gen_mask, t)
        # get topo edges
        topo_edges, topo_edge_type = self._get_topo_edges(bonds, block_ids, generate_mask)
        # sample some ground-truth inter-block topo edges
        inter_topo_edges, inter_topo_edge_type, inter_topo_edge_select_mask = self._unmask_inter_topo_edges(bonds, batch_ids, block_ids, generate_mask)
        topo_edges = torch.cat([topo_edges, inter_topo_edges.T[inter_topo_edge_select_mask].T], dim=1)
        topo_edge_type = torch.cat([topo_edge_type, inter_topo_edge_type[inter_topo_edge_select_mask]], dim=0)

        topo_edge_attr = self.atom_edge_embedding(topo_edge_type)
        # decode structure
        H_t, X_next = self.decode_structure(Zh, X_t, A, position_ids, topo_edges, topo_edge_attr, chain_ids, batch_ids, block_ids, t)
        pred_vector = X_next - X_t
        X_final = X_t + pred_vector * t[:, None] # for local neighborhood
        # decode inter-block bonds
        bond_loss_weight = float(self.loss_weights.get('bond_loss', 0.0))
        if bond_loss_weight > 0.0:
            given_bond_num = (~inter_topo_edge_select_mask).sum() // 2 # bidirectional to one-directional
            bond_to_pred, bond_label = self._get_bond_to_pred(
                X_t, bonds, batch_ids, block_ids, generate_mask,
                given_gt=(
                    inter_topo_edges.T[~inter_topo_edge_select_mask][:given_bond_num].T,
                    inter_topo_edge_type[~inter_topo_edge_select_mask][:given_bond_num]
                ))
            pred_bond_logits = self.bond_type_mlp(H_t[bond_to_pred[0]] + H_t[bond_to_pred[1]]) # commutative property
            bond_loss = F.cross_entropy(input=pred_bond_logits, target=bond_label)
            if torch.isnan(bond_loss):
                bond_loss = torch.tensor(0.0, device=X.device, dtype=torch.float)
        else:
            bond_loss = torch.tensor(0.0, device=X.device, dtype=torch.float)

        # loss
        loss_mask = generate_mask[block_ids]
        binding_site_loss_mask = binding_site_gen_mask[block_ids]
        contrastive_context_mask = scatter_sum((~generate_mask).long(), batch_ids, dim=0) > 0
        contrastive_ligand_mask = scatter_sum(generate_mask.long(), batch_ids, dim=0) > 0
        contrastive_loss_mask = contrastive_context_mask & contrastive_ligand_mask
        pocket_repr = bind_site_repr[contrastive_loss_mask]
        ligand_repr_sel = ligand_repr[contrastive_loss_mask]
        contrastive_weight = float(self.loss_weights.get('contrastive_loss', 0.0))
        contrastive_enabled = contrastive_weight > 0.0
        if contrastive_enabled and self.training:
            self._update_contrastive_queue(pocket_repr, ligand_repr_sel)

        if contrastive_enabled:
            neg_x = None
            neg_y = None
            if self.use_contrastive_queue and self.contrastive_queue_x.numel():
                neg_x = self.contrastive_queue_x.to(device=Zh.device, dtype=Zh.dtype)
            if self.use_contrastive_queue and self.contrastive_queue_y.numel():
                neg_y = self.contrastive_queue_y.to(device=Zh.device, dtype=Zh.dtype)
            contrastive_loss = _contrastive_loss(
                pocket_repr,
                ligand_repr_sel,
                reduction='mean',
                temperature=self.contrastive_temperature,
                negative_x=neg_x,
                negative_y=neg_y,
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=Zh.device, dtype=Zh.dtype)
        target_mask = generate_mask | binding_site_gen_mask
        block_type_weight = float(self.loss_weights.get('block_type_loss', 0.0))
        if block_type_weight <= 0.0:
            block_type_loss = torch.tensor(0.0, device=X.device, dtype=torch.float)
        elif self.retrieval_enabled:
            block_type_loss = torch.tensor(0.0, device=X.device, dtype=torch.float)
        elif target_mask.any():
            block_type_loss = F.cross_entropy(
                input=pred_block_logits[target_mask],
                target=S_gt[target_mask]
            )
        else:
            block_type_loss = torch.tensor(0.0, device=X.device, dtype=torch.float)

        loss_dict = {
            'Zh_kl_loss': Zh_kl_loss,
            'Zx_kl_loss': Zx_kl_loss,
            'block_type_loss': block_type_loss,
            'atom_coord_loss': F.mse_loss(
                pred_vector[loss_mask | binding_site_loss_mask],
                vector[loss_mask | binding_site_loss_mask], reduction='none'
            ).sum(-1).mean(), # sum the xyz dimension and average between atoms
            'contrastive_loss': contrastive_loss,
            'local_distance_loss': _local_distance_loss(X_gt, X_final, t, batch_ids, block_ids, generate_mask | binding_site_gen_mask),
            'bond_loss': bond_loss,
            'bond_length_loss': _bond_length_loss(X_gt, X_final, t, bonds, block_ids, generate_mask),
            # default zero; may be overwritten below if RAG inputs available
            'rag_contrastive_loss': torch.tensor(0.0, device=X.device, dtype=torch.float),
            'retrieval_loss': torch.tensor(0.0, device=X.device, dtype=torch.float),
            'contrastive_pairs': torch.tensor(
                float(pocket_repr.shape[0] if contrastive_enabled else 0.0),
                device=X.device,
                dtype=torch.float
            )
        }
        # TODO: Do we really need normalization? Yes
        block_hidden_norm = F.normalize(block_hidden, dim=-1)
        residue_mask = (~generate_mask) & (is_aa if 'is_aa' in locals() else torch.zeros_like(generate_mask))
        if residue_mask.device != block_hidden_norm.device:
            residue_mask = residue_mask.to(block_hidden_norm.device)
        residue_neg_emb = block_hidden_norm[residue_mask]
        fragment_encoder = getattr(self, 'subgraph_encoder', None)

        def _sample_negative_embeddings(exclude_smiles, device, num_pos, extra_neg_emb=None, use_index=True):
            """Collect negatives from retrieval/index and optionally extra embeddings."""
            neg_tensors: list[torch.Tensor] = []
            neg_smiles: list[str] = []
            neg_per_pos = int(max(self.rag_negatives_per_pos, 0))
            target = neg_per_pos * max(1, num_pos) if neg_per_pos and use_index else 0

            if target and getattr(self, 'rag_index', None) is not None and fragment_encoder is not None:
                candidates: list[str] = []
                for src in exclude_smiles or []:
                    try:
                        candidates.extend(self.rag_index.sample_negatives(src, k=neg_per_pos))
                    except Exception:
                        continue
                seen: set[str] = set()
                unique: list[str] = []
                for smi in candidates:
                    if smi and smi not in seen:
                        seen.add(smi)
                        unique.append(smi)
                if not unique:
                    try:
                        global_candidates = self.rag_index.sample_negatives(
                            None, k=neg_per_pos * max(1, num_pos)
                        )
                    except Exception:
                        global_candidates = []
                    unique = [s for s in global_candidates if s]
                if unique:
                    try:
                        fragment_encoder = fragment_encoder.to(device)
                        encoded = fragment_encoder.encode_smiles_list(unique, device=device)
                        if encoded is not None and encoded.numel():
                            neg_tensors.append(F.normalize(encoded, dim=-1))
                            neg_smiles = unique
                    except Exception:
                        pass

            if extra_neg_emb is not None and extra_neg_emb.numel():
                extra = extra_neg_emb.to(device, non_blocking=True)
                max_extra = target if target else extra.shape[0]
                if 0 < max_extra < extra.shape[0]:
                    idx = torch.randperm(extra.shape[0], device=extra.device)[:max_extra]
                    extra = extra[idx]
                neg_tensors.append(extra)

            if neg_tensors:
                neg_emb = torch.cat(neg_tensors, dim=0).to(device, non_blocking=True)
            else:
                neg_emb = torch.empty(0, block_hidden_norm.shape[-1], device=device)
            return neg_emb, neg_smiles

        def _rag_infonce(anchors, positives, temperature, exclude_smiles=None, source_types=None):
            """Compute InfoNCE loss for dynamic fragments.

            In-batch anchors are always treated as negatives first; if the
            batch only provides a single fragment we fall back to samples from
            the configured fragment indexes.  The returned ``infonote`` keeps
            provenance of negatives so that the caller can surface precise
            failure reasons in the training logs."""
            if anchors is None or positives is None:
                return None, 0, 0, {'reason': 'empty_inputs'}
            if anchors.numel() == 0 or positives.numel() == 0:
                return None, 0, 0, {'reason': 'empty_inputs'}
            if anchors.shape[0] != positives.shape[0]:
                mismatch = {
                    'reason': 'shape_mismatch',
                    'anchor_len': int(anchors.shape[0]),
                    'positive_len': int(positives.shape[0]),
                }
                if getattr(self, 'rag_debug', False):
                    from utils.logger import print_log
                    print_log(f"[RAG] anchor/positive count mismatch: {mismatch}", level='ERROR')
                raise ValueError(f"RAG InfoNCE expects equal counts, got {mismatch}")
            m = anchors.shape[0]
            if m == 0:
                return None, 0, 0, {'reason': 'empty_inputs'}
            positives = positives.to(anchors.device, non_blocking=True)
            pos_logits = torch.sum(anchors * positives, dim=-1, keepdim=True)
            neg_chunks = []
            # collect external negative smiles by modality for logging
            neg_ext_molecule: list = []
            neg_ext_peptide: list = []
            # In-batch negatives first (other positives in the same batch)
            if m > 1:
                cross = torch.matmul(anchors, positives.t())
                mask = torch.eye(m, device=anchors.device, dtype=torch.bool)
                cross = cross.masked_fill(mask, float('-inf'))
                neg_chunks.append(cross)

            try:
                ratio = float(self.rag_cross_modal_neg_ratio or 0.0)
            except Exception:
                ratio = 0.0
            ratio = max(0.0, min(1.0, ratio))

            def _index_has_entries(index_obj):
                if index_obj is None:
                    return False
                if hasattr(index_obj, 'all_frags'):
                    return bool(index_obj.all_frags)
                if hasattr(index_obj, 'smiles_list'):
                    return bool(index_obj.smiles_list)
                return True

            cross_modal_enabled = (
                ratio > 0.0
                and source_types is not None
                and (getattr(self, 'rag_index', None) is not None or getattr(self, 'rag_cross_index', None) is not None)
                and self.rag_negatives_per_pos > 0
            )

            source_list = None
            if cross_modal_enabled:
                if isinstance(source_types, torch.Tensor):
                    source_list = source_types.detach().cpu().tolist()
                else:
                    try:
                        source_list = list(source_types)
                    except Exception:
                        source_list = None
                if source_list:
                    converted = []
                    for item in source_list:
                        if isinstance(item, bool):
                            converted.append(item)
                        elif item is None:
                            converted.append(False)
                        else:
                            converted.append(bool(item))
                    source_list = converted[:m]
                    if len(source_list) < m and source_list:
                        source_list.extend([source_list[-1]] * (m - len(source_list)))
                if not source_list:
                    cross_modal_enabled = False
                else:
                    need_molecule_cross = any(flag is True for flag in source_list)
                    need_peptide_cross = any(flag is False for flag in source_list)
                    has_cross_candidates = False
                    if need_molecule_cross:
                        idx_obj = getattr(self, 'rag_cross_index', None)
                        if _index_has_entries(idx_obj):
                            has_cross_candidates = True
                    if need_peptide_cross:
                        idx_obj = getattr(self, 'rag_index', None)
                        if _index_has_entries(idx_obj):
                            has_cross_candidates = True
                    if not has_cross_candidates:
                        cross_modal_enabled = False

            def _encode_smiles_for_neg(smiles_list):
                if not smiles_list:
                    return None
                try:
                    if fragment_encoder is not None:
                        fragment_encoder.to(anchors.device)
                        emb = fragment_encoder.encode_smiles_list(smiles_list, device=anchors.device)
                        return F.normalize(emb, dim=-1)
                except Exception:
                    return None
                return None

            def _draw_from_index(index_obj, k, exclude):
                if index_obj is None or k <= 0:
                    return []
                try:
                    smiles = index_obj.sample_negatives(source_smiles=None, k=k)
                except Exception:
                    return []
                if exclude:
                    ex = set(exclude)
                    smiles = [s for s in smiles if s not in ex]
                seen = set()
                result = []
                for smi in smiles:
                    if not smi or smi in seen:
                        continue
                    seen.add(smi)
                    result.append(smi)
                    if len(result) >= k:
                        break
                return result

            if cross_modal_enabled:
                neg_tensors = []
                total_capacity = int(max(self.rag_negatives_per_pos, 0) * max(1, m))
                total_added = 0
                exclude_set = set(exclude_smiles or [])
                for is_molecule in (True, False):
                    group_indices = [i for i, flag in enumerate(source_list) if flag is is_molecule]
                    if not group_indices:
                        continue
                    group_target = int(max(self.rag_negatives_per_pos, 0) * len(group_indices))
                    if group_target <= 0:
                        continue
                    cross_index_obj = getattr(self, 'rag_cross_index', None) if is_molecule else getattr(self, 'rag_index', None)
                    same_index_obj = getattr(self, 'rag_index', None) if is_molecule else getattr(self, 'rag_cross_index', None)
                    cross_target = int(round(group_target * ratio))
                    if cross_target > 0 and not _index_has_entries(cross_index_obj):
                        cross_target = 0
                    inmodal_target = max(group_target - cross_target, 0)

                    if inmodal_target > 0:
                        if inmodal_target > 0 and _index_has_entries(same_index_obj):
                            same_smiles = _draw_from_index(same_index_obj, inmodal_target, exclude_set)
                            cand = _encode_smiles_for_neg(same_smiles)
                            if cand is not None and cand.numel():
                                neg_tensors.append(cand)
                                total_added += cand.shape[0]
                                exclude_set.update(same_smiles)
                                # same_index_obj corresponds to in‑modal JSONL: rag_index=molecule, rag_cross_index=peptide
                                if same_index_obj is getattr(self, 'rag_index', None):
                                    neg_ext_molecule.extend([s for s in (same_smiles or []) if s])
                                elif same_index_obj is getattr(self, 'rag_cross_index', None):
                                    neg_ext_peptide.extend([s for s in (same_smiles or []) if s])

                    if cross_target > 0:
                        cross_smiles = _draw_from_index(cross_index_obj, cross_target, exclude_set)
                        cand = _encode_smiles_for_neg(cross_smiles)
                        if cand is not None and cand.numel():
                            neg_tensors.append(cand)
                            total_added += cand.shape[0]
                            exclude_set.update(cross_smiles)
                            # cross_index_obj corresponds to cross‑modal JSONL: rag_index=molecule, rag_cross_index=peptide
                            if cross_index_obj is getattr(self, 'rag_index', None):
                                neg_ext_molecule.extend([s for s in (cross_smiles or []) if s])
                            elif cross_index_obj is getattr(self, 'rag_cross_index', None):
                                neg_ext_peptide.extend([s for s in (cross_smiles or []) if s])

                if residue_neg_emb is not None and residue_neg_emb.numel():
                    extra = residue_neg_emb.to(anchors.device, non_blocking=True)
                    remaining = max(total_capacity - total_added, 0)
                    if 0 < remaining < extra.shape[0]:
                        idx = torch.randperm(extra.shape[0], device=extra.device)[:remaining]
                        extra = extra[idx]
                    if extra.numel():
                        neg_tensors.append(extra)

                if neg_tensors:
                    neg_emb = torch.cat(neg_tensors, dim=0)
                    neg_chunks.append(torch.matmul(anchors, neg_emb.t()))
            else:
                need_external = (m <= 1)
                neg_emb, neg_smiles = _sample_negative_embeddings(
                    exclude_smiles or [],
                    anchors.device,
                    m,
                    extra_neg_emb=residue_neg_emb,
                    use_index=need_external
                )
                if neg_emb.numel():
                    neg_chunks.append(torch.matmul(anchors, neg_emb.t()))
                    # best effort: sampled negatives here come from retrieval/index; treat as molecule if in retrieval_index, otherwise JSONL may be molecule index
                    if neg_smiles:
                        neg_ext_molecule.extend([s for s in neg_smiles if s])

            if not neg_chunks:
                info = {
                    'reason': 'no_negatives',
                    'count': m,
                    'exclude_smiles': list(exclude_smiles) if exclude_smiles else [],
                }
                if self.rag_debug:
                    from utils.logger import print_log
                    print_log(
                        f"[RAG] skipped positives due to missing negatives; anchors={m}; exclude={info['exclude_smiles']}",
                        level='WARN'
                    )
                return None, m, 0, info
            neg_logits = torch.cat(neg_chunks, dim=1)
            logits = torch.cat([pos_logits, neg_logits], dim=1) / max(temperature, 1e-6)
            labels = torch.zeros(m, dtype=torch.long, device=anchors.device)
            # per‑row losses for later per‑sample logging
            per_row_loss = F.cross_entropy(logits, labels, reduction='none')
            loss = per_row_loss.mean()
            info = {
                'neg_ext_molecule': list({s for s in neg_ext_molecule})[:20],
                'neg_ext_peptide': list({s for s in neg_ext_peptide})[:20],
                'per_row_loss': per_row_loss.detach().cpu().tolist(),
            }
            return loss, m, neg_logits.shape[1], info


        # Optional: RAG contrastive between decoder block embeddings and 2D fragment encodings
        rag_pos_count = 0.0
        rag_pos_count_molecule = 0.0
        rag_pos_count_peptide = 0.0
        rag_source_count = 0
        if self.rag_enabled:
            rag_source_smiles = kwargs.get('rag_source_smiles', None)
            if isinstance(rag_source_smiles, (list, tuple)):
                rag_source_count = sum(1 for item in rag_source_smiles if item)
                rag_source_list = list(rag_source_smiles)
            elif rag_source_smiles:
                rag_source_count = 1
                rag_source_list = [rag_source_smiles]
            if len(rag_source_list) < batch_size:
                rag_source_list.extend([None] * (batch_size - len(rag_source_list)))
            elif len(rag_source_list) > batch_size:
                rag_source_list = rag_source_list[:batch_size]
            # Only treat dyn_fragment_smiles as valid when dynamic blocks are active
            dyn_mode_active = (dyn_block_lengths is not None and isinstance(dyn_block_lengths, torch.Tensor) and dyn_block_lengths.numel() > 0)
            dyn_frags_valid = bool(dyn_mode_active and dyn_fragment_smiles is not None and any(bool(s) for s in dyn_fragment_smiles))
            rag_loss_applied = False
            rag_anchor_count = 0
            rag_negative_count = 0
            rag_dyn_pos_candidates = 0
            rag_pool_pos_candidates = 0
            rag_debug_error = 0
            dyn_fragments_per_sample = {}
            skip_reason_map = {}
            # new: per-sample negative record and per-sample contrastive loss
            rag_per_sample_loss = {}
            rag_negatives_map = {}
            if isinstance(dyn_fragment_smiles, list):
                gen_mask_list = generate_mask.detach().cpu().tolist()
                batch_id_list = batch_ids.detach().cpu().tolist()
                for block_idx, smi in enumerate(dyn_fragment_smiles):
                    if block_idx >= len(gen_mask_list) or not gen_mask_list[block_idx]:
                        continue
                    sample_idx = batch_id_list[block_idx]
                    if smi:
                        dyn_fragments_per_sample.setdefault(sample_idx, []).append(smi)
            try:
                # per-block anchors with dynamically generated fragments
                if dyn_frags_valid:
                    rag_indices = [i for i, smi in enumerate(dyn_fragment_smiles) if smi]
                    rag_dyn_pos_candidates = len(rag_indices)
                    if rag_indices:
                        anchor_idx = torch.tensor(rag_indices, dtype=torch.long, device=block_hidden_norm.device)
                        if anchor_idx.device != block_hidden_norm.device:
                            anchor_idx = anchor_idx.to(block_hidden_norm.device, dtype=torch.long, non_blocking=True)
                        if self.rag_debug and anchor_idx.device != block_hidden_norm.device:
                            from utils.logger import print_log
                            print_log(f"[RAG] anchor_idx device mismatch: idx={anchor_idx.device}, hidden={block_hidden_norm.device}", level='WARN')
                        # Guard against rare misalignment
                        if anchor_idx.numel() and anchor_idx.max().item() < block_hidden_norm.shape[0]:
                            anchors = block_hidden_norm[anchor_idx]
                            if self.rag_debug and anchors.device != block_hidden_norm.device:
                                from utils.logger import print_log
                                print_log(f"[RAG] anchors device mismatch: anchors={anchors.device}, hidden={block_hidden_norm.device}", level='WARN')
                            pos_smiles = [dyn_fragment_smiles[i] for i in rag_indices]
                            pos_emb = None
                            if fragment_encoder is not None:
                                frag_device = anchors.device
                                try:
                                    fragment_encoder.to(frag_device)
                                    pos_emb = fragment_encoder.encode_smiles_list(pos_smiles, device=frag_device)
                                except Exception:
                                    pos_emb = None
                        loss_val = None
                        used = 0
                        negs = 0
                        infonote = None
                        if pos_emb is not None:
                            pos_emb = pos_emb.to(anchors.device, non_blocking=True)
                        if pos_emb is None or pos_emb.numel() == 0:
                            if self.rag_debug:
                                from utils.logger import print_log
                                smiles_str = ','.join(pos_smiles) if pos_smiles else '<empty>'
                                print_log(f"[RAG] skipped positives because embeddings empty; smiles={smiles_str}", level='WARN')
                            if anchor_idx.numel():
                                try:
                                    sample_indices = batch_ids[anchor_idx.detach().cpu()].tolist()
                                except Exception:
                                    sample_indices = batch_ids[anchor_idx.long()].detach().cpu().tolist()
                                for si in sample_indices:
                                    if 0 <= si < batch_size:
                                        entry = skip_reason_map.setdefault(si, {'reasons': set(), 'exclude_smiles': []})
                                        entry['reasons'].add('positive_embedding_empty')
                        elif anchors.numel() and pos_emb.shape[0] and pos_emb.shape[-1] == anchors.shape[-1]:
                            # ensure SMILES 列表与 anchor 数一致，便于定位不一致来源（数据准备 vs encoder）
                            assert len(pos_smiles) == anchors.shape[0], (
                                f"Mismatch between dynamic fragment smiles ({len(pos_smiles)}) "
                                f"and anchors ({anchors.shape[0]})."
                            )
                            pos_emb = F.normalize(pos_emb, dim=-1)
                            source_flags = None
                            if anchor_idx.numel():
                                try:
                                    sample_indices = batch_ids[anchor_idx.detach().cpu()].tolist()
                                except Exception:
                                    sample_indices = batch_ids[anchor_idx.long()].detach().cpu().tolist()
                                source_flags = []
                                for si in sample_indices:
                                    if 0 <= si < len(rag_source_list) and rag_source_list[si]:
                                        source_flags.append(True)
                                    else:
                                        source_flags.append(False)
                            loss_val, used, negs, infonote = _rag_infonce(
                                anchors,
                                pos_emb,
                                self.rag_temperature,
                                exclude_smiles=pos_smiles,
                                source_types=source_flags
                            )
                            if loss_val<1e-4:
                                breakpoint()
                        else:
                            if self.rag_debug:
                                from utils.logger import print_log
                                info = {
                                    'anchors_shape': tuple(anchors.shape) if isinstance(anchors, torch.Tensor) else 'NA',
                                    'pos_emb_shape': tuple(pos_emb.shape) if isinstance(pos_emb, torch.Tensor) else 'NA',
                                    'smiles': pos_smiles,
                                }
                                print_log(f"[RAG] skipped positives due to shape mismatch: {info}", level='WARN')
                            if anchor_idx.numel():
                                try:
                                    sample_indices = batch_ids[anchor_idx.detach().cpu()].tolist()
                                except Exception:
                                    sample_indices = batch_ids[anchor_idx.long()].detach().cpu().tolist()
                                for si in sample_indices:
                                    if 0 <= si < batch_size:
                                        entry = skip_reason_map.setdefault(si, {'reasons': set(), 'exclude_smiles': []})
                                        entry['reasons'].add('positive_embedding_shape_mismatch')
                        if infonote and infonote.get('reason') == 'no_negatives' and anchor_idx.numel():
                            count = infonote.get('count', anchor_idx.shape[0])
                            sel = anchor_idx[:min(count, anchor_idx.shape[0])]
                            sample_indices = batch_ids[sel.detach().cpu()].tolist()
                            for si in sample_indices:
                                if 0 <= si < batch_size:
                                    entry = skip_reason_map.setdefault(si, {'reasons': set(), 'exclude_smiles': []})
                                    entry['reasons'].add('no_negatives')
                                    if infonote.get('exclude_smiles'):
                                        entry['exclude_smiles'] = infonote['exclude_smiles'][:10]
                        if loss_val is not None:
                            loss_dict['rag_contrastive_loss'] = loss_val
                            rag_loss_applied = True
                            rag_anchor_count += used
                            rag_negative_count = max(rag_negative_count, negs)
                            rag_pos_count += float(used)
                            if used > 0 and anchor_idx.numel():
                                # Record which batch samples actually contributed positive anchors
                                # so we can later log molecules/peptides that still lack positives.
                                selected_idx = anchor_idx[:used]
                                sample_indices = batch_ids[selected_idx.detach().cpu()].tolist()
                                anchor_flags = is_aa[selected_idx][:used].detach().cpu().tolist()
                                for si, flag in zip(sample_indices, anchor_flags):
                                    if 0 <= si < batch_size:
                                        positive_mask[si] = True
                                        if flag:
                                            rag_pos_count_peptide += 1.0
                                        else:
                                            rag_pos_count_molecule += 1.0
                                # per-sample loss from InfoNCE (if provided)
                                per_row = None
                                try:
                                    per_row = infonote.get('per_row_loss') if isinstance(infonote, dict) else None
                                except Exception:
                                    per_row = None
                                if isinstance(per_row, list) and len(per_row) >= len(sample_indices):
                                    for si, lval in zip(sample_indices, per_row[:len(sample_indices)]):
                                        try:
                                            rag_per_sample_loss[si] = float(lval)
                                        except Exception:
                                            pass
                                # negatives per sample: external and in-batch
                                ext_mol = []
                                ext_pep = []
                                try:
                                    if isinstance(infonote, dict):
                                        ext_mol = list(infonote.get('neg_ext_molecule', []) or [])
                                        ext_pep = list(infonote.get('neg_ext_peptide', []) or [])
                                except Exception:
                                    ext_mol, ext_pep = [], []
                                # in-batch negatives: other positives in this group
                                # derive modalities for in-batch via source_flags
                                in_batch_lists = []
                                try:
                                    # recompute source_flags for safety
                                    recompute_flags = []
                                    if anchor_idx.numel():
                                        try:
                                            sample_indices_all = batch_ids[anchor_idx.detach().cpu()].tolist()
                                        except Exception:
                                            sample_indices_all = batch_ids[anchor_idx.long()].detach().cpu().tolist()
                                        for si2 in sample_indices_all:
                                            if 0 <= si2 < len(rag_source_list) and rag_source_list[si2]:
                                                recompute_flags.append(True)
                                            else:
                                                recompute_flags.append(False)
                                    for i_local, si in enumerate(sample_indices):
                                        inb_mol = []
                                        inb_pep = []
                                        for j_local, smi in enumerate(pos_smiles[:min(len(pos_smiles), used)]):
                                            if j_local == i_local or not smi:
                                                continue
                                            if j_local < len(recompute_flags) and recompute_flags[j_local]:
                                                inb_mol.append(smi)
                                            else:
                                                inb_pep.append(smi)
                                        in_batch_lists.append((si, inb_mol[:10], inb_pep[:10]))
                                except Exception:
                                    in_batch_lists = []
                                # merge into rag_negatives_map per sample
                                for si in sample_indices:
                                    rec = rag_negatives_map.get(si, {})
                                    rec.setdefault('external', {})
                                    rec.setdefault('in_batch', {})
                                    if ext_mol:
                                        prev = rec['external'].get('molecule', [])
                                        rec['external']['molecule'] = (prev + ext_mol)[:20]
                                    if ext_pep:
                                        prev = rec['external'].get('peptide', [])
                                        rec['external']['peptide'] = (prev + ext_pep)[:20]
                                    rag_negatives_map[si] = rec
                                for si, inb_mol, inb_pep in in_batch_lists:
                                    rec = rag_negatives_map.get(si, {})
                                    rec.setdefault('external', {})
                                    rec.setdefault('in_batch', {})
                                    if inb_mol:
                                        prev = rec['in_batch'].get('molecule', [])
                                        rec['in_batch']['molecule'] = (prev + inb_mol)[:20]
                                    if inb_pep:
                                        prev = rec['in_batch'].get('peptide', [])
                                        rec['in_batch']['peptide'] = (prev + inb_pep)[:20]
                                    rag_negatives_map[si] = rec
            except Exception as exc:
                # Keep training even if RAG computation fails; leave rag loss as 0
                rag_debug_error = 1
                if self.rag_debug:
                    from utils.logger import print_log
                    debug_state = {
                        'anchors': str(anchors.device) if 'anchors' in locals() and isinstance(anchors, torch.Tensor) else 'NA',
                        'anchor_idx': str(anchor_idx.device) if 'anchor_idx' in locals() and isinstance(anchor_idx, torch.Tensor) else 'NA',
                        'ligand_repr': str(ligand_repr.device),
                        'block_hidden_norm': str(block_hidden_norm.device),
                        'rag_indices_len': len(rag_indices) if 'rag_indices' in locals() else -1,
                        'valid_rows_len': -1,
                        'pos_emb': str(pos_emb.device) if 'pos_emb' in locals() and isinstance(pos_emb, torch.Tensor) else 'NA',
                    }
                    print_log(f"[RAG] loss computation failed: {exc}; states={debug_state}", level='WARN')
                    self._rag_error_logged = True
        # positives必须来自当前动态片段。
            if self.rag_debug:
                loss_dict['rag_anchor_count'] = torch.tensor(rag_anchor_count, dtype=torch.float, device=X.device)
                loss_dict['rag_negative_count'] = torch.tensor(rag_negative_count, dtype=torch.float, device=X.device)
                loss_dict['rag_dyn_pos_candidates'] = torch.tensor(rag_dyn_pos_candidates, dtype=torch.float, device=X.device)
                loss_dict['rag_pool_pos_candidates'] = torch.tensor(rag_pool_pos_candidates, dtype=torch.float, device=X.device)
                loss_dict['rag_has_encoder'] = torch.tensor(1.0 if fragment_encoder is not None else 0.0, dtype=torch.float, device=X.device)
                loss_dict['rag_debug_error'] = torch.tensor(float(rag_debug_error), dtype=torch.float, device=X.device)

        rag_zero_mol_count = 0
        rag_zero_pep_count = 0
        new_zero_molecule_ids = []
        new_zero_peptide_ids = []
        if self.rag_enabled:
            positive_mask_cpu = positive_mask.detach().cpu()
            zero_indices = (positive_mask_cpu == 0).nonzero(as_tuple=False).view(-1).tolist()
            for idx in zero_indices:
                if not (0 <= idx < batch_size):
                    continue
                sample_id = sample_ids[idx]
                is_molecule = False
                if idx < len(rag_source_list) and rag_source_list[idx]:
                    is_molecule = True
                if is_molecule:
                    rag_zero_mol_count += 1
                    if sample_id:
                        if sample_id not in self.rag_zero_pos_molecule_ids:
                            self.rag_zero_pos_molecule_ids.add(sample_id)
                            new_zero_molecule_ids.append(sample_id)
                else:
                    rag_zero_pep_count += 1
                    if sample_id:
                        if sample_id not in self.rag_zero_pos_peptide_ids:
                            self.rag_zero_pos_peptide_ids.add(sample_id)
                            new_zero_peptide_ids.append(sample_id)

        if self.rag_zero_log_path and (new_zero_molecule_ids or new_zero_peptide_ids):
            try:
                with open(self.rag_zero_log_path, 'a', encoding='utf-8') as f:
                    for sid in new_zero_molecule_ids:
                        f.write(json.dumps({'type': 'molecule', 'id': sid}) + '\n')
                    for sid in new_zero_peptide_ids:
                        f.write(json.dumps({'type': 'peptide', 'id': sid}) + '\n')
            except Exception:
                pass

        # Collect structured diagnostics for samples that failed to produce
        # positives so we can inspect ``failure_reasons`` downstream.
        zero_records = []
        if self.rag_enabled:
            current_epoch = kwargs.get('current_epoch', None)
            zero_every = getattr(self, 'rag_zero_log_every', 1)
            for idx in zero_indices:
                if not (0 <= idx < batch_size):
                    continue
                sample_id = sample_ids[idx]
                dyn_frags = dyn_fragments_per_sample.get(idx, [])
                reason_components = []
                if dyn_frags:
                    reason_components.append('dyn_fragments_available')
                if idx < len(rag_source_list) and rag_source_list[idx]:
                    reason_components.append('rag_source_smiles_present')
                if not reason_components:
                    reason_components.append('no_fragments_available')
                if rag_debug_error:
                    reason_components.append('rag_debug_error')
                skip_entry = skip_reason_map.get(idx)
                if skip_entry:
                    reason_components.extend(sorted(skip_entry.get('reasons', [])))
                record = {
                    'type': 'molecule' if (idx < len(rag_source_list) and rag_source_list[idx]) else 'peptide',
                    'id': sample_id,
                    'rag_source_smiles': rag_source_list[idx] if idx < len(rag_source_list) else None,
                    'dyn_fragments': dyn_frags[:10],
                    'reason': reason_components,
                    'rag_contrastive_loss': 0.0,
                    'outcome': 'failure',
                    'success': False,
                }
                if skip_entry and skip_entry.get('exclude_smiles'):
                    record['exclude_smiles'] = skip_entry['exclude_smiles']
                if current_epoch is not None:
                    try:
                        record['epoch'] = int(current_epoch)
                    except Exception:
                        record['epoch'] = current_epoch
                try:
                    if 'rag_negatives_map' in locals() and idx in rag_negatives_map:
                        record['negatives'] = rag_negatives_map[idx]
                except Exception:
                    pass
                deduped_reasons = sorted({r for r in record['reason'] if r})
                record['reason'] = deduped_reasons
                record['failure_reasons'] = deduped_reasons
                zero_every_safe = max(1, int(zero_every)) if zero_every else 1
                zcnt = getattr(self, '_rag_zero_counter', 0)
                if zero_every_safe == 1 or (zcnt % zero_every_safe == 0):
                    zero_records.append(record)
                self._rag_zero_counter = zcnt + 1

        if zero_records and self.rag_zero_detail_log_path and self.rag_enabled:
            try:
                with open(self.rag_zero_detail_log_path, 'a', encoding='utf-8') as f:
                    for rec in zero_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            except Exception:
                pass

        # Mirror the same structure for successful samples so tooling can
        # consume both success and failure events from a single JSONL stream.
        success_records = []
        if self.rag_success_detail_log_path and self.rag_enabled:
            current_epoch = kwargs.get('current_epoch', None)
            for idx, has_pos in enumerate(positive_mask_cpu.tolist()):
                if not has_pos:
                    continue
                sample_id = sample_ids[idx]
                dyn_frags = dyn_fragments_per_sample.get(idx, [])
                record = {
                    'type': 'molecule' if (idx < len(rag_source_list) and rag_source_list[idx]) else 'peptide',
                    'id': sample_id,
                    'rag_source_smiles': rag_source_list[idx] if idx < len(rag_source_list) else None,
                    'dyn_fragments': dyn_frags[:10],
                    'reason': ['positive'],
                    'outcome': 'success',
                    'success': True,
                }
                if current_epoch is not None:
                    try:
                        record['epoch'] = int(current_epoch)
                    except Exception:
                        record['epoch'] = current_epoch
                if idx in rag_negatives_map:
                    record['negatives'] = rag_negatives_map[idx]
                if idx in rag_per_sample_loss:
                    record['rag_contrastive_loss'] = rag_per_sample_loss[idx]
                skip_entry = skip_reason_map.get(idx) if 'skip_reason_map' in locals() else None
                if skip_entry:
                    record['reason'].extend(sorted(skip_entry.get('reasons', [])))
                    if skip_entry.get('exclude_smiles'):
                        record['exclude_smiles'] = skip_entry['exclude_smiles']
                record['reason'] = sorted({r for r in record['reason'] if r})
                record['failure_reasons'] = []
                log_every = getattr(self, 'rag_success_log_every', 1)
                cnt = getattr(self, '_rag_success_counter', 0)
                log_every_safe = max(1, int(log_every)) if log_every else 1
                if log_every_safe == 1 or (cnt % log_every_safe == 0):
                    success_records.append(record)
                self._rag_success_counter = cnt + 1

        combined_records = []
        if zero_records:
            combined_records.extend(zero_records)
        if success_records:
            combined_records.extend(success_records)
        if combined_records and self.rag_success_detail_log_path and self.rag_enabled:
            try:
                with open(self.rag_success_detail_log_path, 'a', encoding='utf-8') as f:
                    for rec in combined_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            except Exception:
                pass

        ligand_total = max(1, int(ligand_repr.shape[0]))
        molecule_total = max(0, rag_source_count)
        peptide_total = max(0, ligand_total - molecule_total)
        loss_dict['rag_positive_pairs'] = torch.tensor(rag_pos_count, device=X.device, dtype=torch.float)
        loss_dict['rag_positive_ratio'] = torch.tensor(rag_pos_count / max(1, float(ligand_total)), device=X.device, dtype=torch.float)
        loss_dict['rag_positive_ratio_molecule'] = torch.tensor((rag_pos_count_molecule / molecule_total) if molecule_total > 0 else 0.0, device=X.device, dtype=torch.float)
        loss_dict['rag_positive_ratio_peptide'] = torch.tensor((rag_pos_count_peptide / peptide_total) if peptide_total > 0 else 0.0, device=X.device, dtype=torch.float)
        loss_dict['rag_zero_pos_molecules'] = torch.tensor(float(rag_zero_mol_count), device=X.device, dtype=torch.float)
        loss_dict['rag_zero_pos_peptides'] = torch.tensor(float(rag_zero_pep_count), device=X.device, dtype=torch.float)

        # Retrieval module disabled: keep retrieval-specific loss at zero.

        total = 0
        for name in self.loss_weights:
            weight = self.loss_weights[name]
            if 'kl_loss' in name: weight *= min(warmup_progress, 1.0)
            total = total + loss_dict[name] * weight
        loss_dict['total'] = total

        # for evaluation
        with torch.no_grad():
            if not self.retrieval_enabled:
                loss_dict.update({
                    'block_type_accu': batch_accu(
                        pred_block_logits[generate_mask],
                        S_gt[generate_mask],
                        batch_ids[generate_mask],
                        reduction='mean'
                    ),
                })
            x2y_accu, y2x_accu = _contrastive_accu(bind_site_repr[contrastive_loss_mask], ligand_repr[contrastive_loss_mask])
            loss_dict.update({
                'bind_site_to_ligand_accu': x2y_accu,
                'ligand_to_bind_site_accu': y2x_accu,
            })
            loss_dict.update({
                'bond_accu': (torch.argmax(pred_bond_logits, dim=-1) == bond_label).long().sum() / (len(bond_label) + 1e-10)
            })
            # record deviation of Zx and centers
            block_centers = scatter_mean(X, block_ids, dim=0, dim_size=block_ids.max() + 1) # [Nblock, 3]
            zx_rmsd = ((block_centers - Zx_mu) ** 2).sum(-1) # [Nblock]
            loss_dict.update({
                'pocket_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[~generate_mask], batch_ids[~generate_mask], dim=0)).mean(),
                'ligand_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[generate_mask], batch_ids[generate_mask], dim=0)).mean(),
            })
            # record norm of Zh
            zh_norm = torch.norm(Zh, dim=-1) # [Nblock]
            loss_dict.update({
                'pocket_zh_norm': zh_norm[~generate_mask].mean(),
                'ligand_zh_norm': zh_norm[generate_mask].mean()
            })
            # record std of Zh and Zx
            zx_std = torch.exp(-torch.abs(signed_Zx_log_var) / 2) # sigma
            loss_dict.update({
                'pocket_zx_std': zx_std[~generate_mask].mean(),
                'ligand_zx_std': zx_std[generate_mask].mean()
            })
        # print(loss_dict)
        return loss_dict
    
    def encode(self, X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=False, binding_site_gen_mask=None, block_ids_override=None):

        batch_ids = length_to_batch_id(lengths)
        block_ids = block_ids_override if block_ids_override is not None else length_to_batch_id(block_lengths)

        # H = self.embedding(S, A, block_ids) #+ self.rel_pos_embedding(position_ids)[block_ids] # [Natom, embed_size]
        H = self.embedding(S, A, block_ids) + self.ctx_embedding(generate_mask[block_ids].long())
        H = self.enc_embed2hidden(H) # [Natom, hidden_size]
        edges, edge_type = self.get_edges(batch_ids, chain_ids, X, block_ids, generate_mask, allow_gen_to_ctx=False, allow_ctx_to_gen=False)
        edge_attr = self.edge_embedding(edge_type)
        attn_mask = self.get_attn_mask(batch_ids, block_ids, generate_mask) # forbid context attending to generated part during encoding

        # make bonds bidirectional
        bond_row, bond_col, bond_type = bonds.T
        Zh_atom, Zx_atom = self.encoder(
            H, X, block_ids, batch_ids, edges, edge_attr,
            topo_edges=torch.stack([
                torch.cat([bond_row, bond_col], dim=0),
                torch.cat([bond_col, bond_row], dim=0)
            ], dim=0),
            topo_edge_attr=self.atom_edge_embedding(torch.cat([bond_type, bond_type], dim=0)),
            attn_mask=attn_mask
        ) # [Natom, hidden_size], [Natom, 3]
        Zh = std_conserve_scatter_mean(Zh_atom, block_ids, dim=0) # [Nblock, hidden_size]
        Zx = scatter_mean(Zx_atom, block_ids, dim=0) # [Nblock, 3], not use std_conserve_scatter_mean for equivariance

        # contrastive learning between Binding Site and Ligand
        batch_size = batch_ids.max() + 1
        bind_site_repr = scatter_mean(Zh[~generate_mask], batch_ids[~generate_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]
        ligand_repr = scatter_mean(Zh[generate_mask], batch_ids[generate_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]

        # rsample
        Zx_prior_mu = scatter_mean(X, block_ids, dim=0) # [Nblock, 3]
        Zx_mu, signed_Zx_log_var = Zx.clone(), self.Wx_log_var(Zh)
        Zh, Zx, Zh_kl_loss, Zx_kl_loss = self.rsample(Zh, Zx, generate_mask, Zx_prior_mu, deterministic, binding_site_gen_mask)

        return Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, bind_site_repr, ligand_repr
    
    def decode_block_type(self, Zh, Zx, chain_ids, lengths, return_hidden: bool = False):
        '''
            Args:
                Zh: [Nblock, latent_size]
                Zx: [Nblock, 3]
                chain_ids: [Nblock]
                lengths: [batch_size]
            Returns:
                pred_block_logits: [Nblock, n_block_type]
                block_h (optional): [Nblock, hidden_size]
        '''
        batch_ids = length_to_batch_id(lengths)
        latent_block_ids = torch.ones_like(batch_ids).cumsum(dim=-1) - 1
        edges, edge_type = self.get_edges(batch_ids, chain_ids, Zx, latent_block_ids, None, True, True)
        edge_attr = self.edge_embedding(edge_type)
        H = self.dec_latent2hidden(Zh)
        block_h, _ = self.decoder(H, Zx, latent_block_ids, batch_ids, edges, edge_attr)
        pred_block_logits = self.block_type_mlp(block_h)
        if return_hidden:
            return pred_block_logits, block_h
        return pred_block_logits

    def decode_structure(self, Zh, X, A, position_ids, topo_edges, topo_edge_attr, chain_ids, batch_ids, block_ids, t):
        '''
            Args:
                Zh: [Nblock, latent_size]
                X: [Natom, 3]
                A: [Natom]
                position_ids: [Nblock], only work for proteins/peptides. For small molecules, they are the same
                topo_edges: [2, Etopo]
                topo_edge_attr: [Etopo, edge_size]
                chain_ids: [Nblock]
                batch_ids: [Nblock]
                block_ids: [Natom]
                t: [Natom]
        '''
        # decode atom-level structures
        edges, edge_type = self.get_edges(batch_ids, chain_ids, X, block_ids, None, True, True)
        edge_attr = self.edge_embedding(edge_type)
        H = self.dec_input_linear(torch.cat([
            self.dec_atom_embedding(A), self.dec_time_embedding(t), self.dec_pos_embedding(position_ids[block_ids]), Zh[block_ids]
        ], dim=-1)) # [Natom, hidden_size]
        H_t, X_next = self.decoder(H, X, block_ids, batch_ids, edges, edge_attr, topo_edges, topo_edge_attr) # [Natom', hidden_size], [Natom', 3]
        
        return H_t, X_next
    
    def _random_mask(self, Zh, generate_mask, batch_ids):
        Zh = Zh.clone()

        mask = _random_mask(batch_ids, sigma=self.decode_mask_ratio)
        mask = mask & generate_mask

        Zh[mask] = self.mask_embedding
        return Zh

    def _sample_from_prior(self, X, Zx_mu, block_ids, generate_mask, t):
        atom_generate_mask = expand_like(generate_mask[block_ids], X)
        Zx_mu = Zx_mu[block_ids]

        # sample random noise (directly use gaussian)
        noise = torch.randn_like(X) * self.prior_coord_std
        
        # sample each atom from block prior (only the generation part)
        X_init = torch.where(atom_generate_mask, Zx_mu + noise, X)

        # vector
        vector = X - X_init

        # state at timestep t (0.0 is the data, 1.0 is the prior)
        X_t = torch.where(atom_generate_mask, X_init + vector * (1 - t)[..., None], X)

        return X_t, vector
    
    @torch.no_grad()
    def _get_inter_block_nbh(self, X_t, batch_ids, block_ids, topo_generate_mask, dist_th):
        # local neighborhood for negative bonds
        row, col = fully_connect_edges(batch_ids[block_ids])

        # inter-block and at least one end is in topo generation part, and row < col to avoid repeated bonds
        select_mask = (block_ids[row] != block_ids[col]) & (topo_generate_mask[block_ids[row]] | topo_generate_mask[block_ids[col]]) * (row < col)
        row, col = row[select_mask], col[select_mask]

        # get edges within 3.5A
        select_mask = torch.norm(X_t[row] - X_t[col], dim=-1) < dist_th
        row, col = row[select_mask], col[select_mask]

        return torch.stack([row, col], dim=0) # [2, E]

    @torch.no_grad()
    def _get_bond_to_pred(self, X_t, gt_bonds, batch_ids, block_ids, generate_mask, neg_dist_th=3.5, neg_to_pos_ratio=1.0, given_gt=None):

        if given_gt is None:
            # get ground truth
            gt_row, gt_col, gt_type = gt_bonds[:, 0], gt_bonds[:, 1], gt_bonds[:, 2]
            # inter-block and at least one end is in generation part
            select_mask = (block_ids[gt_row] != block_ids[gt_col]) & (generate_mask[block_ids[gt_row]] | generate_mask[block_ids[gt_col]])
            gt_row, gt_col, gt_type = gt_row[select_mask], gt_col[select_mask], gt_type[select_mask]
        else:
            gt_row, gt_col = given_gt[0]
            gt_type = given_gt[1]

        # local neighborhood for negative bonds
        row, col = self._get_inter_block_nbh(X_t, batch_ids, block_ids, generate_mask, neg_dist_th)

        # negative sampling from local neighborhood (low possibility to coincide with postive bonds)
        if len(row) == 0: ratio = 0.1
        else: ratio = len(gt_row) / len(row) * neg_to_pos_ratio # neg:pos ~ 2:1
        select_mask = torch.rand_like(row, dtype=torch.float) < ratio
        row, col = row[select_mask], col[select_mask]
        neg_type = torch.zeros_like(row, dtype=torch.long)

        bonds_to_pred = torch.stack([
            torch.cat([gt_row, row], dim=0),
            torch.cat([gt_col, col], dim=0),
        ], dim=0)
        labels = torch.cat([gt_type, neg_type])

        return bonds_to_pred, labels
    
    @torch.no_grad()
    def get_edges(self, batch_ids, segment_ids, Z, block_ids, generate_mask, allow_gen_to_ctx, allow_ctx_to_gen):
        row, col = fully_connect_edges(batch_ids)
        if not allow_gen_to_ctx: # forbid message passing from generated part to context
            select_mask = generate_mask[row] | (~generate_mask[col]) # src is in generated part or dst is not in generated part
            row, col = row[select_mask], col[select_mask]
        if not allow_ctx_to_gen: # forbid message passing from context to generated part
            select_mask = (~generate_mask[row]) | (generate_mask[col])
            row, col = row[select_mask], col[select_mask]
        is_intra = segment_ids[row] == segment_ids[col]
        intra_edges = torch.stack([row[is_intra], col[is_intra]], dim=0)
        inter_edges = torch.stack([row[~is_intra], col[~is_intra]], dim=0)
        intra_edges = knn_edges(block_ids, batch_ids, Z.unsqueeze(1), self.k_neighbors, intra_edges)
        inter_edges = knn_edges(block_ids, batch_ids, Z.unsqueeze(1), self.k_neighbors, inter_edges)
        
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        edge_type = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])

        return edges, edge_type
    
    @torch.no_grad()
    def get_attn_mask(self, batch_ids, block_ids, generate_mask):
        '''
            Args:
                batch_ids: [Nblock]
                block_ids: [Natom]
                generate_mask: [Nblock]
            Returns:
                attn_mask: [bs, max_n_atom, max_n_atom]
        '''
        generate_mask = generate_mask[block_ids] # [Natom]
        mask, _ = graph_to_batch_nx(generate_mask, batch_ids[block_ids], padding_value=True, factor_req=8) # [bs, max_n]
        bs, N = mask.shape
        
        # Create base attention mask allowing all tokens to attend to all other tokens
        attention_mask = torch.ones(bs, N, N, dtype=torch.bool, device=mask.device)

        # Create symmetric restriction: no attention between context and generated tokens
        context_to_generated = (mask == 0).unsqueeze(2) & (mask == 1).unsqueeze(1)
        generated_to_context = (mask == 1).unsqueeze(2) & (mask == 0).unsqueeze(1)
    
        # Remove attention from context to generated and vice versa
        attention_mask = attention_mask & ~(context_to_generated | generated_to_context)

        return attention_mask

    @torch.no_grad()
    def _get_topo_edges(self, bonds, block_ids, generate_mask):
        '''
        Only used in training
            bonds: [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            block_ids: [N]
            generate_mask: [Nblock]
        '''
        row, col, bond_type = bonds.T
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0) # bidirectional
        bond_type = torch.cat([bond_type, bond_type], dim=-1) #[2*Nbonds, edge_size]

        # for atom-level chemical bonds: only give intra-block ones for generated part, but all for context
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block == col_block) | ((~generate_mask[row_block]) & (~generate_mask[col_block]))
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]
        topo_edges = torch.stack([row, col], dim=0)

        return topo_edges, bond_type
    
    @torch.no_grad()
    def _unmask_inter_topo_edges(self, bonds, batch_ids, block_ids, generate_mask):
        atom_batch_ids = batch_ids[block_ids]
        row, col, bond_type = bonds.T

        # get inter-block bonds
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block != col_block) & (generate_mask[row_block] | generate_mask[col_block])
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]

        # # sample some to provide as contexts, others for prediction
        # unmask_ratio = torch.rand(batch_ids.max() + 1, device=bonds.device)
        # select_mask = torch.rand_like(atom_batch_ids[row], dtype=torch.float) < unmask_ratio[atom_batch_ids[row]]
        # 50% cases for structure prediction, others for design
        unmask = torch.rand(batch_ids.max() + 1, device=bonds.device) < 0.5
        select_mask = unmask[atom_batch_ids[row]]

        # bi-directional
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row])
        bond_type = torch.cat([bond_type, bond_type], dim=0)
        select_mask = torch.cat([select_mask, select_mask], dim=0)

        return torch.stack([row, col], dim=0), bond_type, select_mask

    def rsample(self, Zh, Zx, generate_mask, Zx_prior_mu=None, deterministic=False, binding_site_gen_mask=None):
        '''
            Zh: [Nblock, latent_size]
            Zx: [Nblock, 3]
            Zx_prior_mu: [Nblock, 3]
        '''

        if binding_site_gen_mask is not None: generate_mask = generate_mask | binding_site_gen_mask

        # if hasattr(self, 'kl_on_pocket') and self.kl_on_pocket: # also exert kl regularizations on latent points of pocket
        if self.kl_on_pocket: # also exert kl regularizations on latent points of pocket
            generate_mask = torch.ones_like(generate_mask)

        # data_size = Zh.shape[0]
        data_size = generate_mask.long().sum()

        # invariant latent features
        Zh_mu = self.Wh_mu(Zh)
        Zh_log_var = -torch.abs(self.Wh_log_var(Zh)) #Following Mueller et al., z_log_var is log(\sigma^2))
        Zh_kl_loss = -0.5 * torch.sum((1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var))[generate_mask]) / (data_size * Zh_mu.shape[-1])
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)

        # equivariant latent features
        if Zx_prior_mu is None:
            Zx_sampled, Zx_kl_loss = Zx + torch.randn_like(Zx), 0 # fix as standard gaussian
        else:
            Zx_mu_delta = Zx - Zx_prior_mu # [Nblock, 3], if perfectly from prior, the expectation should be zero
            Zx_log_var = -torch.abs(self.Wx_log_var(Zh)).expand_as(Zx)
            Zx_kl_loss = -0.5 * torch.sum((1.0 + Zx_log_var - Zx_mu_delta * Zx_mu_delta - torch.exp(Zx_log_var))[generate_mask]) / (data_size * Zx.shape[-1])
            Zx_sampled = Zx if deterministic else Zx + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx)
        
        return Zh_sampled, Zx_sampled, Zh_kl_loss, Zx_kl_loss
    
    def _init_atoms(
            self,
            pred_block_type,
            X,
            A,
            bonds,
            Zx_mu,
            block_ids,
            generate_mask,
            fragment_smiles: Optional[List[Optional[str]]] = None,
            topo_generate_mask=None
        ):
        
        gt_bonds = bonds.clone()
        if topo_generate_mask is None:
            topo_generate_mask = generate_mask

        # isolate context bonds
        ctx_block = (~generate_mask[block_ids[bonds[:, 0]]]) & (~generate_mask[block_ids[bonds[:, 1]]])
        bonds_ctx = bonds[ctx_block]

        # extract context atoms
        atom_ctx_mask = ~generate_mask[block_ids]
        ctx_X = X[atom_ctx_mask]
        ctx_A = A[atom_ctx_mask]
        ctx_block_ids = block_ids[atom_ctx_mask]

        # remap context bonds to local indices
        ctx_atom_order_map = -torch.ones_like(block_ids, dtype=torch.long)
        ctx_atom_order_map[atom_ctx_mask] = torch.arange(atom_ctx_mask.long().sum(), device=atom_ctx_mask.device)
        bonds_ctx = torch.stack([
            ctx_atom_order_map[bonds_ctx[:, 0]],
            ctx_atom_order_map[bonds_ctx[:, 1]],
            bonds_ctx[:, 2]
        ], dim=-1)

        gen_block_indices = torch.nonzero(generate_mask, as_tuple=False).view(-1)

        gen_atom_types_tensors: List[torch.Tensor] = []
        gen_block_ids_tensors: List[torch.Tensor] = []
        gen_bonds_local: List[torch.Tensor] = []

        for block_idx in gen_block_indices.tolist():
            use_fragment = False
            fragment = None
            if fragment_smiles is not None and block_idx < len(fragment_smiles):
                fragment = fragment_smiles[block_idx]
                if fragment:
                    use_fragment = True
            if not topo_generate_mask[block_idx]:
                use_fragment = False

            if use_fragment:
                atom_types_tensor, bonds_tensor = self._fragment_to_atoms(fragment, X.device)
            else:
                atom_types_tensor, bonds_tensor = self._extract_block_atoms_from_data(block_idx, block_ids, A, gt_bonds, X.device)

            if (atom_types_tensor is None or atom_types_tensor.numel() == 0) and not use_fragment:
                fallback_types, fallback_block_ids, fallback_bonds = block_to_atom_map(
                    pred_block_type[block_idx:block_idx + 1],
                    torch.tensor([block_idx], device=pred_block_type.device)
                )
                atom_types_tensor = fallback_types.to(device=X.device)
                bonds_tensor = fallback_bonds.to(device=X.device)

            if atom_types_tensor is None or atom_types_tensor.numel() == 0:
                continue

            gen_atom_types_tensors.append(atom_types_tensor)
            gen_block_ids_tensors.append(torch.full((atom_types_tensor.shape[0],), block_idx, dtype=torch.long, device=X.device))
            if bonds_tensor is None:
                bonds_tensor = torch.empty((0, 3), dtype=torch.long, device=X.device)
            gen_bonds_local.append(bonds_tensor)

        if gen_atom_types_tensors:
            gen_A = torch.cat(gen_atom_types_tensors, dim=0)
            gen_block_ids = torch.cat(gen_block_ids_tensors, dim=0)
            counts = torch.tensor([tensor.shape[0] for tensor in gen_atom_types_tensors], dtype=torch.long, device=X.device)
            offsets = torch.cumsum(counts, dim=0) - counts
            bond_list = []
            for bonds_tensor, offset in zip(gen_bonds_local, offsets):
                if bonds_tensor.numel() == 0:
                    continue
                bond_list.append(torch.stack([
                    bonds_tensor[:, 0] + offset,
                    bonds_tensor[:, 1] + offset,
                    bonds_tensor[:, 2]
                ], dim=-1))
            gen_bonds = torch.cat(bond_list, dim=0) if bond_list else torch.empty((0, 3), dtype=torch.long, device=X.device)
        else:
            gen_A = torch.empty((0,), dtype=torch.long, device=X.device)
            gen_block_ids = torch.empty((0,), dtype=torch.long, device=X.device)
            gen_bonds = torch.empty((0, 3), dtype=torch.long, device=X.device)

        gen_X = Zx_mu[gen_block_ids] + torch.randn_like(Zx_mu[gen_block_ids]) * self.prior_coord_std if gen_block_ids.numel() else torch.empty((0, 3), dtype=X.dtype, device=X.device)

        X_combined = torch.cat([ctx_X, gen_X], dim=0)
        A_combined = torch.cat([ctx_A, gen_A], dim=0)
        block_ids_combined = torch.cat([ctx_block_ids, gen_block_ids], dim=0)

        ctx_row, ctx_col, ctx_bond_type = bonds_ctx[:, 0], bonds_ctx[:, 1], bonds_ctx[:, 2]
        if gen_bonds.numel():
            gen_row = gen_bonds[:, 0] + ctx_A.shape[0]
            gen_col = gen_bonds[:, 1] + ctx_A.shape[0]
            gen_bond_type = gen_bonds[:, 2]
            bonds_combined = torch.stack([
                torch.cat([ctx_row, ctx_col, gen_row, gen_col], dim=0),
                torch.cat([ctx_col, ctx_row, gen_col, gen_row], dim=0),
                torch.cat([ctx_bond_type, ctx_bond_type, gen_bond_type, gen_bond_type], dim=0)
            ], dim=-1)
        else:
            bonds_combined = torch.stack([
                torch.cat([ctx_row, ctx_col], dim=0),
                torch.cat([ctx_col, ctx_row], dim=0),
                torch.cat([ctx_bond_type, ctx_bond_type], dim=0)
            ], dim=-1)

        block_ids_sorted, perm = scatter_sort(block_ids_combined, block_ids_combined, dim=0)
        X_sorted = X_combined[perm]
        A_sorted = A_combined[perm]
        atom_order_map = torch.ones_like(A_sorted, dtype=torch.long)
        atom_order_map[perm] = torch.arange(len(A_sorted), device=A_sorted.device)
        bonds_sorted = torch.stack([
            atom_order_map[bonds_combined[:, 0]],
            atom_order_map[bonds_combined[:, 1]],
            bonds_combined[:, 2]
        ], dim=-1)

        return X_sorted, A_sorted, block_ids_sorted, bonds_sorted

    def _fragment_to_atoms(self, smiles: Optional[str], device: torch.device):
        if smiles is None:
            return None, None
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None
        if mol is None:
            return None, None
        atom_types: List[int] = []
        valid_atom_indices: List[int] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            valid_atom_indices.append(atom.GetIdx())
            atom_types.append(VOCAB.atom_to_idx(atom.GetSymbol()))
        if not atom_types:
            return None, None
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_atom_indices)}
        bonds: List[List[int]] = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin not in idx_map or end not in idx_map:
                continue
            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:
                bond_type = 1
            elif bt == Chem.rdchem.BondType.DOUBLE:
                bond_type = 2
            elif bt == Chem.rdchem.BondType.TRIPLE:
                bond_type = 3
            elif bt == Chem.rdchem.BondType.AROMATIC:
                bond_type = 4
            else:
                bond_type = 1
            bonds.append([idx_map[begin], idx_map[end], bond_type])
        atom_tensor = torch.tensor(atom_types, dtype=torch.long, device=device)
        bond_tensor = torch.tensor(bonds, dtype=torch.long, device=device) if bonds else torch.empty((0, 3), dtype=torch.long, device=device)
        return atom_tensor, bond_tensor

    def _extract_block_atoms_from_data(self, block_idx: int, block_ids: torch.Tensor, atom_types: torch.Tensor, bonds: torch.Tensor, device: torch.device):
        mask = (block_ids == block_idx)
        atom_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if atom_indices.numel() == 0:
            return None, None
        atom_tensor = atom_types[atom_indices].to(device=device)
        index_map = torch.full((block_ids.shape[0],), -1, dtype=torch.long, device=device)
        index_map[atom_indices] = torch.arange(atom_indices.shape[0], device=device)
        bond_mask = mask[bonds[:, 0]] & mask[bonds[:, 1]]
        bonds_local = bonds[bond_mask]
        if bonds_local.numel():
            bonds_local = torch.stack([
                index_map[bonds_local[:, 0]],
                index_map[bonds_local[:, 1]],
                bonds_local[:, 2]
            ], dim=-1)
        else:
            bonds_local = torch.empty((0, 3), dtype=torch.long, device=device)
        return atom_tensor, bonds_local

    def _bond_length_guidance(self, t, H_t, X_t, batch_ids, block_ids, generate_mask, dist_th=3.5, bond_th=0.9):
        
        # get inter-block bonding distribution
        row, col = self._get_inter_block_nbh(X_t, batch_ids, block_ids, generate_mask, dist_th=dist_th)
        pred_bond_logits = self.bond_type_mlp(H_t[row] + H_t[col]) # [E, 5], commutative property
        pred_bond_probs = F.softmax(pred_bond_logits, dim=-1) # [E, 5]
        has_bond_mask = torch.argmax(pred_bond_probs, dim=-1) != 0
        pred_bond_probs = pred_bond_probs[has_bond_mask] # [E', 5], not None bond
        row, col = row[has_bond_mask], col[has_bond_mask]
        bond_prob, bond_type = torch.max(pred_bond_probs, dim=-1)
        
        bond_select_mask = (bond_prob > bond_th) & (row < col)
        row, col, bond_type = row[bond_select_mask], col[bond_select_mask], bond_type[bond_select_mask]
        bond_prob = bond_prob[bond_select_mask]

        # get approaching vector
        BOND_DIST = 1.6
        relative_x = X_t[col] - X_t[row]   # [E, 3]
        relative_dist = torch.norm(relative_x, dim=-1) # [E]
        relative_x = relative_x / (relative_dist[:, None] + 1e-10)
        approaching_speed = (relative_dist - BOND_DIST) * 0.5 # a->b and b->a, therefore 0.5
        approaching_speed = approaching_speed * bond_prob
        v = torch.where(approaching_speed > 0, approaching_speed, torch.zeros_like(approaching_speed))[:, None] * relative_x

        # aggregation
        block_row = block_ids[torch.cat([row, col], dim=0)]
        v = torch.cat([v, -v], dim=0)
        aggr_v = scatter_sum(v, block_row, dim=0, dim_size=block_ids.max() + 1)   # [Nblock]
        aggr_v = aggr_v[block_ids]  # [Natom]

        # weights
        w = min(t / (1 - t + 1e-10), 10)

        return w * aggr_v
    
    def generate(
        self,
        X,                  # [Natom, 3], atom coordinates     
        S,                  # [Nblock], block types
        A,                  # [Natom], atom types
        bonds,              # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
        position_ids,       # [Nblock], block position ids
        chain_ids,          # [Nblock], split different chains
        generate_mask,      # [Nblock], 1 for generation, 0 for context
        block_lengths,      # [Nblock], number of atoms in each block
        lengths,            # [batch_size]
        is_aa,              # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        n_iter=10,          # number of iterations
        fixseq=False,       # whether to only predict the structure
        return_x_only=False,# return x only (used in validation)
        topo_generate_mask=None,
        **kwargs
    ):
        dyn_block_lengths = kwargs.pop('dyn_block_lengths', None)
        dyn_block_types = kwargs.pop('dyn_block_types', None)
        dyn_generate_mask = kwargs.pop('dyn_generate_mask', None)
        dyn_block_ids = kwargs.pop('dyn_block_ids', None)
        dyn_chain_ids = kwargs.pop('dyn_chain_ids', None)
        dyn_position_ids = kwargs.pop('dyn_position_ids', None)
        dyn_is_aa = kwargs.pop('dyn_is_aa', None)
        dyn_num_blocks = kwargs.pop('dyn_num_blocks', None)
        dyn_fragment_smiles = kwargs.pop('dyn_fragment_smiles', None)

        block_lengths_device, block_lengths_dtype = block_lengths.device, block_lengths.dtype
        chain_ids_device, chain_ids_dtype = chain_ids.device, chain_ids.dtype
        position_ids_device, position_ids_dtype = position_ids.device, position_ids.dtype
        generate_mask_device = generate_mask.device
        S_device, S_dtype = S.device, S.dtype
        is_aa_device, is_aa_dtype = is_aa.device, is_aa.dtype

        block_ids_override = None
        if dyn_block_ids is not None and dyn_block_ids.numel():
            block_ids_override = dyn_block_ids.to(device=X.device, dtype=torch.long)

        if dyn_block_lengths is not None and dyn_block_lengths.numel():
            block_lengths = dyn_block_lengths.to(device=block_lengths_device, dtype=block_lengths_dtype)
            if dyn_block_types is not None and dyn_block_types.numel():
                S = dyn_block_types.to(device=S_device, dtype=S_dtype)
            if dyn_generate_mask is not None and dyn_generate_mask.numel():
                generate_mask = dyn_generate_mask.to(device=generate_mask_device)
            if dyn_chain_ids is not None and dyn_chain_ids.numel():
                chain_ids = dyn_chain_ids.to(device=chain_ids_device, dtype=chain_ids_dtype)
            if dyn_position_ids is not None and dyn_position_ids.numel():
                position_ids = dyn_position_ids.to(device=position_ids_device, dtype=position_ids_dtype)
            if dyn_is_aa is not None and dyn_is_aa.numel():
                is_aa = dyn_is_aa.to(device=is_aa_device, dtype=is_aa_dtype)
            if dyn_num_blocks is not None and dyn_num_blocks.numel():
                lengths = dyn_num_blocks.to(device=lengths.device, dtype=lengths.dtype)

        # if self.discrete_timestep: assert n_iter == self.default_num_steps
        if 'given_latent' in kwargs:
            Zh, Zx, _ = kwargs.pop('given_latent')
            # Zx_log_var = -torch.abs(self.Wx_log_var(Zh)).view(*Zx.shape)
        else:
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=True, block_ids_override=block_ids_override
            ) # [Nblock, d_latent], [Nblock, 3], [1], [1]
        block_ids = block_ids_override if block_ids_override is not None else length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        total_blocks = int(block_lengths.shape[0])
        fragment_smiles_list: List[Optional[str]] = [None] * total_blocks
        if dyn_fragment_smiles is not None:
            for idx, smi in enumerate(dyn_fragment_smiles):
                if idx < total_blocks:
                    fragment_smiles_list[idx] = smi

        # start_t = 0.5
        if not fixseq:
            # decode block types from latent graph
            pred_block_logits, block_hidden = self.decode_block_type(Zh, Zx, chain_ids, lengths, return_hidden=True)
            # mask non aa positions if is_aa == True
            non_aa_mask = ~torch.tensor(VOCAB.aa_mask, dtype=torch.bool, device=is_aa.device)
            pred_block_logits = pred_block_logits.masked_fill(non_aa_mask[None, :] & is_aa[:, None], float('-inf'))
            pred_block_prob = torch.softmax(pred_block_logits, dim=-1) # [Nblock, num_block_type]
            prob, pred_block_type = torch.max(pred_block_prob, dim=-1) # [Nblock]
            pred_block_type[~topo_generate_mask] = S[~topo_generate_mask]

            gen_block_indices = torch.nonzero(generate_mask, as_tuple=False).view(-1)

            # initialize (append atoms and sample coordinates)
            X_t, A, block_ids, bonds = self._init_atoms(
                pred_block_type, X, A, bonds, Zx, block_ids, generate_mask,
                fragment_smiles=fragment_smiles_list, topo_generate_mask=topo_generate_mask
            )
        else:
            pred_block_type = S
            # only need to initialize atoms
            random_X = Zx[block_ids] + torch.randn_like(Zx[block_ids]) * self.prior_coord_std
            X_t = torch.where(expand_like(generate_mask[block_ids], X), random_X, X)
            # for consistency, inter-block bonds of generation parts should be removed
            intra_block_mask = block_ids[bonds[:, 0]] == block_ids[bonds[:, 1]]
            ctx_bond_mask = (~generate_mask[block_ids][bonds[:, 0]]) & (~generate_mask[block_ids][bonds[:, 1]])
            select_bond_mask = intra_block_mask | ctx_bond_mask
            bonds = bonds[select_bond_mask]
            # bidirectional
            _row, _col, _type = bonds[:, 0], bonds[:, 1], bonds[:, 2]
            bonds = torch.stack([
                torch.cat([_row, _col], dim=0),
                torch.cat([_col, _row], dim=0),
                torch.cat([_type, _type], dim=0)
            ], dim=1)
        
        # concat context bonds and generated bonds
        topo_edge_type = bonds[:, 2]
        topo_edges, topo_edge_attr = bonds[:, :2].T, self.atom_edge_embedding(topo_edge_type)

        # iterative
        X_init = X_t.clone()
        all_vectors, span = [], 1.0 / n_iter
        X_gen_mask = expand_like(generate_mask[block_ids], X_t)

        topo_edges_add, topo_edge_attr_add = topo_edges, topo_edge_attr

        for i in range(n_iter):
            t = (1.0 - i * span) * torch.ones_like(block_ids, dtype=torch.float)
            H_t, X_next = self.decode_structure(Zh, X_t, A, position_ids, topo_edges_add, topo_edge_attr_add, chain_ids, batch_ids, block_ids, t)
            pred_vector = torch.where(
                X_gen_mask,
                X_next - X_t,
                torch.zeros_like(X_t))
            X_t = X_t + pred_vector * span # update
            X_t = _avoid_clash(A, X_t, batch_ids, block_ids, chain_ids, generate_mask, is_aa)
            all_vectors.append(pred_vector)

        X = X_t

        if return_x_only:
            return X
        # VLB for iterative process (the smaller, the better)
        ll = ((X_t - X_init).unsqueeze(0) - torch.stack(all_vectors, dim=0)) ** 2 # [T, Natom, 3]
        ll = ll.sum(-1).mean(0) # [Natom]

        # bonds
        row, col = self._get_inter_block_nbh(X, batch_ids, block_ids, topo_generate_mask, dist_th=3.5)
        pred_bond_logits = self.bond_type_mlp(H_t[row] + H_t[col]) # [E, 5], commutative property
        pred_bond_probs = F.softmax(pred_bond_logits, dim=-1) # [E, 5]
        has_bond_mask = torch.argmax(pred_bond_probs, dim=-1) != 0
        pred_bond_probs = pred_bond_probs[has_bond_mask] # [E', 5], not None bond
        # predicted bonds
        row, col = row[has_bond_mask], col[has_bond_mask]
        bond_prob, bond_type = torch.max(pred_bond_probs, dim=-1)
        # topo-fix bonds
        topo_fix_mask = (~topo_generate_mask) & generate_mask
        topo_inter_mask = block_ids[bonds[:, 0]] != block_ids[bonds[:, 1]]
        topo_fix_bonds = bonds[topo_inter_mask & topo_fix_mask[block_ids[bonds[:, 0]]] & topo_fix_mask[block_ids[bonds[:, 1]]]] # [Efix, 3]
        topo_fix_bonds = topo_fix_bonds[topo_fix_bonds[:, 0] < topo_fix_bonds[:, 1]] # avoid repeated bonds
        row = torch.cat([row, topo_fix_bonds[:, 0]], dim=0)
        col = torch.cat([col, topo_fix_bonds[:, 1]], dim=0)
        bond_type = torch.cat([bond_type, topo_fix_bonds[:, 2]], dim=0)
        bond_prob = torch.cat([bond_prob, torch.ones_like(topo_fix_bonds[:, 2], dtype=torch.float)], dim=0)
        # concat prob and distance
        bond_prob = torch.stack([bond_prob, torch.norm(X[row] - X[col], dim=-1)], dim=-1) # [E, 2]

        # intra block bonds for generated part
        intra_block_bond_mask = generate_mask[block_ids[topo_edges[0]]] & generate_mask[block_ids[topo_edges[1]]] # in generation
        intra_block_bond_mask = intra_block_bond_mask & (block_ids[topo_edges[0]] == block_ids[topo_edges[1]]) # in the same block
        intra_block_bond_mask = intra_block_bond_mask & (topo_edges[0] < topo_edges[1])  # avoid redundance
        intra_row, intra_col = topo_edges[0][intra_block_bond_mask], topo_edges[1][intra_block_bond_mask]
        intra_bond_type = topo_edge_type[intra_block_bond_mask]

        # get results
        batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = [], [], [], [], [], []
        batch_ids = length_to_batch_id(lengths)
        for i, l in enumerate(lengths):
            batch_X.append([])
            batch_A.append([])
            batch_ll.append([])
            batch_intra_bonds.append([])
            cur_mask = (batch_ids == i) & generate_mask # [Nblock]
            cur_mask = cur_mask[block_ids] # [Natom]
            cur_atom_type, cur_atom_coord, cur_atom_ll = A[cur_mask], X[cur_mask], ll[cur_mask]
            cur_block_ids = block_ids[cur_mask] # [Natom']

            for j in range(cur_block_ids.min(), cur_block_ids.max() + 1):
                batch_X[-1].append(cur_atom_coord[cur_block_ids == j].tolist())
                batch_A[-1].append(cur_atom_type[cur_block_ids == j].tolist())
                batch_ll[-1].append(cur_atom_ll[cur_block_ids == j].tolist())

            batch_S.append(pred_block_type[generate_mask & (batch_ids == i)].tolist())

            # get bonds (inter-block)
            global2local = -torch.ones_like(cur_mask, dtype=torch.long) # set non-related to -1 for later check
            global2local[cur_mask] = torch.arange(cur_mask.long().sum(), device=cur_mask.device) # assume atoms sorted by block ids
            bond_mask = cur_mask[row] & cur_mask[col]
            local_row, local_col = global2local[row[bond_mask]], global2local[col[bond_mask]]
            assert not torch.any(local_row == -1)
            assert not torch.any(local_col == -1)
            batch_bonds.append((local_row.tolist(), local_col.tolist(), bond_prob[bond_mask].tolist(), bond_type[bond_mask].tolist()))
            # get bonds (intra-block)
            block_offsets = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0).cumsum(dim=0)
            block_offsets = F.pad(block_offsets[:-1], pad=(1, 0), value=0)
            for j in range(cur_block_ids.min(), cur_block_ids.max() + 1):
                bond_mask = cur_mask[intra_row] & cur_mask[intra_col] & (block_ids[intra_row] == j)
                local_row = intra_row[bond_mask] - block_offsets[block_ids[intra_row[bond_mask]]]
                local_col = intra_col[bond_mask] - block_offsets[block_ids[intra_col[bond_mask]]]
                batch_intra_bonds[-1].append((local_row.tolist(), local_col.tolist(), intra_bond_type[bond_mask].tolist()))

        return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds # inter-block bonds and intra-block bonds
