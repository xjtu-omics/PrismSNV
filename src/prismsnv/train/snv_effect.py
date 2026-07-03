import argparse
import datetime
import os
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from anndata import AnnData, read_h5ad
from numba.core.errors import NumbaPendingDeprecationWarning
from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings(
    "ignore",
    category=NumbaPendingDeprecationWarning
)

try:
    from .utility import (
        DEFAULT_PAIR_CHUNK,
        RESOLUTION,
        _align_celltypes_for_scores,
        _distributed_env_ready,
        _encode_latent_representation,
        _filter_snv_to_standard_chromosomes,
        _normalise_score_column_minmax,
        _prepare_annotation_dataframe,
        _prepare_celltype_score_lookup,
        _read_annotation_table,
        _strip_distributed_prefix,
        align_adata_rna_with_genes,
        batch_score_all_snvs,
        batch_score_snvs_by_cell,
        cluster_cells,
        compute_cell_perturbation_scores,
        filter_highly_cooccurring_snvs_by_celltype,
        filter_snv_by_counts,
        get_device,
        log,
        log_model_parameter_summary,
        make_lazy_collate_fn,
        LazyAnndataDataset,
        plot_cell_perturbation_umap,
        rank_snvs_by_attention,
        resolve_eval_only_checkpoint,
        score_by_celltype,
        set_seed,
        tensors_from_anndata,
        chunk_iter,
        warmup_cosine_lr,
    )
except ImportError:
    from utility import (
        DEFAULT_PAIR_CHUNK,
        RESOLUTION,
        _align_celltypes_for_scores,
        _distributed_env_ready,
        _encode_latent_representation,
        _filter_snv_to_standard_chromosomes,
        _normalise_score_column_minmax,
        _prepare_annotation_dataframe,
        _prepare_celltype_score_lookup,
        _read_annotation_table,
        _strip_distributed_prefix,
        align_adata_rna_with_genes,
        batch_score_all_snvs,
        batch_score_snvs_by_cell,
        cluster_cells,
        compute_cell_perturbation_scores,
        filter_highly_cooccurring_snvs_by_celltype,
        filter_snv_by_counts,
        get_device,
        log,
        log_model_parameter_summary,
        make_lazy_collate_fn,
        LazyAnndataDataset,
        plot_cell_perturbation_umap,
        rank_snvs_by_attention,
        resolve_eval_only_checkpoint,
        score_by_celltype,
        set_seed,
        tensors_from_anndata,
        chunk_iter,
        warmup_cosine_lr,
    )

# Utilities for transferring a pre-trained RNA backbone
try:  # pragma: no cover - import resolution differs for package/script usage
    from .pre_train import (
        safe_load_backbone_into_snv_model,
        load_backbone_into_snv_model,
        freeze_encoder_only,
    )
except ImportError:  # pragma: no cover - fallback when executed as a script
    from pre_train import (  # type: ignore
        safe_load_backbone_into_snv_model,
        load_backbone_into_snv_model,
        freeze_encoder_only,
    )


# Utilities
SCORE_COLUMN = "score_euclidean"  # score_cosine_distance / score_euclidean
COSINE_DISTANCE_SCORE_COLUMN = "score_cosine_distance"


RANK_LOSS_ENABLED = True
RANK_LAMBDA_MAX = 0.05
RANK_MARGIN_MAX = 0.05
RANK_WARMUP_EPOCHS = 10
RANK_MIN_DIFF_SNV = 5
RANK_START_EPOCH = 11


def _save_snv_name_array(path: str, snv_names: Iterable[str], label: str) -> None:
    """Save the SNV order used by a checkpoint as a plain NumPy string array."""

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    snv_array = np.asarray([str(name) for name in snv_names], dtype=str)
    np.save(path, snv_array)
    log(f"[INFO] Saved {label} ({snv_array.size} SNVs) to: {path}")


def _setup_distributed_training(device: Optional[torch.device] = None) -> Tuple[
    bool, int, int, torch.device
]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for snv_effect.py. Please run this workflow in a CUDA-enabled environment."
        )

    rank = 0
    world_size = 1
    use_distributed = False

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    can_initialise = (
        dist.is_available()
        and torch.cuda.is_available()
        and _distributed_env_ready()
        and env_world_size > 1
    )

    if can_initialise and not dist.is_initialized():
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                timeout=datetime.timedelta(hours=5))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_distributed = True
    if device is None:
        device = get_device("cuda")

    return use_distributed, rank, world_size, device


class SNVPerturbationModel(nn.Module):
    """
    Attention-based SNV perturbation model.
    Training: dense attention decode (batch-level).
    Inference: sparse decode to save memory (avoids [batch, n_snvs, *] tensors).
    """

    def __init__(
        self,
        n_genes: int,
        n_snvs: int,
        latent_dim: int = 128,
        snv_emb_dim: int = 64,
        n_batches: int = 1,
        batch_emb_dim: int = 16,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_snvs = n_snvs
        self.latent_dim = latent_dim
        self.snv_emb_dim = snv_emb_dim
        self.n_batches = max(1, int(n_batches))
        self.batch_emb_dim = int(batch_emb_dim)

        # Encoder q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_log_var = nn.Linear(256, latent_dim)

        # SNV embedding & attention head
        self.snv_embedding = nn.Embedding(n_snvs, snv_emb_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(latent_dim + snv_emb_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        # Conditional embedding module: maps the concatenated latent state [z, e_j]
        # into a context-aware SNV embedding e_{j|z} that reflects the current cell.
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + self.snv_emb_dim, self.snv_emb_dim),
            nn.SiLU(),  # SiLU yields smoother gradients
            nn.Dropout(0.1),
            nn.Linear(self.snv_emb_dim, self.snv_emb_dim),
        )

        # Project SNV-embedding contribution to latent space
        self.snv_proj = nn.Linear(snv_emb_dim, latent_dim)

        # Decoder p(x|z)
        self.batch_emb = nn.Embedding(self.n_batches, self.batch_emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.batch_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_genes),
        )

        # Gene-wise dispersion (stable via softplus)
        self.raw_theta = nn.Parameter(
            torch.zeros(n_genes)
        )  # theta = softplus(raw_theta)

    def encode(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_representation = self.encoder(input_tensor)
        mean = self.encoder_mu(hidden_representation)
        log_variance = self.encoder_log_var(hidden_representation)
        return mean, log_variance

    def reparameterize(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        std_dev = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(std_dev)
        return mean + noise * std_dev

    def _decode_train_sparse(
        self,
        z: torch.Tensor,
        G: torch.Tensor,
        batch_indices: torch.Tensor,
        return_delta: bool = False,
    ):
        """
        Sparse training decoder:
        - Only compute attention and embeddings for SNVs with G != 0.
        - Avoid constructing a massive tensor of shape [B, S, latent].
        - Use small per-cell loops for stable softmax (batch size is usually a few
          hundred, so the overhead is minimal).
        """
        batch_size, num_snvs = G.shape
        device = z.device

        batch_embedding = self.batch_emb(
            batch_indices.to(device=device, dtype=torch.long)
        )

        # locate all non-zero SNVs
        nz_idx = torch.nonzero(G, as_tuple=False)  # [num_cell_snv_pairs, 2] where each row is (cell_idx, snv_idx)
        if nz_idx.shape[0] == 0:
            # No SNVs present; return baseline directly.
            decoder_inp = torch.cat([z, batch_embedding], dim=1)
            mu = torch.exp(torch.clamp(self.decoder(decoder_inp), max=20.0))
            theta = F.softplus(self.raw_theta) + 1e-5
            if return_delta:
                delta_latent = torch.zeros_like(z)
                return mu, theta, delta_latent
            return mu, theta

        row_idx = nz_idx[:, 0]  # cell index for each (cell, SNV) pair
        col_idx = nz_idx[:, 1]  # SNV index for each (cell, SNV) pair

        # Gather the needed vectors
        z_gather = z[row_idx]                      # [num_cell_snv_pairs, latent_dim]
        e_gather = self.snv_embedding(col_idx)     # [num_cell_snv_pairs, snv_emb_dim]
        signs = torch.sign(G[row_idx, col_idx]).unsqueeze(1).float()  # [num_cell_snv_pairs, 1] in {-1,1}

        # conditional embedding
        cond_inp = torch.cat([z_gather, e_gather], dim=1)   # [num_cell_snv_pairs, latent_dim+snv_emb_dim]
        e_cond = self.cond_mlp(cond_inp)                    # [num_cell_snv_pairs, snv_emb_dim]

        # attention logits
        attn_inp = torch.cat([z_gather, e_cond], dim=1)     # [num_cell_snv_pairs, latent_dim+snv_emb_dim]
        logits = self.attn_mlp(attn_inp).squeeze(-1)        # [num_cell_snv_pairs]

        # per-cell softmax
        unique_cells, counts = torch.unique_consecutive(row_idx, return_counts=True)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device),
             counts.cumsum(0)]
        )

        emb_dim = self.snv_emb_dim
        weighted_emb = torch.zeros(batch_size, emb_dim, device=device)

        for cell_offset, cell_index in enumerate(unique_cells):
            start = offsets[cell_offset].item()
            end = offsets[cell_offset + 1].item()

            # All (cell, SNV) pairs for the current cell
            logits_seg = logits[start:end]          # [num_snvs_in_cell]
            e_seg = e_cond[start:end]               # [num_snvs_in_cell, snv_emb_dim]
            sign_seg = signs[start:end]             # [num_snvs_in_cell, 1]

            # Numerically stable softmax: exp(l - max)
            max_logit = logits_seg.max()
            logits_shift = logits_seg - max_logit
            exp_seg = torch.exp(logits_shift).unsqueeze(1)  # [num_snvs_in_cell, 1]
            softmax_denominator = exp_seg.sum() + 1e-12
            attention_weights = exp_seg / softmax_denominator  # [num_snvs_in_cell, 1]

            # Attention weighting with sign
            weighted_contribution = (e_seg * sign_seg) * attention_weights  # [num_snvs_in_cell, snv_emb_dim]

            weighted_emb[cell_index] = weighted_contribution.sum(dim=0)  # [snv_emb_dim]

        # project to latent space and decode
        delta_latent = self.snv_proj(weighted_emb)  # [batch_size, latent_dim]
        z_pert = z + delta_latent                   # [batch_size, latent_dim]

        decoder_inp = torch.cat([z_pert, batch_embedding], dim=1)
        mu = torch.exp(torch.clamp(self.decoder(decoder_inp), max=20.0))        # [batch_size, genes]
        theta = F.softplus(self.raw_theta) + 1e-5   # [genes]
        if return_delta:
            return mu, theta, delta_latent
        return mu, theta

    @torch.no_grad()
    def decode_sparse(self, z, G, batch_indices, pair_chunk: int = DEFAULT_PAIR_CHUNK):
        device = z.device
        batch_size, n_snvs = G.shape
        batch_embedding = self.batch_emb(
            batch_indices.to(device=device, dtype=torch.long)
        )
        nonzero_pairs = (G != 0).nonzero(as_tuple=False)  # [num_pairs, 2]
        if nonzero_pairs.numel() == 0:
            decoder_inp = torch.cat([z, batch_embedding], dim=1)
            mu = torch.exp(torch.clamp(self.decoder(decoder_inp), max=20.0))
            theta = F.softplus(self.raw_theta) + 1e-5
            return mu, theta

        snv_embedding_matrix = self.snv_embedding.weight.to(device)  # [n_snvs, emb]
        emb_dim = snv_embedding_matrix.shape[1]

        num_pairs = nonzero_pairs.shape[0]

        # Running statistics per cell to avoid materializing tensors of size [P, ...].
        cell_max_logit = torch.full((batch_size,), float("-inf"), device=device)
        cell_sum_exp = torch.zeros(batch_size, device=device)
        cell_weighted_sum = torch.zeros(batch_size, emb_dim, device=device)

        for start_pair, end_pair in chunk_iter(num_pairs, pair_chunk):
            pair_slice = nonzero_pairs[start_pair:end_pair]
            cell_idx_slice = pair_slice[:, 0]
            snv_idx_slice = pair_slice[:, 1]
            z_slice = z[cell_idx_slice]  # [pair_slice.shape[0], latent]
            snv_emb_slice = snv_embedding_matrix[snv_idx_slice]  # [pair_slice.shape[0], emb]

            cond_inp = torch.cat([z_slice, snv_emb_slice], dim=1)
            cond_emb = self.cond_mlp(cond_inp)  # [pair_slice.shape[0], emb]

            attn_inp = torch.cat([z_slice, cond_emb], dim=1)
            logits_slice = self.attn_mlp(attn_inp).squeeze(-1)  # [pair_slice.shape[0]]
            signs = torch.sign(G[cell_idx_slice, snv_idx_slice]).float().unsqueeze(1)  # [pair_slice.shape[0], 1]
            signed_cond_emb = cond_emb * signs  # [pair_slice.shape[0], emb]

            if logits_slice.numel() == 0:
                continue

            unique_cells_local, counts_per_cell = torch.unique_consecutive(
                cell_idx_slice, return_counts=True
            )
            offsets = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    counts_per_cell.cumsum(0),
                ]
            )

            for uidx in range(unique_cells_local.shape[0]):
                seg_start = offsets[uidx].item()
                seg_end = offsets[uidx + 1].item()

                cell_id = unique_cells_local[uidx]
                cell_seg_logits = logits_slice[seg_start:seg_end]
                cell_seg_signed_emb = signed_cond_emb[seg_start:seg_end]

                seg_max_logit = cell_seg_logits.max()
                prev_max_logit = cell_max_logit[cell_id]
                updated_max_logit = torch.maximum(prev_max_logit, seg_max_logit)

                prev_scale = torch.exp(prev_max_logit - updated_max_logit)
                exp_segment = torch.exp(cell_seg_logits - updated_max_logit).unsqueeze(1)

                cell_weighted_sum[cell_id] = (
                    cell_weighted_sum[cell_id] * prev_scale
                    + (exp_segment * cell_seg_signed_emb).sum(dim=0)
                )
                cell_sum_exp[cell_id] = (
                    cell_sum_exp[cell_id] * prev_scale + exp_segment.sum()
                )
                cell_max_logit[cell_id] = updated_max_logit

        weighted_embedding = torch.zeros(batch_size, emb_dim, device=device)
        has_contributions = cell_sum_exp > 0
        if has_contributions.any():
            weighted_embedding[has_contributions] = (
                cell_weighted_sum[has_contributions] / cell_sum_exp[has_contributions].unsqueeze(1)
            )

        delta_latent = self.snv_proj(weighted_embedding)  # [batch_size, latent]
        z_pert = z + delta_latent
        decoder_inp = torch.cat([z_pert, batch_embedding], dim=1)
        mu = torch.exp(torch.clamp(self.decoder(decoder_inp), max=20.0))  # [batch_size, genes]
        theta = F.softplus(self.raw_theta) + 1e-5
        return mu, theta

    # forward / loss
    def forward(
        self,
        expression_tensor: torch.Tensor,
        snv_tensor: torch.Tensor,
        batch_indices: torch.Tensor,
        use_mean_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_mean, latent_log_var = self.encode(expression_tensor)
        latent_sample = (
            latent_mean
            if use_mean_latent
            else self.reparameterize(latent_mean, latent_log_var)
        )
        reconstruction_mean, dispersion, delta_latent = self._decode_train_sparse(
            latent_sample, snv_tensor, batch_indices, return_delta=True
        )

        loss = self.compute_loss(
            expression_tensor,
            reconstruction_mean,
            dispersion,
            latent_mean,
            latent_log_var,
        )
        delta_norm = torch.norm(delta_latent, p=2, dim=1)
        z_norm = torch.norm(latent_sample, p=2, dim=1)
        delta_ratio = delta_norm / (z_norm + 1e-8)
        monitor_metrics = torch.stack(
            [delta_norm.mean(), z_norm.mean(), delta_ratio.mean()], dim=0
        )
        return loss.unsqueeze(0), reconstruction_mean, latent_mean, monitor_metrics

    def compute_loss(
        self,
        expression_tensor: torch.Tensor,
        reconstruction_mean: torch.Tensor,
        dispersion: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_log_var: torch.Tensor,
    ) -> torch.Tensor:
        #VAE loss
        eps = 1e-5

        # clamps for stability
        counts = torch.clamp(expression_tensor, min=0.0, max=1e6)
        mean_counts = torch.clamp(reconstruction_mean, min=eps, max=1e6)
        theta = torch.clamp(
            dispersion.unsqueeze(0), min=eps, max=1e6
        )  # [1, G] for broadcast

        # NB per cell, sum over genes
        log_gamma_term = (
            torch.lgamma(theta + counts)
            - torch.lgamma(theta)
            - torch.lgamma(counts + 1.0)
        )
        log_dispersion_ratio = torch.log(theta) - torch.log(
            theta + mean_counts
        )
        log_mean_ratio = torch.log(mean_counts) - torch.log(theta + mean_counts)
        log_nb = (
            log_gamma_term
            + theta * log_dispersion_ratio
            + counts * log_mean_ratio
        ).sum(dim=1)  # [B]
        nb_nll = -log_nb.mean()

        # KL
        free_bits = 0.5
        kl_per_dim = 0.5 * (
            latent_mean.pow(2) + latent_log_var.exp() - latent_log_var - 1.0
        )
        kl = torch.clamp(kl_per_dim, min=free_bits).sum(dim=1).mean()
        return nb_nll + kl

    def loss_fn(
        self,
        expression_tensor: torch.Tensor,
        snv_tensor: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        latent_mean, latent_log_var = self.encode(expression_tensor)
        latent_sample = self.reparameterize(latent_mean, latent_log_var)
        reconstruction_mean, dispersion = self._decode_train_sparse(
            latent_sample, snv_tensor, batch_indices
        )
        return self.compute_loss(
            expression_tensor,
            reconstruction_mean,
            dispersion,
            latent_mean,
            latent_log_var,
        )


# Training
def _make_derangement(indices: torch.Tensor) -> Optional[torch.Tensor]:
    num_items = int(indices.numel())
    if num_items < 2:
        return None
    if num_items == 2:
        return indices.flip(0)

    for _ in range(16):
        shuffled = indices[torch.randperm(num_items, device=indices.device)]
        if not torch.any(shuffled == indices):
            return shuffled

    shift = int(torch.randint(1, num_items, (1,), device=indices.device).item())
    return torch.roll(indices, shifts=shift, dims=0)


def _build_same_batch_derangement(batch_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(batch_indices.shape[0])
    permutation = torch.arange(batch_size, device=batch_indices.device)
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=batch_indices.device)

    for batch_value in torch.unique(batch_indices):
        group_indices = torch.nonzero(batch_indices == batch_value, as_tuple=False).squeeze(1)
        if group_indices.numel() < 2:
            continue
        group_perm = _make_derangement(group_indices)
        if group_perm is None:
            continue
        permutation[group_indices] = group_perm
        valid_mask[group_indices] = True

    return permutation, valid_mask


def train_snv_perturbation_model(
    model: SNVPerturbationModel,
    dataloader,
    num_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    clip_grad: float = 1.0,
    trainable_params: Optional[Iterable[nn.Parameter]] = None,
    world_size: int = 1,
    rank: int = 0,
    sampler: Optional[torch.utils.data.Sampler] = None,
    use_distributed: bool = False,
    rank_lambda_max: float = RANK_LAMBDA_MAX,
    rank_margin_max: float = RANK_MARGIN_MAX,
):
    if device is None:
        device = get_device()

    enable_rank_loss = RANK_LOSS_ENABLED
    rank_warmup_epochs = RANK_WARMUP_EPOCHS
    rank_min_diff_snv = RANK_MIN_DIFF_SNV
    rank_start_epoch = RANK_START_EPOCH

    model = model.to(device)
    params = trainable_params if trainable_params is not None else model.parameters()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    # )
    warmup_epochs = 10
    sched = warmup_cosine_lr(
        opt,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        base_lr=lr,
        eta_min=1e-8,
    )

    is_rank0 = (not use_distributed) or rank == 0

    for epoch in range(1, num_epochs + 1):
        if use_distributed and sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_rank_valid = 0.0
        total_delta_norm = 0.0
        total_z_norm = 0.0
        total_delta_ratio = 0.0
        n_train_samples = 0
        for expression_batch, snv_batch, batch_indices in dataloader:
            expression_batch = expression_batch.to(device).float()
            snv_batch = snv_batch.to(device).float()
            batch_indices = batch_indices.to(device).long()
            loss, _reconstruction_mean, _latent_mean, monitor_metrics = model(
                expression_batch, snv_batch, batch_indices
            )
            rank_loss_scaled = torch.zeros((), device=device)
            rank_valid_count = torch.zeros((), device=device)

            if enable_rank_loss and epoch >= rank_start_epoch:
                rank_epoch = epoch - rank_start_epoch + 1
                rank_progress = min(1.0, rank_epoch / max(1, rank_warmup_epochs))
                rank_lambda = rank_lambda_max * rank_progress
                rank_margin = rank_margin_max * rank_progress

                permutation, valid_rank_mask = _build_same_batch_derangement(batch_indices)
                if rank_min_diff_snv > 0 and valid_rank_mask.any():
                    diff_counts = (snv_batch != snv_batch[permutation]).sum(dim=1)
                    valid_rank_mask = valid_rank_mask & (diff_counts >= rank_min_diff_snv)

                if valid_rank_mask.any():
                    expression_rank = expression_batch[valid_rank_mask]
                    batch_rank = batch_indices[valid_rank_mask]
                    snv_pos_rank = snv_batch[valid_rank_mask]
                    snv_neg_rank = snv_batch[permutation[valid_rank_mask]]
                    rank_weight = torch.ones((), device=device)
                    rank_valid_count = valid_rank_mask.sum().float()
                else:
                    expression_rank = expression_batch[:1]
                    batch_rank = batch_indices[:1]
                    snv_pos_rank = snv_batch[:1]
                    snv_neg_rank = snv_batch[:1]
                    rank_weight = torch.zeros((), device=device)

                pos_loss_rank, _mu_pos_rank, _z_pos_rank, _metrics_pos_rank = model(
                    expression_rank,
                    snv_pos_rank,
                    batch_rank,
                    use_mean_latent=True,
                )
                neg_loss_rank, _mu_neg_rank, _z_neg_rank, _metrics_neg_rank = model(
                    expression_rank,
                    snv_neg_rank,
                    batch_rank,
                    use_mean_latent=True,
                )

                rank_term = F.relu(
                    rank_margin + pos_loss_rank.squeeze(0) - neg_loss_rank.squeeze(0)
                )
                rank_loss_scaled = rank_weight * (rank_lambda * rank_term)
                loss = loss + rank_loss_scaled.reshape(1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                clip_params = [
                    parameter
                    for param_group in opt.param_groups
                    for parameter in param_group["params"]
                    if parameter.grad is not None
                ]
                if clip_params:
                    nn.utils.clip_grad_norm_(clip_params, max_norm=clip_grad)
            opt.step()

            loss_sum = loss.detach() * expression_batch.size(0)
            if use_distributed:
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            total_loss += float(loss_sum.item())

            rank_loss_sum = rank_loss_scaled.detach() * expression_batch.size(0)
            if use_distributed:
                dist.all_reduce(rank_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(rank_valid_count, op=dist.ReduceOp.SUM)
            total_rank_loss += float(rank_loss_sum.item())
            total_rank_valid += float(rank_valid_count.item())

            metrics_sum = monitor_metrics.detach() * expression_batch.size(0)
            if use_distributed:
                dist.all_reduce(metrics_sum, op=dist.ReduceOp.SUM)
            total_delta_norm += float(metrics_sum[0].item())
            total_z_norm += float(metrics_sum[1].item())
            total_delta_ratio += float(metrics_sum[2].item())

            n_train_samples += expression_batch.size(0) * (
                world_size if use_distributed else 1
            )

        average_loss = total_loss / max(1, n_train_samples)
        average_rank_loss = total_rank_loss / max(1, n_train_samples)
        average_rank_valid_ratio = total_rank_valid / max(1, n_train_samples)
        average_delta_norm = total_delta_norm / max(1, n_train_samples)
        average_z_norm = total_z_norm / max(1, n_train_samples)
        average_delta_ratio = total_delta_ratio / max(1, n_train_samples)
        if is_rank0:
            log(
                "Epoch %03d | Loss: %.4f | Rank: %.6f | RankValid: %.4f | ||ΔZ||: %.6f | ||Z||: %.6f | ΔZ/Z: %.6f",
                epoch,
                average_loss,
                average_rank_loss,
                average_rank_valid_ratio,
                average_delta_norm,
                average_z_norm,
                average_delta_ratio,
            )
        sched.step()

    return model

def main_run(
    adata_rna: AnnData,
    adata_snv: AnnData,
    ann_df: Optional[pd.DataFrame] = None,
    batch_size_train: int = 256,
    num_epochs: int = 500,
    latent_dim: int = 128,
    snv_emb_dim: int = 64,
    batch_emb_dim: int = 16,
    top_k_attention: int = 2000,
    attn_batch: int = 256,
    score_attn_batch: Optional[int] = None,
    score_cell_batch: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    backbone_ckpt: Optional[str] = None,
    freeze_encoder: bool = True,
    gene_list_path: Optional[str] = None,
    marker_genes_top10_path: Optional[str] = None,
    model_ckpt: Optional[str] = None,
    eval_only: bool = False,
    result_folder: str = "./snv_result",
    celltype_key: str = "cell_cluster",
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    cell_type_free: bool = False,
    rna_batch_key: str = "batch",
    snv_batch_key: str = "sample",
    cluster_resolution: float = RESOLUTION,
    rank_lambda_max: float = RANK_LAMBDA_MAX,
    rank_margin_max: float = RANK_MARGIN_MAX,
):
    os.makedirs(result_folder, exist_ok=True)
    if backbone_ckpt is None:
        candidate_backbone = os.path.join(result_folder, "rna_backbone_pretrained.pt")
        if os.path.exists(candidate_backbone):
            backbone_ckpt = candidate_backbone
    use_distributed, rank, world_size, device = _setup_distributed_training(
        device=device
    )
    if score_attn_batch is None:
        score_attn_batch = attn_batch
    if score_cell_batch is None:
        score_cell_batch = attn_batch
    set_seed(seed)
    is_rank0 = (not use_distributed) or rank == 0
    if is_rank0:
        log("=" * 50)
        log("[INFO] Start SNV perturbation workflow")
        log(
            "[INFO] Pretrain backbone: "
            + (backbone_ckpt if backbone_ckpt is not None else "None")
        )
        log(f"[INFO] Device: {device}")
        log(f"[INFO] Top SNV number: {top_k_attention}")
        log(f"[INFO] Number epochs: {num_epochs}")
        log("=" * 50)

    if model_ckpt is None:
        model_ckpt = os.path.join(result_folder, "snv_perturbation_model.pt")

    eval_only = resolve_eval_only_checkpoint(eval_only, model_ckpt)

    ann_df = _prepare_annotation_dataframe(ann_df, adata_snv.var_names)

    if marker_genes_top10_path is None:
        marker_genes_top10_path = os.path.join(
            result_folder, "cell_cluster_marker_genes_top10.csv"
        )

    # Optionally align RNA data columns using pretrain gene order
    if gene_list_path is None and backbone_ckpt is not None:
        default_gene_list = backbone_ckpt + ".genes.npy"
        if os.path.exists(default_gene_list):
            gene_list_path = default_gene_list
        else:
            log(f"[WARNING] Expected gene list {default_gene_list} was not found; proceeding without alignment.")
    if gene_list_path is not None and os.path.exists(gene_list_path):
        pretrain_genes = np.load(gene_list_path, allow_pickle=True).tolist()
        adata_rna = align_adata_rna_with_genes(adata_rna, pretrain_genes)
    elif gene_list_path is not None:
        log(f"[WARNING] Gene list {gene_list_path} was provided but not found; proceeding without alignment.")

    if (
        rna_batch_key in adata_rna.obs
        and not pd.api.types.is_categorical_dtype(adata_rna.obs[rna_batch_key])
    ):
        adata_rna.obs[rna_batch_key] = adata_rna.obs[rna_batch_key].astype("category")
    if snv_batch_key not in adata_snv.obs:
        log(
            "[WARN] SNV batch key '%s' not found in SNV AnnData.obs.",
            snv_batch_key,
        )


    # Prepare tensors
    (
        X_tensor,
        G_tensor,
        B_tensor,
        gene_names,
        snv_names,
        adata_rna,
        adata_snv,
        n_batches,
    ) = tensors_from_anndata(
        adata_rna, adata_snv, batch_key=rna_batch_key, dense=False
    )
    if is_rank0:
        _save_snv_name_array(
            model_ckpt + ".snvs.npy",
            snv_names,
            "initial model SNV list",
        )

    # Build DataLoader
    if isinstance(X_tensor, torch.Tensor) and isinstance(G_tensor, torch.Tensor):
        dataset = torch.utils.data.TensorDataset(X_tensor, G_tensor, B_tensor)
        collate_fn = None
    else:
        dataset = LazyAnndataDataset(B_tensor.shape[0])
        collate_fn = make_lazy_collate_fn(X_tensor, G_tensor, B_tensor)
    sampler = None
    shuffle = True
    if use_distributed and world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_train,
        shuffle=shuffle,
        drop_last=False,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    # Build model
    model = SNVPerturbationModel(
        n_genes=len(gene_names),
        n_snvs=len(snv_names),
        latent_dim=latent_dim,
        snv_emb_dim=snv_emb_dim,
        n_batches=n_batches,
        batch_emb_dim=batch_emb_dim,
    )

    # Optionally load a pre-trained backbone and freeze
    trainables = None
    if backbone_ckpt is not None:
        model = safe_load_backbone_into_snv_model(model, backbone_ckpt, device=device)
        if freeze_encoder:
            trainables = freeze_encoder_only(model)

    model = model.to(device)
    if is_rank0:
        log_model_parameter_summary(model)

    if use_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
        if is_rank0:
            log(
                f"[INFO] Using DistributedDataParallel with world size {world_size} (rank {rank})"
            )

    if eval_only:
        state = torch.load(model_ckpt, map_location=device)
        state = _strip_distributed_prefix(state)
        target_model = model.module if isinstance(model, DistributedDataParallel) else model
        target_model.load_state_dict(state)
        target_model.eval()
        if is_rank0:
            log(f"[INFO] Loaded model checkpoint from {model_ckpt} for evaluation.")
    else:
        model = train_snv_perturbation_model(
            model,
            dataloader,
            num_epochs=num_epochs,
            lr=1e-3,
            device=device,
            weight_decay=1e-5,
            clip_grad=5.0,
            trainable_params=trainables,
            world_size=world_size,
            rank=rank,
            sampler=sampler,
            use_distributed=use_distributed,
            rank_lambda_max=rank_lambda_max,
            rank_margin_max=rank_margin_max,
        )

        if dist.is_initialized():
            dist.barrier()

        if is_rank0:
            # checkpoint
            ckpt_dir = os.path.dirname(model_ckpt)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            state = (
                model.module.state_dict()
                if isinstance(model, DistributedDataParallel)
                else model.state_dict()
            )
            state = _strip_distributed_prefix(state)
            torch.save(state, model_ckpt)
            log(f"[INFO] Saved trained model checkpoint to {model_ckpt}")

    if dist.is_initialized():
        dist.barrier()
    if isinstance(model, DistributedDataParallel):
        model = model.module
        model.to(device)
        if is_rank0:
            log("[INFO] Detached model from DistributedDataParallel")

    if dist.is_initialized():
        dist.barrier()

    log("=" * 50)

    top_snv_names = []
    if is_rank0:
        log("[INFO] Compute mean attention per SNV")
        top_attn_df = rank_snvs_by_attention(
            model,
            X_tensor,
            G_tensor,
            adata_snv,
            top_k=top_k_attention,
            batch_size=attn_batch,
            pair_chunk=pair_chunk,
            device=device,
        )
        top_attn_path = os.path.join(result_folder, "top_snv_attention.csv")
        top_attn_df.to_csv(top_attn_path, index=False)
        log(f"[INFO] Saved top_snv_attention to {top_attn_path}")
        top_snv_names = top_attn_df["SNV"].tolist()

    # all ranks
    if dist.is_available() and dist.is_initialized():
        obj = [top_snv_names]
        dist.broadcast_object_list(obj, src=0)
        top_snv_names = obj[0] or []

    top_snv_indices = []
    if not top_snv_names:
        log("[WARN] No SNVs passed the attention prefilter; skipping per-cell SNV scoring.")
    else:
        top_snv_indices = [adata_snv.var_names.get_loc(name) for name in top_snv_names]
        log("[INFO] Scoring per-cell perturbation for top-attention SNVs.")
        per_cell_snv_scores = batch_score_snvs_by_cell(
            model,
            X_tensor,
            G_tensor,
            B_tensor,
            snv_indices=top_snv_indices,
            snv_names=top_snv_names,
            cell_ids=adata_snv.obs_names.tolist(),
            device=device,
            cell_batch=attn_batch,
            pair_chunk=pair_chunk,
        )
        per_cell_scores_path = os.path.join(
            result_folder, "snv_perturbation_scores_by_cell.csv"
        )
        if is_rank0:
            per_cell_snv_scores.to_csv(per_cell_scores_path, index=False)
        log(f"[INFO] Saved per-cell SNV perturbation scores to {per_cell_scores_path}")

    if cell_type_free:
        log("[INFO] Cell-type-free mode enabled: scoring SNVs using all carrier cells.")

        if not top_snv_names:
            log("[WARN] No SNVs passed the attention prefilter; skipping scoring.")
            if is_rank0:
                _save_snv_name_array(
                    model_ckpt + ".final_snvs.npy",
                    [],
                    "final scored SNV list",
                )
            _cleanup_distributed()
            return
        df_scores = batch_score_all_snvs(
            model,
            X_tensor,
            G_tensor,
            B_tensor,
            snv_indices=top_snv_indices,
            snv_names=top_snv_names,
            gene_names=gene_names,
            ann_df=ann_df,
            device=device,
            cell_batch=attn_batch,
            pair_chunk=pair_chunk,
        )

        scores_path = os.path.join(result_folder, "snv_perturbation_scores.csv")
        if is_rank0:
            df_scores.to_csv(scores_path, index=False)
            _save_snv_name_array(
                model_ckpt + ".final_snvs.npy",
                top_snv_names,
                "final scored SNV list",
            )
        if dist.is_initialized():
            dist.barrier()
        log(f"[INFO] Saved cell-type-free SNV perturbation scores to {scores_path}")
        _cleanup_distributed()
        return

    # scoring
    log("[INFO] Score selected SNVs counterfactually (sparse)")

    if is_rank0:
        log(
            "[INFO] Encoding RNA profiles with the model encoder to obtain batch-corrected latents."
        )
        adata_rna.obsm["X_latent"] = (
            _encode_latent_representation(
                model=model,
                rna_matrix=X_tensor,
                device=device,
                batch_size=attn_batch,
            )
            .cpu()
            .numpy()
        )

    # Clustering
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        log("[INFO] Clustering cells and identifying marker genes.")
        cluster_cells(
            adata_rna,
            marker_genes_path=marker_genes_top10_path,
            result_folder=result_folder,
            celltype_key=celltype_key,
            resolution=cluster_resolution,
        )


    # Broadcast the per-cell cluster labels to all ranks to keep score_by_celltype in sync
    if dist.is_available() and dist.is_initialized():
        # Ensure same cell order across ranks
        if rank == 0:
            labels = adata_rna.obs[celltype_key].astype(str).tolist()
        else:
            labels = None

        obj = [labels]
        dist.broadcast_object_list(obj, src=0)
        labels = obj[0]
        adata_rna.obs[celltype_key] = pd.Categorical(labels)
        dist.barrier()
    else:
        pass

    by_ct = score_by_celltype(
        model,
        adata_rna,
        adata_snv,
        ann_df,
        celltype_key=celltype_key,
        top_k_attention=top_k_attention,  # top_k_attention / cluster
        attn_batch=score_attn_batch,
        device=device,
        cell_batch=score_cell_batch,
        pair_chunk=pair_chunk,
        batch_key=rna_batch_key,
    )
    if is_rank0:
        all_attn = pd.concat(
            [celltype_result[0] for celltype_result in by_ct.values()], axis=0
        )
        all_scores = pd.concat(
            [celltype_result[1] for celltype_result in by_ct.values()], axis=0
        )
        attn_by_ct_path = os.path.join(result_folder, "top_snv_attention_by_celltype.csv")
        scores_by_ct_path = os.path.join(
            result_folder, "snv_perturbation_scores_by_celltype.csv"
        )
        all_scores, cooccurrence_audit = filter_highly_cooccurring_snvs_by_celltype(
            all_scores=all_scores,
            adata_snv=adata_snv,
            adata_rna=adata_rna,
            celltype_key=celltype_key,
            sample_key=snv_batch_key,
        )
        cooccurrence_audit_path = os.path.join(
            result_folder, "snv_cooccurrence_dedup_removed.csv"
        )
        cooccurrence_audit.to_csv(cooccurrence_audit_path, index=False)
        log(
            "[INFO] Saved SNV co-occurrence de-duplication audit to %s",
            cooccurrence_audit_path,
        )
        all_scores = _normalise_score_column_minmax(
            all_scores,
            SCORE_COLUMN,
            output_column=f"{SCORE_COLUMN}_norm",
        )
        all_scores = _normalise_score_column_minmax(
            all_scores,
            COSINE_DISTANCE_SCORE_COLUMN,
            output_column=f"{COSINE_DISTANCE_SCORE_COLUMN}_norm",
        )
        all_attn.to_csv(attn_by_ct_path, index=False)
        all_scores.to_csv(scores_by_ct_path, index=False)
        final_snv_names = list(dict.fromkeys(all_scores["SNV"].astype(str).tolist()))
        _save_snv_name_array(
            model_ckpt + ".final_snvs.npy",
            final_snv_names,
            "final scored SNV list",
        )

        log("[INFO] Saved snv_perturbation_scores.")

        log("[INFO] Aggregating per-cell perturbation coefficients.")
        score_column = SCORE_COLUMN
        log("[INFO] Using SNV score column: %s", score_column)
        adata_snv_aligned, adata_rna_aligned, celltypes = _align_celltypes_for_scores(
            adata_snv, adata_rna, celltype_key
        )
        grouped_scores = _prepare_celltype_score_lookup(all_scores, score_column)
        secondary_score_column = (
            COSINE_DISTANCE_SCORE_COLUMN
            if score_column == "score_euclidean"
            else "score_euclidean"
        )
        grouped_scores_secondary = None
        if secondary_score_column in all_scores.columns:
            grouped_scores_secondary = _prepare_celltype_score_lookup(
                all_scores, secondary_score_column
            )
        else:
            log(
                "[WARN] Secondary SNV score column '%s' is missing; only primary perturbation scores will be exported.",
                secondary_score_column,
            )
        cell_perturbation_df = compute_cell_perturbation_scores(
            adata_snv=adata_snv_aligned,
            celltypes=celltypes,
            grouped_scores=grouped_scores,
            score_column=score_column,
            result_folder=result_folder,
            grouped_scores_secondary=grouped_scores_secondary,
            secondary_score_column=secondary_score_column,
        )
        if is_rank0:
            plot_cell_perturbation_umap(
                adata_rna=adata_rna_aligned,
                perturbation_df=cell_perturbation_df,
                result_folder=result_folder,
                celltype_key=celltype_key,
                adata_snv=adata_snv_aligned,
                rasterized=True,
                output_filename="cell_perturbation_and_celltype_umap.pdf",
            )
    _cleanup_distributed()


def _cleanup_distributed() -> None:
    """Synchronize and tear down distributed process groups if active."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Command-line entry point for executing the SNV perturbation workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the SNV perturbation model using aligned RNA and barcode-by-SNV AnnData inputs. "
            "Exports SNV attention rankings, perturbation scores, and downstream plots for functional effect analysis."
        )
    )
    parser.add_argument(
        "-y",
        "--yaml",
        "--config",
        dest="yaml_path",
        help="Path to YAML configuration file containing all script options.",
    )
    parser.add_argument(
        "yaml_path_pos",
        nargs="?",
        help="Positional alternative for the YAML configuration file path.",
    )
    args = parser.parse_args(argv)

    yaml_path = args.yaml_path or args.yaml_path_pos
    if yaml_path is None:
        parser.error("A YAML configuration path must be provided.")

    with open(yaml_path, "r", encoding="utf-8") as yaml_handle:
        yaml_full = yaml.safe_load(yaml_handle) or {}

    yaml_config = yaml_full.get("snv_eff", yaml_full)

    result_folder = yaml_config.get(
        "result_folder", yaml_full.get("result_folder", "./snv_result")
    )
    os.makedirs(result_folder, exist_ok=True)

    adata_snv_path = yaml_config.get("adata_snv")
    ann_csv_path = yaml_config.get("ann_csv")
    if isinstance(ann_csv_path, str):
        ann_csv_path = ann_csv_path.strip() or None
    if adata_snv_path is None:
        parser.error("adata_snv path must be provided in YAML.")

    adata_rna_path = os.path.join(result_folder, "finetune_aligned.h5ad")
    min_cnt = yaml_config.get("min_cnt", 100)
    n_top = yaml_config.get("n_top", 50)
    batch_size_train = yaml_config.get("batch_size_train", 256)
    num_epochs = yaml_config.get("num_epochs", 500)
    latent_dim = yaml_config.get("latent_dim", 128)
    snv_emb_dim = yaml_config.get("snv_emb_dim", 64)
    top_k_attention = yaml_config.get("top_k_attention", 2000)
    attn_batch = yaml_config.get("attn_batch", 256)
    score_attn_batch = yaml_config.get("score_attn_batch", attn_batch)
    score_cell_batch = yaml_config.get("score_cell_batch", attn_batch)
    seed = yaml_config.get("seed", 42)
    rank_lambda_max = yaml_config.get("rank_lambda_max", RANK_LAMBDA_MAX)
    rank_margin_max = yaml_config.get("rank_margin_max", RANK_MARGIN_MAX)
    celltype_key = yaml_config.get("celltype_key", "cell_cluster")
    cluster_resolution = yaml_config.get("cluster_resolution", RESOLUTION)
    try:
        rank_lambda_max = float(rank_lambda_max)
        rank_margin_max = float(rank_margin_max)
    except (TypeError, ValueError):
        parser.error("rank_lambda_max and rank_margin_max must be numeric values greater than or equal to 0.")
    if rank_lambda_max < 0.0 or rank_margin_max < 0.0:
        parser.error("rank_lambda_max and rank_margin_max must be greater than or equal to 0.")
    try:
        cluster_resolution = float(cluster_resolution)
    except (TypeError, ValueError):
        parser.error("cluster_resolution must be a numeric value greater than 0.")
    if cluster_resolution <= 0.0:
        parser.error("cluster_resolution must be greater than 0.")
    cell_type_free = bool(yaml_config.get("cell_type_free", False))
    freeze_encoder = bool(
        yaml_config.get("freeze_encoder", yaml_config.get("freeze_backbone", False))
    )
    eval_only = bool(yaml_config.get("eval_only", False))
    device_name = yaml_config.get("device")
    pair_chunk = yaml_config.get("pair_chunk", DEFAULT_PAIR_CHUNK)
    try:
        score_attn_batch = int(score_attn_batch)
        score_cell_batch = int(score_cell_batch)
    except (TypeError, ValueError):
        parser.error("score_attn_batch and score_cell_batch must be integer values greater than 0.")
    if score_attn_batch <= 0 or score_cell_batch <= 0:
        parser.error("score_attn_batch and score_cell_batch must be greater than 0.")
    pre_train_cfg = yaml_full.get("pre_train", {})
    pre_train_training_cfg = (
        pre_train_cfg.get("training", {}) if isinstance(pre_train_cfg, dict) else {}
    )
    rna_batch_key = (
        pre_train_training_cfg.get("batch_key", "batch")
        if isinstance(pre_train_training_cfg, dict)
        else "batch"
    )
    batch_emb_dim = (
        yaml_config.get("batch_emb_dim")
        if "batch_emb_dim" in yaml_config
        else pre_train_training_cfg.get("batch_emb_dim", 16)
    )
    try:
        batch_emb_dim = int(batch_emb_dim)
    except (TypeError, ValueError):
        parser.error("batch_emb_dim must be an integer value greater than 0.")
    if batch_emb_dim <= 0:
        parser.error("batch_emb_dim must be greater than 0.")
    snv_batch_key = yaml_config.get("batch_key", "sample")
    if isinstance(rna_batch_key, str):
        rna_batch_key = rna_batch_key.strip() or "batch"
    if isinstance(snv_batch_key, str):
        snv_batch_key = snv_batch_key.strip() or "sample"
    device = torch.device(device_name) if device_name else None

    _use_distributed, _rank, _world_size, device = _setup_distributed_training(device=device)

    is_distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if is_distributed else 0

    if is_distributed:
        if rank == 0:
            adata_snv = read_h5ad(adata_snv_path)
            adata_snv = filter_snv_by_counts(
                adata_snv, min_cnt=min_cnt, n_top=n_top, result_folder=result_folder
            )
            adata_snv = _filter_snv_to_standard_chromosomes(adata_snv)
            kept_snv_names = adata_snv.var_names.tolist()
        else:
            kept_snv_names = None

        obj = [kept_snv_names]
        dist.broadcast_object_list(obj, src=0)
        kept_snv_names = obj[0] or []

        if rank != 0:
            adata_snv = read_h5ad(adata_snv_path)
            adata_snv = adata_snv[:, kept_snv_names].copy()
    else:
        adata_snv = read_h5ad(adata_snv_path)
        adata_snv = filter_snv_by_counts(
            adata_snv, min_cnt=min_cnt, n_top=n_top, result_folder=result_folder
        )
        adata_snv = _filter_snv_to_standard_chromosomes(adata_snv)
    adata_rna = read_h5ad(adata_rna_path)
    if ann_csv_path is None:
        ann_df = None
        if rank == 0:
            log(
                "[INFO] ann_csv is not provided; annotation metadata fields will be left empty."
            )
    else:
        ann_df = _read_annotation_table(ann_csv_path)

    main_run(
        adata_rna,
        adata_snv,
        ann_df,
        batch_size_train=batch_size_train,
        num_epochs=num_epochs,
        latent_dim=latent_dim,
        snv_emb_dim=snv_emb_dim,
        batch_emb_dim=batch_emb_dim,
        top_k_attention=top_k_attention,
        attn_batch=attn_batch,
        score_attn_batch=score_attn_batch,
        score_cell_batch=score_cell_batch,
        device=device,
        seed=seed,
        freeze_encoder=freeze_encoder,
        eval_only=eval_only,
        result_folder=result_folder,
        celltype_key=celltype_key,
        cluster_resolution=cluster_resolution,
        cell_type_free=cell_type_free,
        pair_chunk=pair_chunk,
        rna_batch_key=rna_batch_key,
        snv_batch_key=snv_batch_key,
        rank_lambda_max=rank_lambda_max,
        rank_margin_max=rank_margin_max,
    )
    log("[INFO] Finished all workflow!")


if __name__ == "__main__":
    main()
