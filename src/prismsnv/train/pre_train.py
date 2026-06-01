import argparse
import os
from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

try:
    from .utility import align_and_preprocess_adata, log, remove_high_mt, set_seed, tensor_from_anndata_X, warmup_cosine_lr
except ImportError:
    from utility import align_and_preprocess_adata, log, remove_high_mt, set_seed, tensor_from_anndata_X, warmup_cosine_lr  # type: ignore

BETA_MAX = 1.0


def _coerce_str_to_number(name, value, cast):
    if isinstance(value, str):
        try:
            return cast(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be {cast.__name__}-compatible, got {value!r}") from exc
    return value


def _normalize_mu_activation(mu_activation: str) -> str:
    """Normalize and validate ``mu_activation``."""

    activation = str(mu_activation).strip().lower()
    if activation not in {"umi", "softplus", "identity"}:
        raise ValueError(
            "mu_activation must be one of {'umi', 'softplus', 'identity'}, "
            f"got {mu_activation!r}."
        )
    return activation


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grl(x, lambda_adv):
    return GradientReversal.apply(x, lambda_adv)


class BatchDiscriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_batches):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_batches)
        )

    def forward(self, z):
        return self.net(z)

# RNA-only VAE backbone module for pre-training on single-cell RNA-seq data.
class RNAOnlyBackbone(nn.Module):
    """
        - Encoder: q(z|x) maps gene expression to latent space
        - Decoder: p(x|z,b) optionally conditioned on batch embeddings
        - Negative-binomial reconstruction loss for UMI counts
        - KL divergence with free-bits for latent regularization
        - Optional adversarial discriminator on latent z to remove batch effects
    """
    def __init__(
            self,
            n_genes: int,
            latent_dim: int = 128,
            mu_activation: str = "umi",
            n_batches: int = None,
            batch_emb_dim: int = 16,  # Dimensionality of the optional batch embedding
            lambda_adv: float = 0.0,
            disc_hidden_dim: int = None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.mu_activation = _normalize_mu_activation(mu_activation)


        # adversarial settings
        self.lambda_adv = float(lambda_adv)

        # Encoder q(z|x) aligned with the SNV model architecture
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

        # Batch embedding
        self.n_batches = n_batches
        self.batch_emb_dim = batch_emb_dim if n_batches is not None else 0
        if n_batches is not None:
            self.batch_emb = nn.Embedding(n_batches, self.batch_emb_dim)
            log("[INFO] Batch embedding enabled: n_batches =", n_batches, ", emb_dim =", self.batch_emb_dim)
        else:
            self.batch_emb = None
            log("[WARNING] No batch covariate provided; batch embedding disabled.")

        # Decoder
        dec_in = latent_dim + self.batch_emb_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 256),
            nn.ReLU(),
            nn.Linear(256, n_genes),
        )

        # NB
        self.raw_theta = nn.Parameter(torch.zeros(n_genes))

        if (self.lambda_adv > 0.0) and (self.n_batches is not None) and (self.n_batches > 1):
            if disc_hidden_dim is None:
                disc_hidden_dim = latent_dim
            self.discriminator = BatchDiscriminator(
                z_dim=latent_dim,
                hidden_dim=disc_hidden_dim,
                n_batches=self.n_batches,
            )
            log(f"[ADV] Adversarial batch discriminator enabled (lambda_adv={self.lambda_adv}, "
                f"hidden_dim={disc_hidden_dim}, n_batches={self.n_batches})")
        else:
            self.discriminator = None
            if self.lambda_adv > 0.0:
                log("[ADV] lambda_adv > 0 but no valid n_batches; adversarial branch disabled.")

    def encode(self, x: torch.Tensor):
        hidden_representation = self.encoder(x)
        mu = self.encoder_mu(hidden_representation)
        log_var = self.encoder_log_var(hidden_representation)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        stddev = torch.exp(0.5 * log_var)
        noise = torch.randn_like(stddev)
        return mu + noise * stddev

    def decode_logits_to_mu(self, decoder_output: torch.Tensor):
        if self.mu_activation == "umi":
            decoder_output_clamped = torch.clamp(decoder_output, max=20)
            return torch.exp(decoder_output_clamped)
        elif self.mu_activation == "softplus":
            return F.softplus(decoder_output)
        else:
            return decoder_output

    def decode(self, latent: torch.Tensor, batch_indices: torch.Tensor = None):
        """Decode latent representations into gene expression means and dispersion parameters.

        Parameters
        ----------
        latent : torch.Tensor
            Latent representation with shape ``[batch_size, latent_dim]``.
        batch_indices : torch.Tensor, optional
            Batch indices with shape ``[batch_size]``. When provided and batch embeddings
            are enabled, the corresponding embeddings are concatenated to ``latent`` before
            passing through the decoder.
        """
        if (self.batch_emb is not None) and (batch_indices is not None):
            batch_embedding = self.batch_emb(batch_indices)  # [B, batch_emb_dim]
            decoder_input = torch.cat([latent, batch_embedding], dim=1)  # [B, latent_dim + batch_emb_dim]
        else:
            decoder_input = latent

        decoder_output = self.decoder(decoder_input)  # [B, G] (logits)
        mu = self.decode_logits_to_mu(decoder_output)
        dispersion = F.softplus(self.raw_theta) + 1e-5  # Dispersion parameter with numerical stabilisation
        return mu, dispersion

    @staticmethod
    def kl_gauss(mu: torch.Tensor, log_var: torch.Tensor):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    @staticmethod
    def nb_nll(counts: torch.Tensor, mean: torch.Tensor, dispersion: torch.Tensor, epsilon: float = 1e-6):
        counts = torch.clamp(counts, min=0.)
        mean = torch.clamp(mean, min=epsilon)
        dispersion = torch.clamp(dispersion, min=epsilon)

        term_lgamma = torch.lgamma(dispersion + counts) - torch.lgamma(dispersion) - torch.lgamma(counts + 1.0)
        term_dispersion = dispersion * (torch.log(dispersion + epsilon) - torch.log(dispersion + mean + epsilon))
        term_counts = counts * (torch.log(mean + epsilon) - torch.log(dispersion + mean + epsilon))
        log_prob_nb = (term_lgamma + term_dispersion + term_counts).sum(dim=1)  # [B]
        return -log_prob_nb.mean()

    def loss_fn(self, x: torch.Tensor, use_nb: bool = True, beta: float = 1.0, batch_ids: torch.Tensor = None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        mu_out, theta = self.decode(z, batch_indices=batch_ids)
        if use_nb:
            recon = self.nb_nll(x, mu_out, theta)
        else:
            recon = F.mse_loss(mu_out, x, reduction="mean")
        free_bits = 0.5
        kl_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)
        kl = torch.clamp(kl_per_dim, min=free_bits).sum(dim=1).mean()
        total_loss = recon + beta * kl
        logs = {"recon": float(recon.item()), "kl": float(kl.item())}

        if (self.lambda_adv > 0.0) and (self.discriminator is not None) and (batch_ids is not None):
            # GRL
            z_rev = grl(z, self.lambda_adv)
            logits = self.discriminator(z_rev)         # [batch_size, n_batches]
            adv_loss = F.cross_entropy(logits, batch_ids)
            total_loss = total_loss + adv_loss
            logs["adv"] = float(adv_loss.item())
        else:
            logs["adv"] = 0.0

        return total_loss, logs


def pretrain_rna_backbone(
        adata_path: str,
        latent_dim: int = 128,
        mu_activation: str = "umi",  # "umi" for UMI counts; "identity"/"softplus" for corrected/normalized data
        use_nb: bool = False,  # True for raw counts; False for corrected/log data
        batch_size: int = 512,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        kl_warmup_epochs: int = 20,
        beta_max: float = BETA_MAX,
        grad_clip: float = 5.0,
        device: str = "cuda",
        hvg_only: bool = True,
        out_ckpt: str = "rna_backbone_pretrained.pt",
        batch_key: str = "batch",
        batch_emb_dim: int = 16,
        lambda_adv: float = 0.0,
        disc_hidden_dim: int = None,
        lambda_adv_warmup_epochs: int = 10,
):
    set_seed(42)
    latent_dim = _coerce_str_to_number("latent_dim", latent_dim, int)
    batch_size = _coerce_str_to_number("batch_size", batch_size, int)
    epochs = _coerce_str_to_number("epochs", epochs, int)
    if epochs < 10:
        log(
            f"[WARNING] epochs={epochs} is smaller than 10; overriding epochs to 10 "
            "to keep the LR warmup/cosine schedule stable."
        )
        epochs = 10
    lr = _coerce_str_to_number("lr", lr, float)
    weight_decay = _coerce_str_to_number("weight_decay", weight_decay, float)
    kl_warmup_epochs = _coerce_str_to_number("kl_warmup_epochs", kl_warmup_epochs, int)
    beta_max = _coerce_str_to_number("beta_max", beta_max, float)
    grad_clip = _coerce_str_to_number("grad_clip", grad_clip, float)
    batch_emb_dim = _coerce_str_to_number("batch_emb_dim", batch_emb_dim, int)
    lambda_adv = _coerce_str_to_number("lambda_adv", lambda_adv, float)
    lambda_adv_warmup_epochs = _coerce_str_to_number(
        "lambda_adv_warmup_epochs", lambda_adv_warmup_epochs, int
    )
    mu_activation = _normalize_mu_activation(mu_activation)
    if disc_hidden_dim is not None:
        disc_hidden_dim = _coerce_str_to_number("disc_hidden_dim", disc_hidden_dim, int)

    adata = sc.read_h5ad(adata_path)
    if hvg_only and ("highly_variable" in adata.var.columns):
        adata = adata[:, adata.var["highly_variable"]].copy()

    rna_tensor = tensor_from_anndata_X(adata)
    batch_ids_tensor = None
    n_batches = None
    if (batch_key is not None) and (batch_key in adata.obs.columns):
        cats = adata.obs[batch_key].astype("category")
        adata.obs[batch_key] = cats
        n_batches = len(cats.cat.categories)
        if cats.isna().any():
            missing_count = int(cats.isna().sum())
            raise ValueError(
                f"obs['{batch_key}'] contains {missing_count} missing batch labels. "
                "Please fill or remove missing values before pretraining."
            )
        batch_ids = cats.cat.codes.to_numpy().astype("int64")  # [N]
        batch_ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
        log(f"[Batch] Using obs['{batch_key}'] with {n_batches} levels.")
    else:
        log("[Batch] No batch covariate found/used.")

    # configuration
    use_batch_embedding = batch_ids_tensor is not None
    log("======== Pretraining Configuration ========")
    log(f"[Config] The mu activation: {mu_activation}")
    log(f"[Config] Total epochs: {epochs}")
    log(f"[Config] Use batch_embedding: {use_batch_embedding}")
    log(f"[Config] The batch_key: {batch_key}")
    log(f"[Config] beta_max: {beta_max}")
    exists = (batch_key is not None) and (batch_key in adata.obs.columns)
    log(f"[Config] Is batch_key_exists: {exists}")
    if use_batch_embedding:
        log(f"[Config] The batches count: {n_batches}, emb_dim: {batch_emb_dim}")
    adv_hidden = disc_hidden_dim if disc_hidden_dim is not None else latent_dim
    adv_enabled = (lambda_adv > 0.0) and (n_batches is not None) and (n_batches > 1)
    log(f"[Config] lambda_adv: {lambda_adv}")
    log(f"[Config] Discriminator hidden_dim: {adv_hidden}")
    log(f"[Config] GRL + discriminator enabled: {adv_enabled}")
    if lambda_adv > 0.0 and not adv_enabled:
        log("[Config] Adversarial branch requested but disabled due to insufficient batch information.")
    log("===========================================")

    n_genes = rna_tensor.shape[1]
    gene_list = adata.var_names.tolist()

    model = RNAOnlyBackbone(
        n_genes=n_genes,
        latent_dim=latent_dim,
        mu_activation=mu_activation,
        n_batches=n_batches,
        batch_emb_dim=batch_emb_dim,
        lambda_adv=lambda_adv,
        disc_hidden_dim=disc_hidden_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)
    scheduler = warmup_cosine_lr(opt, warmup_epochs=10, total_epochs=epochs, base_lr=lr)
    if batch_ids_tensor is not None:
        ds = TensorDataset(rna_tensor, batch_ids_tensor)
    else:
        ds = TensorDataset(rna_tensor)
    n_cells = len(ds)
    train_size = int(0.9 * n_cells)
    val_size = n_cells - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size],
                                                     generator=torch.Generator().manual_seed(42))

    def make_loader(subset, shuffle):
        if batch_ids_tensor is not None:
            rna_sub = subset.dataset.tensors[0][subset.indices]
            batch_sub = subset.dataset.tensors[1][subset.indices]
            d = TensorDataset(rna_sub, batch_sub)
        else:
            rna_sub = subset.dataset.tensors[0][subset.indices]
            d = TensorDataset(rna_sub)
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_dl = make_loader(train_ds, shuffle=True)
    val_dl = make_loader(val_ds, shuffle=False)

    best = None
    best_epoch = None
    patience = 50
    bad_epochs = 0
    checkpoint_saved = False

    for epoch in range(1, epochs + 1):
        beta = beta_max * min(1.0, epoch / max(1, kl_warmup_epochs))
        if lambda_adv > 0.0:
            adv_scale = min(1.0, epoch / max(1, lambda_adv_warmup_epochs))
            current_lambda_adv = lambda_adv * adv_scale
        else:
            current_lambda_adv = 0.0
        if hasattr(model, "lambda_adv"):
            model.lambda_adv = current_lambda_adv

        total = 0.0
        n_train_samples = 0
        train_logs_total = {"recon": 0.0, "kl": 0.0, "adv": 0.0}
        model.train()
        for batch in train_dl:
            if batch_ids_tensor is not None:
                input_batch, batch_ids_batch = batch
                input_batch = input_batch.to(device, non_blocking=True)
                batch_ids_batch = batch_ids_batch.to(device, non_blocking=True)
                loss, logs = model.loss_fn(input_batch, use_nb=use_nb, beta=beta, batch_ids=batch_ids_batch)
            else:
                (input_batch,) = batch
                input_batch = input_batch.to(device, non_blocking=True)
                loss, logs = model.loss_fn(input_batch, use_nb=use_nb, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item() * input_batch.size(0)
            n_train_samples += input_batch.size(0)
            for k in train_logs_total:
                train_logs_total[k] += float(logs.get(k, 0.0)) * input_batch.size(0)

        avg = total / max(1, n_train_samples)

        model.eval()
        val_total = 0.0
        n_val_samples = 0
        val_logs_total = {"recon": 0.0, "kl": 0.0, "adv": 0.0}
        with torch.no_grad():
            for batch in val_dl:
                if batch_ids_tensor is not None:
                    input_batch, batch_ids_batch = batch
                    input_batch = input_batch.to(device, non_blocking=True)
                    batch_ids_batch = batch_ids_batch.to(device, non_blocking=True)
                    vloss, vlogs = model.loss_fn(input_batch, use_nb=use_nb, beta=beta, batch_ids=batch_ids_batch)
                else:
                    (input_batch,) = batch
                    input_batch = input_batch.to(device, non_blocking=True)
                    vloss, vlogs = model.loss_fn(input_batch, use_nb=use_nb, beta=beta)

                val_total += vloss.item() * input_batch.size(0)
                n_val_samples += input_batch.size(0)
                for k in val_logs_total:
                    val_logs_total[k] += float(vlogs.get(k, 0.0)) * input_batch.size(0)

        val_avg = val_total / max(1, n_val_samples)
        train_recon = train_logs_total["recon"] / max(1, n_train_samples)
        train_kl = train_logs_total["kl"] / max(1, n_train_samples)
        train_adv = train_logs_total["adv"] / max(1, n_train_samples)
        val_recon = val_logs_total["recon"] / max(1, n_val_samples)
        val_kl = val_logs_total["kl"] / max(1, n_val_samples)
        val_adv = val_logs_total["adv"] / max(1, n_val_samples)
        scheduler.step()
        if epoch % 5 == 0 or epoch <= 10:
            log(
                f"Epoch {epoch:03d} | "
                f"Train {avg:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}, adv {train_adv:.4f}) | "
                f"Val {val_avg:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f}, adv {val_adv:.4f}) | "
                f"beta {beta:.2f} | lambda_adv {current_lambda_adv:.4f}"
            )

        # Simple patience-based early stopping (patience starts after warmups reach max)
        patience_active = (beta >= beta_max)
        if lambda_adv > 0.0:
            patience_active = patience_active and (current_lambda_adv >= lambda_adv)

        if patience_active:
            if (best is None) or (val_avg + 1e-5 < best):
                best = val_avg
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), out_ckpt)
                checkpoint_saved = True
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log(f"[INFO] Early stop at epoch {epoch}. Best Loss {best:.4f}")
                    break

    if not checkpoint_saved:
        torch.save(model.state_dict(), out_ckpt)
        log("[INFO] Patience never activated; saved final model instead of early-stop best.")
    else:
        log(f"[INFO] Saved best backbone to: {out_ckpt} (epoch {best_epoch})")
    gene_list_path = out_ckpt + ".genes.npy"
    np.save(gene_list_path, np.array(gene_list))
    log(f"[INFO] Saved gene list to: {gene_list_path}")
    return out_ckpt, gene_list


def _infer_n_batches_from_state_dict(state_dict):
    weight = state_dict.get("batch_emb.weight")
    if isinstance(weight, torch.Tensor):
        return weight.shape[0]
    return None


@torch.no_grad()
def _encode_finetune_latents(
    adata_path: str,
    checkpoint_path: str,
    latent_dim: int,
    mu_activation: str,
    batch_emb_dim: int,
    lambda_adv: float,
    disc_hidden_dim: Optional[int],
    device: str,
    batch_size: int = 512,
):
    log("[INFO] Encoding finetune AnnData with trained encoder to populate 'X_latent'.")
    adata = sc.read_h5ad(adata_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    n_batches = _infer_n_batches_from_state_dict(state_dict)

    model = RNAOnlyBackbone(
        n_genes=adata.n_vars,
        latent_dim=latent_dim,
        mu_activation=mu_activation,
        n_batches=n_batches,
        batch_emb_dim=batch_emb_dim,
        lambda_adv=lambda_adv,
        disc_hidden_dim=disc_hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rna_tensor = tensor_from_anndata_X(adata)
    dataloader = DataLoader(rna_tensor, batch_size=batch_size, shuffle=False, drop_last=False)
    latents = []
    for input_batch in dataloader:
        input_batch = input_batch.to(device, non_blocking=True)
        mu, _ = model.encode(input_batch)
        latents.append(mu.cpu())

    if latents:
        adata.obsm["X_latent"] = torch.cat(latents, dim=0).numpy()
    else:
        adata.obsm["X_latent"] = np.empty((0, latent_dim), dtype=np.float32)

    adata.write_h5ad(adata_path)
    log(f"[INFO] Wrote finetune AnnData with 'X_latent' to: {adata_path}")


def safe_load_backbone_into_snv_model(
    snv_model: nn.Module,
    backbone_ckpt: str,
    device: Optional[Union[str, torch.device]] = "cuda",
):
    """Load only shape-matching tensors from pretrained backbone into current SNV model.
    - Skips batch_emb.* and any layer whose shape doesn't match (e.g., decoder first layer).
    """
    map_location = device if device is not None else "cpu"
    ckpt_sd = torch.load(backbone_ckpt, map_location=map_location)
    tgt_sd = snv_model.state_dict()
    copied = 0
    skipped = []

    for k, v in ckpt_sd.items():
        # Skip explicit batch embedding parameters; shape filtering below would also exclude them
        if k.startswith("batch_emb."):
            skipped.append((k, "batch_emb"))
            continue
        # Copy parameters only when the name exists in the SNV model and the tensor shapes match
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            tgt_sd[k] = v.clone()
            copied += 1
        else:
            skipped.append((k, "shape_mismatch_or_not_found"))

    snv_model.load_state_dict(tgt_sd, strict=False)
    log(f"[Transfer] Copied {copied} tensors; skipped {len(skipped)} (batch_emb/shape-mismatch).")
    for name, reason in skipped:
        log(f"[Transfer] Skipped {name} ({reason}).")
    return snv_model


def load_backbone_into_snv_model(snv_model: nn.Module, backbone_ckpt: str, device="cuda"):
    """Transfer encoder/decoder/raw_theta weights from the pretrained backbone into the SNV model.

    Because the submodule names and shapes match, the weights can be loaded with
    ``strict=False`` while only copying the tensors that have an exact name match.
    """
    # Build a temporary backbone with identical architecture to load pretrained weights
    backbone_model = RNAOnlyBackbone(
        n_genes=snv_model.n_genes,
        latent_dim=snv_model.latent_dim,
        mu_activation="identity",
    ).to(device)
    backbone_model.load_state_dict(torch.load(backbone_ckpt, map_location=device), strict=True)

    # Collect the SNV model state_dict and replace the parameters that match the backbone
    target_state_dict = snv_model.state_dict()

    # Keep track of parameters that were aligned based on an exact name match
    keys_to_replace = []
    for key in target_state_dict.keys():
        if key.startswith("encoder.") or key in (
            "encoder_mu.weight",
            "encoder_mu.bias",
            "encoder_log_var.weight",
            "encoder_log_var.bias",
        ) or key.startswith("decoder.") or key == "raw_theta":
            keys_to_replace.append(key)

    backbone_state_dict = backbone_model.state_dict()
    for key in keys_to_replace:
        if key in backbone_state_dict and target_state_dict[key].shape == backbone_state_dict[key].shape:
            target_state_dict[key] = backbone_state_dict[key].clone()

    snv_model.load_state_dict(target_state_dict, strict=False)
    log(f"[Transfer] Copied backbone weights into SNV model ({len(keys_to_replace)} keys).")
    return snv_model


def freeze_backbone_train_snv_only(snv_model: nn.Module):
    # Freeze encoder/decoder/raw_theta to preserve pretrained representations
    for p in snv_model.encoder.parameters(): p.requires_grad = False
    snv_model.encoder_mu.weight.requires_grad = False
    snv_model.encoder_mu.bias.requires_grad = False
    snv_model.encoder_log_var.weight.requires_grad = False
    snv_model.encoder_log_var.bias.requires_grad = False

    for p in snv_model.decoder.parameters(): p.requires_grad = False
    if hasattr(snv_model, "raw_theta"):
        snv_model.raw_theta.requires_grad = False

    # Only allow SNV-specific submodules to be updated during finetuning
    trainables = []
    for name, p in snv_model.named_parameters():
        if p.requires_grad and any(k in name for k in ["snv_embedding", "cond_mlp", "attn_mlp", "snv_proj"]):
            trainables.append(p)

    log(f"[Finetune] Trainable SNV params: {sum(p.numel() for p in trainables):,}")
    return trainables


def freeze_encoder_only(model: nn.Module):
    """Freeze encoder side only; keep decoder & SNV heads trainable."""
    for name, p in model.named_parameters():
        if (name.startswith("encoder") or
                name.startswith("encoder_mu") or
                name.startswith("encoder_log_var")):
            p.requires_grad = False
        else:
            p.requires_grad = True
    return (p for p in model.parameters() if p.requires_grad)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Align pretraining and finetuning RNA AnnData inputs, then train the RNA-only backbone model. "
            "Writes aligned .h5ad files and a pretrained checkpoint for the downstream SNV effect workflow."
        )
    )
    parser.add_argument(
        "-y",
        "--yaml",
        "--config",
        dest="yaml_path",
        help="Path to YAML configuration file containing script options.",
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

    with open(yaml_path, "r", encoding="utf-8") as fh:
        yaml_full = yaml.safe_load(fh) or {}
    yaml_config = yaml_full.get("pre_train", yaml_full)

    align_cfg = yaml_config.get("align", {}) if isinstance(yaml_config, dict) else {}
    train_cfg = yaml_config.get("training", {}) if isinstance(yaml_config, dict) else {}

    def pick(key, *sources, default=None):
        for src in sources:
            if not src:
                continue
            if isinstance(src, dict):
                if key in src and src[key] is not None:
                    return src[key]
        return default

    result_folder = yaml_config.get("result_folder", yaml_full.get("result_folder", "./snv_result"))
    os.makedirs(result_folder, exist_ok=True)

    pretrain_adata = pick("pretrain_adata", align_cfg, yaml_config)
    finetune_adata = pick("finetune_adata", align_cfg, yaml_config)
    if pretrain_adata is None or finetune_adata is None:
        parser.error("pretrain_adata and finetune_adata must be provided in the YAML config.")

    pretrain_out = os.path.join(result_folder, "pretrain_aligned.h5ad")
    finetune_out = os.path.join(result_folder, "finetune_aligned.h5ad")

    num_epochs = pick("num_epochs", train_cfg, yaml_config)
    if num_epochs is None:
        num_epochs = pick("epochs", train_cfg, yaml_config)
    if num_epochs is None:
        parser.error("num_epochs must be specified in the YAML config.")

    batch_key = pick("batch_key", train_cfg, yaml_config, default="batch")
    batch_emb_dim = pick("batch_emb_dim", train_cfg, yaml_config, default=16)

    expected_doublet_rate = pick(
        "expected_doublet_rate", align_cfg, yaml_config, default=0.05
    )
    remove_scrublet = pick("remove_scrublet", align_cfg, yaml_config, default=False)
    mt_percent = pick("mt_percent", align_cfg, yaml_config, default=15)
    try:
        mt_percent = float(mt_percent)
    except (TypeError, ValueError):
        parser.error("mt_percent must be a numeric value in [0, 100].")
    if not (0.0 <= mt_percent <= 100.0):
        parser.error("mt_percent must be in [0, 100].")

    aligned_pre, aligned_ft = align_and_preprocess_adata(
        pretrain_adata,
        finetune_adata,
        pretrain_out,
        finetune_out,
        expected_doublet_rate=expected_doublet_rate,
        remove_scrublet=remove_scrublet,
        mt_percent=mt_percent,
    )

    training_defaults = {
        "latent_dim": 128,
        "mu_activation": "umi",
        "use_nb": True,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "kl_warmup_epochs": 20,
        "beta_max": BETA_MAX,
        "grad_clip": 5.0,
        "device": "cuda",
        "hvg_only": True,
        "out_ckpt": os.path.join(result_folder, "rna_backbone_pretrained.pt"),
        "lambda_adv": 0.0,
        "disc_hidden_dim": None,
        "lambda_adv_warmup_epochs": 10,
    }

    training_kwargs = {}
    for key, default in training_defaults.items():
        training_kwargs[key] = pick(key, train_cfg, yaml_config, default=default)

    training_kwargs.update({
        "adata_path": aligned_pre,
        "epochs": num_epochs,
        "batch_key": batch_key,
        "batch_emb_dim": batch_emb_dim,
    })

    checkpoint_path, _ = pretrain_rna_backbone(**training_kwargs)

    try:
        _encode_finetune_latents(
            adata_path=aligned_ft,
            checkpoint_path=checkpoint_path,
            latent_dim=training_kwargs["latent_dim"],
            mu_activation=training_kwargs["mu_activation"],
            batch_emb_dim=batch_emb_dim,
            lambda_adv=training_kwargs.get("lambda_adv", 0.0),
            disc_hidden_dim=training_kwargs.get("disc_hidden_dim"),
            device=training_kwargs.get("device", "cuda"),
            batch_size=training_kwargs.get("batch_size", 512),
        )
    except Exception as exc:  # pragma: no cover - best-effort enrichment
        log(f"[WARNING] Failed to precompute 'X_latent' for finetune AnnData: {exc}")


if __name__ == '__main__':
    main()
