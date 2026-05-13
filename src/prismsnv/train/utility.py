import gc
import math
import os
import random
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from pandas.api import types as pdt
from scipy import sparse
from anndata import AnnData
from tqdm import tqdm

# Default pair chunk size.
DEFAULT_PAIR_CHUNK = 20_000
# Default Leiden resolution.
RESOLUTION = 0.4
# Defaults for SNV co-occurrence de-duplication before final celltype score export.
COOCCURRENCE_DEDUP_ENABLED = True
COOCCURRENCE_MIN_CARRIER_CELLS = 20
COOCCURRENCE_MIN_SHARED_CELLS = 20
COOCCURRENCE_JACCARD_THRESHOLD = 0.85
COOCCURRENCE_OVERLAP_THRESHOLD = 0.95

def _should_log_to_console() -> bool:
    """Return ``True`` when the current process should emit log output."""

    if dist.is_available():
        if dist.is_initialized():
            return dist.get_rank() == 0
        env_rank = os.environ.get("RANK")
        try:
            return env_rank is None or int(env_rank) == 0
        except (TypeError, ValueError):
            return True

    return True


def _colorize_tag(message: str) -> str:
    """Apply ANSI color to tags like ``[INFO]`` when present."""

    match = re.match(r"^(\[(.+?)\])", message)
    if not match:
        return message

    tag_full, tag_name = match.groups()
    color_map = {
        "Config": "36",
        "INFO": "32",
        "WARNING": "33",
        "ERROR": "31",
    }
    code = color_map.get(tag_name, "35")
    colored_tag = f"\033[{code}m{tag_full}\033[0m"
    return colored_tag + message[match.end():]


def log(message: object = "", *args, sep: str = " ", end: str = "\n") -> None:
    """Print a log message to the console with a timestamp.

    Supports printf-style string formatting when ``args`` are provided, falling
    back to simple string joining if formatting fails. Messages starting with a
    tag like ``[INFO]`` will be colorized for readability.
    """

    if not _should_log_to_console():
        return

    if args:
        if isinstance(message, str):
            try:
                message = message % args
            except (TypeError, ValueError):
                message = sep.join([message, *(str(arg) for arg in args)])
        else:
            message = sep.join([str(message), *(str(arg) for arg in args)])

    if isinstance(message, str):
        message = _colorize_tag(message)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", end=end)


def resolve_eval_only_checkpoint(eval_only: bool, model_ckpt: str) -> bool:
    """Fall back to training when eval-only checkpoint is missing."""

    if eval_only and not os.path.exists(model_ckpt):
        log(
            "[WARNING] Evaluation-only mode requested but checkpoint not found: %s. "
            "Starting training instead.",
            model_ckpt,
        )
        return False

    return eval_only


def log_model_parameter_summary(model: nn.Module) -> None:
    """Log a human-readable summary of model parameters."""

    log(
        "[INFO] Model instantiated with",
        f"n_genes={getattr(model, 'n_genes', 'unknown')}",
        f"n_snvs={getattr(model, 'n_snvs', 'unknown')}",
        f"latent_dim={getattr(model, 'latent_dim', 'unknown')}",
        f"snv_emb_dim={getattr(model, 'snv_emb_dim', 'unknown')}",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log(f"{'Parameter name':<40} {'Shape':<20} {'Count':<12}")
    log("=" * 60)
    log(f"Total parameters:      {total_params:,}")
    log(f"Trainable parameters:  {trainable_params:,}")
    log("=" * 60)

    def count_params(module_or_tensor) -> int:
        if isinstance(module_or_tensor, torch.Tensor):
            return module_or_tensor.numel()
        return sum(p.numel() for p in module_or_tensor.parameters() if p.requires_grad)

    modules_to_check = {
        "Encoder (MLP)": getattr(model, "encoder", None),
        "Encoder (Mu Head)": getattr(model, "encoder_mu", None),
        "Encoder (LogVar Head)": getattr(model, "encoder_log_var", None),
        "Decoder (MLP)": getattr(model, "decoder", None),
        "Batch Embedding": getattr(model, "batch_emb", None),
        "Dispersion (Theta)": getattr(model, "raw_theta", None),
        "SNV Embedding": getattr(model, "snv_embedding", None),
        "Attention MLP": getattr(model, "attn_mlp", None),
        "Conditional MLP": getattr(model, "cond_mlp", None),
        "SNV Projection": getattr(model, "snv_proj", None),
    }

    summary_total = 0
    log("--- Parameter summary by module ---")
    for name, module in modules_to_check.items():
        if module is None:
            continue
        count = count_params(module)
        summary_total += count
        log(f"{name:<25}: {count:12,d} parameters")

    log("=" * 60)
    log(f"{'Summary total':<25}: {summary_total:12,d} parameters")

    if summary_total != trainable_params:
        log(
            f"Warning: summary total ({summary_total:,}) does not match trainable total ({trainable_params:,})."
        )
        log(
            "This may indicate parameters missing from the summary or being counted multiple times."
        )
    else:
        log("Summary total matches trainable parameters.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def warmup_cosine_lr(optimizer, warmup_epochs, total_epochs, base_lr, eta_min=1e-8):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return eta_min / base_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (1 - eta_min / base_lr)

    return LambdaLR(optimizer, lr_lambda)


def _normalise_chromosome(value: object) -> str:
    """Return a lower-case chromosome string prefixed with ``chr`` for lookup."""

    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = re.sub(r"^chr", "", text, flags=re.IGNORECASE)
    return f"chr{text.lower()}"


def _read_annotation_table(path: str) -> pd.DataFrame:
    """Read optional SNV annotation metadata from CSV/TSV-like tables."""

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t")
        except pd.errors.ParserError:
            # Some annotation exports use variable whitespace instead of tabs.
            return pd.read_csv(path, sep=r"\s+", engine="python")

    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception as fallback_exc:
            raise pd.errors.ParserError(
                "Failed to parse ann_csv as comma-delimited or auto-detected "
                f"table: {path}. If this is an ANNOVAR multianno file, save it "
                "as a tab-delimited .txt/.tsv or set ann_csv to that file."
            ) from fallback_exc


def _prepare_annotation_dataframe(
    ann_df: pd.DataFrame, snv_names: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Normalise annotation tables for downstream SNV metadata lookup."""

    if ann_df is None:
        return ann_df

    ann_df = ann_df.copy()
    ann_df.columns = [str(col).strip() for col in ann_df.columns]

    lower_to_col = {col.lower(): col for col in ann_df.columns}

    chr_col = next(
        (lower_to_col[key] for key in ("chr", "#chr", "chrom", "#chrom") if key in lower_to_col),
        None,
    )
    pos_col = next(
        (lower_to_col[key] for key in ("pos", "position", "start") if key in lower_to_col),
        None,
    )
    ref_col = lower_to_col.get("ref")
    alt_col = lower_to_col.get("alt")

    if chr_col and pos_col and ref_col and alt_col:
        chr_series = ann_df[chr_col]
        chr_lookup = chr_series.map(_normalise_chromosome)
        valid_chr = chr_lookup != ""

        pos_numeric = pd.to_numeric(ann_df[pos_col], errors="coerce")
        pos_lookup = pos_numeric.round().astype("Int64")
        valid_pos = pos_lookup.notna()

        ref_lookup = (
            ann_df[ref_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .where(~ann_df[ref_col].isna(), "")
        )
        valid_ref = ref_lookup != ""

        alt_lookup = (
            ann_df[alt_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .where(~ann_df[alt_col].isna(), "")
        )
        valid_alt = alt_lookup != ""

        ann_df["_snv_lookup_chr"] = chr_lookup
        ann_df["_snv_lookup_pos"] = pos_lookup
        ann_df["_snv_lookup_ref"] = ref_lookup
        ann_df["_snv_lookup_alt"] = alt_lookup

        if "name" not in ann_df.columns:
            ann_df["name"] = ""
            mask = valid_chr & valid_pos & valid_ref & valid_alt
            if mask.any():
                ann_df.loc[mask, "name"] = (
                    ann_df.loc[mask, "_snv_lookup_chr"].astype(str)
                    + ":"
                    + ann_df.loc[mask, "_snv_lookup_pos"].astype("Int64").astype(str)
                    + "_"
                    + ann_df.loc[mask, "_snv_lookup_ref"].astype(str)
                    + ">"
                    + ann_df.loc[mask, "_snv_lookup_alt"].astype(str)
                )
    else:
        if "name" not in ann_df.columns:
            ann_df["name"] = ""

    if "name" in ann_df.columns:
        ann_df["name"] = ann_df["name"].astype(str)

    return ann_df


_SNV_NAME_REGEX = re.compile(r"^(chr[^:]+):([0-9]+)_([^>]+)>([^>]+)$", re.IGNORECASE)


def _parse_snv_identifier(snv_name: str) -> Optional[Tuple[str, int, str, str]]:
    """Extract (chromosome, position, ref, alt) tuple from an SNV identifier.

    Examples
    --------
    >>> _parse_snv_identifier("chr1:123456_A>T")
    ("chr1", 123456, "A", "T")
    >>> _parse_snv_identifier("CHR2:789012_C>G")
    ("chr2", 789012, "C", "G")
    >>> _parse_snv_identifier("invalid_format")
    None
    """

    if snv_name is None:
        return None

    # Normalize input text.
    snv_str = str(snv_name).strip()

    # Match SNV format.
    match = _SNV_NAME_REGEX.match(snv_str)
    if not match:
        return None

    # Parse matched fields.
    chrom = _normalise_chromosome(match.group(1))  # Normalized chr.
    pos = int(match.group(2))  # Genomic position.
    ref = match.group(3).upper()
    alt = match.group(4).upper()

    return chrom, pos, ref, alt


STANDARD_CHROMOSOMES = {
    *(f"chr{i}" for i in range(1, 23)),
    "chrx",
    "chry",
}


def _filter_snv_to_standard_chromosomes(adata_snv: AnnData) -> AnnData:
    """Keep SNVs on autosomes, chrX, or chrY."""

    chromosome_columns = ("chr", "#chr", "chrom", "#chrom", "chromosome")
    lower_to_column = {str(column).lower(): column for column in adata_snv.var.columns}
    chromosome_column = next(
        (
            lower_to_column[column_name]
            for column_name in chromosome_columns
            if column_name in lower_to_column
        ),
        None,
    )

    chromosome_values = (
        adata_snv.var[chromosome_column].to_numpy()
        if chromosome_column is not None
        else None
    )

    keep_mask = []
    unknown_count = 0
    for snv_position, snv_name in enumerate(adata_snv.var_names):
        snv_name_text = str(snv_name)
        parsed = _parse_snv_identifier(snv_name_text)
        chromosome = parsed[0] if parsed is not None else ""
        if not chromosome and chromosome_values is not None:
            chromosome = _normalise_chromosome(chromosome_values[snv_position])
        if not chromosome and ":" in snv_name_text:
            chromosome = _normalise_chromosome(snv_name_text.split(":", 1)[0])
        if not chromosome:
            unknown_count += 1
        keep_mask.append(chromosome in STANDARD_CHROMOSOMES)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    total_before = adata_snv.n_vars
    kept = int(keep_mask.sum())
    removed = total_before - kept
    if kept == 0:
        raise ValueError(
            "No SNVs remain after standard chromosome filtering. "
            "Expected chromosome labels chr1-chr22, chrX, or chrY."
        )

    log(
        "[INFO] Standard chromosome filter: kept %d/%d SNVs "
        "(removed %d; unrecognized chromosome labels: %d).",
        kept,
        total_before,
        removed,
        unknown_count,
    )
    return adata_snv[:, keep_mask].copy()


def _lookup_annotation_row(ann_df: pd.DataFrame, snv_name: str) -> Optional[pd.Series]:
    """Return the first annotation row matching ``snv_name`` if available."""

    if ann_df is None or ann_df.empty or snv_name is None:
        return None

    snv_str = str(snv_name)

    candidate_columns = [
        col
        for col in ann_df.columns
        if str(col).lower() in {"name", "snv", "variant", "id"}
    ]
    for col in candidate_columns:
        try:
            matches = ann_df[ann_df[col].astype(str) == snv_str]
        except Exception:
            continue
        if not matches.empty:
            return matches.iloc[0]

    parsed = _parse_snv_identifier(snv_str)
    if parsed and {"_snv_lookup_chr", "_snv_lookup_pos", "_snv_lookup_ref", "_snv_lookup_alt"}.issubset(
        ann_df.columns
    ):
        chrom, pos, ref, alt = parsed
        variant_mask = (
            (ann_df["_snv_lookup_chr"] == chrom)
            & (ann_df["_snv_lookup_pos"] == pos)
            & (ann_df["_snv_lookup_ref"] == ref)
            & (ann_df["_snv_lookup_alt"] == alt)
        )
        matches = ann_df[variant_mask]
        if not matches.empty:
            return matches.iloc[0]

    try:
        index_str = ann_df.index.astype(str)
    except Exception:
        index_str = pd.Index(ann_df.index.map(str))
    idx_matches = np.flatnonzero(index_str == snv_str)
    if idx_matches.size:
        return ann_df.iloc[idx_matches[0]]

    return None


_ANNOTATION_FIELD_MAP = {
    "Func": ["Func.refGeneWithVer", "Func.refGene"],
    "Gene": ["Gene.refGeneWithVer", "Gene.refGene"],
    "GeneDetail": ["GeneDetail.refGeneWithVer", "GeneDetail.refGene"],
    "ExonicFunc": ["ExonicFunc.refGeneWithVer", "ExonicFunc.refGene"],
    "CLNDN": ["CLNDN"],
    "CLNALLELEID": ["CLNALLELEID"],
    "CLNSIG": ["CLNSIG"],
}


def _extract_annotation_metadata(ann_df: pd.DataFrame, snv_name: str) -> Dict[str, str]:
    """Collect selected annotation fields for ``snv_name`` safely."""

    row = _lookup_annotation_row(ann_df, snv_name)

    def _value(columns) -> str:
        if row is None:
            return ""
        candidates = columns if isinstance(columns, (list, tuple)) else [columns]
        for column in candidates:
            if column not in row.index:
                continue
            val = row[column]
            if pd.isna(val):
                continue
            return str(val)
        return ""

    return {key: _value(cols) for key, cols in _ANNOTATION_FIELD_MAP.items()}


def _distributed_env_ready() -> bool:
    """Return ``True`` when environment variables for ``init_process_group`` exist."""

    required_env = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    return required_env.issubset(os.environ)


def filter_snv_by_counts(
    adata_snv, min_cnt=100, n_top=50, result_folder: str = "./snv_result"
):
    snv_matrix = adata_snv.X
    if sparse.isspmatrix(snv_matrix):
        n_pos = (snv_matrix == 1).sum(axis=0).A1
        n_neg = (snv_matrix == -1).sum(axis=0).A1
    else:
        n_pos = np.sum(snv_matrix == 1, axis=0)
        n_neg = np.sum(snv_matrix == -1, axis=0)
        if hasattr(n_pos, "A1"):
            n_pos = n_pos.A1
            n_neg = n_neg.A1
        else:
            n_pos = np.asarray(n_pos).ravel()
            n_neg = np.asarray(n_neg).ravel()

    adata_snv.var["n_pos_1"] = n_pos
    adata_snv.var["n_neg_-1"] = n_neg

    total_before = adata_snv.shape[1]
    total_pos_before = int(n_pos.sum())
    total_neg_before = int(n_neg.sum())

    log(
        f"SNV stats before filtering: {total_before} SNVs, "
        f"{total_pos_before} positives, {total_neg_before} negatives"
    )

    keep = (n_pos >= min_cnt) & (n_neg >= min_cnt)
    adata_snv = adata_snv[:, keep].copy()

    total_after = int(keep.sum())
    total_pos_after = int(n_pos[keep].sum())
    total_neg_after = int(n_neg[keep].sum())

    log(
        f"SNV stats after filtering: {total_after} SNVs, "
        f"{total_pos_after} positives, {total_neg_after} negatives"
    )
    log(f"Filtered out {total_before - total_after} SNVs")

    top_snvs = adata_snv.var[["n_pos_1", "n_neg_-1"]].copy()
    top_snvs["total_nonzero"] = top_snvs["n_pos_1"] + top_snvs["n_neg_-1"]
    top_snvs = top_snvs.sort_values("total_nonzero", ascending=False).head(n_top)

    if _should_log_to_console():
        plt.figure(figsize=(12, 6))
        idx = np.arange(len(top_snvs))
        plt.bar(idx, top_snvs["n_pos_1"], label="1")
        plt.bar(idx, top_snvs["n_neg_-1"], bottom=top_snvs["n_pos_1"], label="-1")
        plt.xticks(idx, top_snvs.index, rotation=90)
        plt.ylabel("Count")
        plt.title("Top SNVs by non-zero counts")
        plt.legend()
        plt.tight_layout()
        os.makedirs(result_folder, exist_ok=True)
        stacked_bar_path = os.path.join(result_folder, "snv_top_stacked_bar.png")
        plt.savefig(stacked_bar_path)
        plt.close()

    return adata_snv


def tensor_from_anndata_X(adata) -> torch.Tensor:
    expression_matrix = adata.X
    if hasattr(expression_matrix, "toarray"):  # Sparse to dense.
        expression_matrix = expression_matrix.toarray()
    return torch.tensor(expression_matrix, dtype=torch.float32)


def align_and_preprocess_adata(
    pretrain_path: str,
    finetune_path: str,
    out_pretrain: str = "pretrain_aligned.h5ad",
    out_finetune: str = "finetune_aligned.h5ad",
    n_top_genes: int = 10000,
    hvg_only: bool = True,
    expected_doublet_rate: float = 0.05,
    remove_scrublet: bool = False,
    mt_percent: float = 15,
):
    """Align genes between two RNA h5ad files and save the processed results."""
    log(f"[INFO] Loading data {pretrain_path} & {finetune_path}.")
    adata_pre = sc.read_h5ad(pretrain_path)
    adata_ft = sc.read_h5ad(finetune_path)
    # Basic QC filters.
    sc.pp.filter_cells(adata_pre, min_genes=100)
    sc.pp.filter_cells(adata_ft, min_genes=100)
    sc.pp.filter_genes(adata_pre, min_cells=100)
    sc.pp.filter_genes(adata_ft, min_cells=100)
    adata_pre = remove_high_total_counts_by_batch(adata_pre, batch_key="batch", percentile=99.5)
    adata_ft = remove_high_total_counts_by_batch(adata_ft, batch_key="batch", percentile=99.5)
    adata_pre = remove_high_mt(adata_pre, mt_percent)
    adata_ft = remove_high_mt(adata_ft, mt_percent)
    # Optional scrublet.
    if remove_scrublet:
        log(f"[INFO] Remove scrublet, expected_doublet_rate: {expected_doublet_rate}.")
        sc.external.pp.scrublet(adata_pre, expected_doublet_rate=expected_doublet_rate)
        sc.external.pp.scrublet(adata_ft, expected_doublet_rate=expected_doublet_rate)
        adata_pre = adata_pre[~adata_pre.obs["predicted_doublet"], :].copy()
        adata_ft = adata_ft[~adata_ft.obs["predicted_doublet"], :].copy()
    else:
        log("[INFO] Skip scrublet doublet removal (remove_scrublet=false).")

    def mark_hvg(adata):
        tmp = adata.copy()
        sc.pp.normalize_total(tmp)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes)
        adata.var["highly_variable"] = tmp.var["highly_variable"].values
        return adata

    adata_pre = mark_hvg(adata_pre)
    adata_ft = mark_hvg(adata_ft)

    if hvg_only:
        adata_pre = adata_pre[:, adata_pre.var["highly_variable"]].copy()
        adata_ft = adata_ft[:, adata_ft.var["highly_variable"]].copy()
        log(f"[INFO] HVG only mode, top gene number: {n_top_genes}.")

    target_genes = sorted(set(adata_pre.var_names).union(set(adata_ft.var_names)))

    def reindex_adata(adata, genes):
        genes = list(genes)
        idx_map = {
            gene: gene_index for gene_index, gene in enumerate(adata.var_names)
        }

        expression_matrix = adata.X
        if sparse.issparse(expression_matrix):
            expression_matrix = expression_matrix.tocsc().astype(np.float32)
            zero_col = sparse.csc_matrix((adata.n_obs, 1), dtype=np.float32)
        else:
            expression_matrix = np.asarray(expression_matrix, dtype=np.float32)
            zero_col = np.zeros((adata.n_obs, 1), dtype=np.float32)

        columns = []
        for gene in genes:
            col_idx = idx_map.get(gene)
            if col_idx is None:
                columns.append(zero_col)
            else:
                columns.append(expression_matrix[:, col_idx])

        if sparse.issparse(expression_matrix):
            aligned_X = sparse.hstack(columns, format="csr")
        else:
            aligned_X = np.hstack(
                [np.asarray(col) if col.ndim == 2 else col[:, None] for col in columns]
            ).astype(np.float32)

        var = adata.var.reindex(genes).copy()
        if "highly_variable" in var.columns:
            var["highly_variable"] = var["highly_variable"].fillna(False).astype(bool)
        if "mt" in var.columns:
            var["mt"] = var["mt"].fillna(False).astype(bool)

        # Make var dtypes AnnData-safe.
        for col in var.columns:
            series = var[col]
            if (
                pdt.is_object_dtype(series.dtype)
                or pdt.is_categorical_dtype(series.dtype)
                or pdt.is_string_dtype(series.dtype)
            ):
                non_na = series.dropna()
                # Avoid categorical ".all()" issues.
                is_all_bool = bool(
                    len(non_na)
                    and all(
                        isinstance(value, (bool, np.bool_))
                        for value in non_na.astype(object).to_numpy()
                    )
                )
                if is_all_bool:
                    var[col] = series.fillna(False).astype(bool)
                else:
                    # Cast to plain strings.
                    var[col] = (
                        series.astype(object)
                        .where(series.notna(), "")
                        .astype(str)
                    )
        aligned = sc.AnnData(aligned_X, obs=adata.obs.copy(), var=var)
        aligned.obs_names = adata.obs_names.copy()
        aligned.var_names = genes
        return aligned

    adata_pre = reindex_adata(adata_pre, target_genes)
    adata_ft = reindex_adata(adata_ft, target_genes)

    adata_pre.write(out_pretrain)
    adata_ft.write(out_finetune)

    log(f"[Align] Union genes: {len(target_genes)}")
    log(f"[Stats] Pretrain RNA - cells: {adata_pre.n_obs}, genes: {adata_pre.n_vars}")
    log(f"[Stats] Finetune RNA - cells: {adata_ft.n_obs}, genes: {adata_ft.n_vars}")

    return out_pretrain, out_finetune


def remove_high_mt(adata, mt_percent=15):
    if "gene_name" in adata.var.columns:
        gene_symbols = adata.var["gene_name"].astype(str)
    elif "gene_symbol" in adata.var.columns:
        gene_symbols = adata.var["gene_symbol"].astype(str)
    else:
        gene_symbols = adata.var_names.astype(str)

    adata.var["mt"] = gene_symbols.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    before_cells = adata.n_obs
    adata = adata[adata.obs.pct_counts_mt <= mt_percent].copy()
    log(f"[INFO] Removed {before_cells - adata.n_obs} cells with >{mt_percent}% mitochondrial content")
    return adata


def remove_high_total_counts_by_batch(adata, batch_key: str = "batch", percentile: float = 99.5):
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile must be in (0, 100), got {percentile}.")

    before_cells = adata.n_obs
    if before_cells == 0:
        return adata

    if sparse.issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        total_counts = np.asarray(adata.X, dtype=np.float64).sum(axis=1).ravel()

    adata.obs["total_counts"] = total_counts

    if batch_key in adata.obs.columns:
        batch_labels = adata.obs[batch_key].astype("string").fillna("__MISSING__")
        keep_mask = np.ones(before_cells, dtype=bool)

        for batch_name in batch_labels.unique():
            batch_mask = (batch_labels == batch_name).fillna(False).astype(bool).to_numpy()
            batch_counts = total_counts[batch_mask]
            if batch_counts.size == 0:
                continue
            upper = float(np.percentile(batch_counts, percentile))
            drop_mask = batch_mask & (total_counts > upper)
            keep_mask &= ~drop_mask
            log(
                f"[INFO] Batch {batch_name}: total_counts P{percentile:g}={upper:.2f}, "
                f"removed {int(drop_mask.sum())} cells."
            )
    else:
        upper = float(np.percentile(total_counts, percentile))
        keep_mask = total_counts <= upper
        log(
            f"[WARNING] Batch key '{batch_key}' not found; "
            f"using global total_counts P{percentile:g}={upper:.2f}."
        )

    adata = adata[keep_mask].copy()
    log(
        f"[INFO] Removed {before_cells - adata.n_obs} cells with total_counts above "
        f"per-batch P{percentile:g}."
    )
    return adata


def chunk_iter(total_size: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start < total_size:
        end = min(start + chunk_size, total_size)
        yield start, end
        start = end


def cluster_cells(
    adata_rna: AnnData,
    marker_genes_path: Optional[str] = None,
    result_folder: str = "./snv_result",
    celltype_key: str = "cell_cluster",
    top_marker_genes: int = 20,
    resolution: float = RESOLUTION,
):
    os.makedirs(result_folder, exist_ok=True)
    if "X_latent" not in adata_rna.obsm:
        raise KeyError(
            "'X_latent' missing from AnnData.obsm. Encode RNA profiles before clustering and UMAP."
        )
    adata_tmp = adata_rna.copy()
    log("[INFO] Running normalization and log transformation on reference AnnData copy...")
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=3000)
    log("[INFO] Using encoder latent representation 'X_latent' to compute neighbors and UMAP.")
    sc.pp.neighbors(adata_tmp, use_rep="X_latent")
    sc.tl.umap(adata_tmp)
    sc.tl.leiden(adata_tmp, resolution=resolution, key_added=celltype_key)
    adata_rna.obs[celltype_key] = adata_tmp.obs[celltype_key]
    if "X_umap" in adata_tmp.obsm:
        adata_rna.obsm["X_umap"] = adata_tmp.obsm["X_umap"].copy()
    if "umap" in adata_tmp.uns:
        adata_rna.uns["umap"] = adata_tmp.uns["umap"].copy()
    for obsp_key in ("connectivities", "distances"):
        if obsp_key in adata_tmp.obsp:
            adata_rna.obsp[obsp_key] = adata_tmp.obsp[obsp_key].copy()

    log("Computing marker genes for each cell cluster...")
    sc.tl.rank_genes_groups(adata_tmp, celltype_key, method="wilcoxon", use_raw=False)
    marker_df = sc.get.rank_genes_groups_df(adata_tmp, group=None)
    marker_top = (
        marker_df.groupby("group", sort=False)
        .head(top_marker_genes)
        .reset_index(drop=True)
    )
    if marker_genes_path is None:
        marker_genes_path = os.path.join(
            result_folder, f"{celltype_key}_marker_genes.csv"
        )
    marker_dir = os.path.dirname(marker_genes_path)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    marker_top.to_csv(marker_genes_path, index=False)
    log(f"Saved top marker genes per cluster to {marker_genes_path}")
    latent_h5ad_path = os.path.join(result_folder, "adata_rna_latent_labeled.h5ad")
    adata_rna.write_h5ad(latent_h5ad_path)
    log(f"[INFO] Saved RNA AnnData with latent representation to {latent_h5ad_path}")


def tensors_from_anndata(
    adata_rna,
    adata_snv,
    batch_key: str = "batch",
    dense: bool = True,
) -> Tuple[
    Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    torch.Tensor,
    List[str],
    List[str],
    AnnData,
    AnnData,
    int,
]:
    adata_rna_aligned = adata_rna
    adata_snv_aligned = adata_snv
    if not np.array_equal(adata_rna.obs_names, adata_snv.obs_names):
        log("[INFO] Aligning cells between RNA and SNV data...")
        # Keep RNA order for shared cells.
        shared_barcodes = adata_rna.obs_names[adata_rna.obs_names.isin(adata_snv.obs_names)]
        log(
            f"[INFO] Retaining {len(shared_barcodes)} shared cells after alignment "
            f"(RNA-only: {len(adata_rna.obs_names) - len(shared_barcodes)}, "
            f"SNV-only: {len(adata_snv.obs_names) - len(shared_barcodes)})."
        )
        if len(shared_barcodes) > 0:
            adata_rna_aligned = adata_rna[shared_barcodes].copy()
            adata_snv_aligned = adata_snv[shared_barcodes].copy()
        elif batch_key in adata_rna.obs:
            rna_composite_names = (
                adata_rna.obs[batch_key].astype(str).str.strip()
                + "_"
                + pd.Index(adata_rna.obs_names).astype(str).str.strip()
            )
            rna_composite_index = pd.Index(rna_composite_names)
            shared_composite = rna_composite_index[rna_composite_index.isin(adata_snv.obs_names)]
            log(
                f"[INFO] Retaining {len(shared_composite)} shared cells after "
                f"trying RNA '{batch_key}' + '_' + barcode IDs."
            )
            if len(shared_composite) > 0:
                if rna_composite_index.has_duplicates:
                    duplicate_preview = (
                        rna_composite_index[rna_composite_index.duplicated()]
                        .unique()
                        .astype(str)
                        .tolist()[:5]
                    )
                    raise ValueError(
                        "Cannot align RNA/SNV cells with composite "
                        f"'{batch_key}_barcode' IDs because duplicates were found: "
                        f"{duplicate_preview}"
                    )

                rna_lookup = pd.Series(np.arange(adata_rna.n_obs), index=rna_composite_index)
                rna_positions = rna_lookup.loc[shared_composite].to_numpy(dtype=int)
                adata_rna_aligned = adata_rna[rna_positions].copy()
                adata_snv_aligned = adata_snv[shared_composite].copy()
                adata_rna_aligned.obs_names = adata_snv_aligned.obs_names.copy()

        if adata_rna_aligned is adata_rna and len(shared_barcodes) == 0:
            rna_preview = list(map(str, adata_rna.obs_names[:5]))
            snv_preview = list(map(str, adata_snv.obs_names[:5]))
            raise ValueError(
                "No overlapping cell barcodes between RNA and SNV AnnData. "
                f"RNA cells={adata_rna.n_obs}, SNV cells={adata_snv.n_obs}. "
                f"RNA preview={rna_preview}; SNV preview={snv_preview}. "
                f"Tried direct obs_names matching and, when available, '{batch_key}_barcode' matching. "
                "Fix obs_names/barcode formatting before SNV effect training."
            )
    assert np.array_equal(
        adata_rna_aligned.obs_names, adata_snv_aligned.obs_names
    ), "Cells not aligned."
    if adata_rna_aligned.n_obs == 0:
        raise ValueError("Aligned RNA/SNV AnnData contains zero cells.")

    if dense:
        # RNA: float32.
        log("[INFO] Converting RNA to dense tensor...")
        if sparse.issparse(adata_rna_aligned.X):
            X_data = adata_rna_aligned.X.toarray().astype(np.float32)
        else:
            X_data = np.asarray(adata_rna_aligned.X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_data)
        del X_data
        gc.collect()

        # SNV: int8 for memory.
        log("[INFO] Converting SNV to dense tensor (int8)...")
        if sparse.issparse(adata_snv_aligned.X):
            G_data = adata_snv_aligned.X.toarray().astype(np.int8)
        else:
            G_data = np.asarray(adata_snv_aligned.X, dtype=np.int8)
        G_tensor = torch.from_numpy(G_data)
        del G_data
        gc.collect()
    else:
        log("[INFO] Keeping RNA/SNV matrices in sparse/array format for lazy densification.")
        X_tensor = adata_rna_aligned.X
        if not sparse.issparse(X_tensor):
            X_tensor = np.asarray(X_tensor, dtype=np.float32)
        G_tensor = adata_snv_aligned.X
        if not sparse.issparse(G_tensor):
            G_tensor = np.asarray(G_tensor, dtype=np.int8)

    # Batch codes.
    if batch_key in adata_rna_aligned.obs:
        batch_series = adata_rna_aligned.obs[batch_key]
        if not pd.api.types.is_categorical_dtype(batch_series):
            batch_series = batch_series.astype("category")
        batch_codes = batch_series.cat.codes.values.astype(np.int64)
        n_batches = len(batch_series.cat.categories)
        if (batch_codes < 0).any():
            log(f"[WARN] Found missing batch labels under key '{batch_key}'; assigning them to 0.")
            batch_codes = np.where(batch_codes < 0, 0, batch_codes)
    else:
        log(
            f"[WARN] Batch key '{batch_key}' not found in RNA AnnData.obs. Assuming single batch."
        )
        batch_codes = np.zeros(adata_rna_aligned.n_obs, dtype=np.int64)
        n_batches = 1

    B_tensor = torch.from_numpy(batch_codes)

    return (
        X_tensor,
        G_tensor,
        B_tensor,
        list(adata_rna_aligned.var_names),
        list(adata_snv_aligned.var_names),
        adata_rna_aligned,
        adata_snv_aligned,
        n_batches,
    )


def align_adata_rna_with_genes(adata_rna, gene_list: List[str]):
    """Reorder columns of ``adata_rna`` to match ``gene_list`` from pre-training."""
    missing = [gene_name for gene_name in gene_list if gene_name not in adata_rna.var_names]
    if missing:
        raise ValueError(
            f"adata_rna missing {len(missing)} genes from pretrain: {missing[:5]}"
        )
    return adata_rna[:, gene_list].copy()


def _strip_cell_prefix_before_underscore(cell_id: object) -> str:
    """Return the barcode portion after a sample prefix like ``SCC01_BARCODE``."""

    text = str(cell_id).strip()
    if "_" not in text:
        return text
    return text.rsplit("_", 1)[1]


def _prefix_stripped_cell_index(obs_names: Iterable[object]) -> pd.Index:
    return pd.Index(_strip_cell_prefix_before_underscore(name) for name in obs_names)


def _dense_batch_from_matrix(
    matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    indices,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        if not isinstance(indices, slice) and not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=matrix.device)
        batch = matrix[indices]
        if device is not None:
            batch = batch.to(device)
        return batch.to(dtype)

    if sparse.issparse(matrix):
        batch_np = matrix[indices].toarray()
    else:
        batch_np = np.asarray(matrix[indices])

    batch = torch.from_numpy(batch_np).to(dtype)
    if device is not None:
        batch = batch.to(device)
    return batch


class LazyAnndataDataset(Dataset):
    """Return indices for lazy densification inside the collate function."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index


def make_lazy_collate_fn(
    rna_matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    snv_matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    batch_labels: Union[torch.Tensor, np.ndarray],
):
    def _collate(indices: List[int]):
        idx = np.asarray(indices, dtype=np.int64)
        expression_batch = _dense_batch_from_matrix(rna_matrix, idx, torch.float32)
        snv_batch = _dense_batch_from_matrix(snv_matrix, idx, torch.float32)
        if isinstance(batch_labels, torch.Tensor):
            batch_labels_batch = batch_labels[idx]
        else:
            batch_labels_batch = torch.from_numpy(np.asarray(batch_labels)[idx])
        return expression_batch, snv_batch, batch_labels_batch

    return _collate


@torch.no_grad()
def mean_attention_per_snv(
    model: nn.Module,
    rna_matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    snv_matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    batch_size: int = 256,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute mean attention per SNV across cells.
    Memory Optimized: Moves data to GPU batch-by-batch to avoid OOM.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    num_snvs = snv_matrix.shape[1]
    # Small accumulators stay on GPU.
    sum_attention = torch.zeros(num_snvs, device=device)
    count_attention = torch.zeros(num_snvs, device=device)

    snv_embedding_matrix = model.snv_embedding.weight.to(device)  # [num_snvs, emb]

    num_cells = rna_matrix.shape[0]
    for batch_start, batch_end in chunk_iter(num_cells, batch_size):
        expression_batch = _dense_batch_from_matrix(
            rna_matrix, slice(batch_start, batch_end), torch.float32, device=device
        )
        snv_batch = _dense_batch_from_matrix(
            snv_matrix, slice(batch_start, batch_end), torch.float32, device=device
        )
        # Encode.
        latent_mean, _latent_log_var = model.encode(expression_batch)
        latent_representation = latent_mean  # Use mean latent.

        nonzero_pairs = (snv_batch != 0).nonzero(as_tuple=False)  # [p, 2]

        if nonzero_pairs.numel() == 0:
            continue

        # Pair logits.
        num_pairs = nonzero_pairs.shape[0]
        logits = torch.empty(num_pairs, device=device)

        # Chunk pairs.
        for pair_start, pair_end in chunk_iter(num_pairs, pair_chunk):
            cell_index_slice = nonzero_pairs[pair_start:pair_end, 0]  # Local cell idx.
            snv_index_slice = nonzero_pairs[pair_start:pair_end, 1]  # Global SNV idx.

            # Gather features.
            z_gather = latent_representation[cell_index_slice]
            e_gather = snv_embedding_matrix[snv_index_slice]

            cond_inp = torch.cat([z_gather, e_gather], dim=1)
            e_cond = model.cond_mlp(cond_inp)

            attn_inp = torch.cat([z_gather, e_cond], dim=1)
            # Forward logits.
            logits[pair_start:pair_end] = model.attn_mlp(attn_inp).squeeze(-1)

        # Stable per-cell softmax.
        logits = torch.clamp(logits, max=50.0)
        exp_logits = torch.exp(logits)

        # Denominator.
        current_batch_size = expression_batch.shape[0]
        denominator = torch.zeros(current_batch_size, device=device)
        denominator.index_add_(0, nonzero_pairs[:, 0], exp_logits)

        # Weights.
        gathered_denominator = denominator[nonzero_pairs[:, 0]] + 1e-12
        attention_weights = exp_logits / gathered_denominator
        # Accumulate global stats.
        segment_snv_indices = nonzero_pairs[:, 1]  # Global SNV idx.
        sum_attention.index_add_(0, segment_snv_indices, attention_weights)
        count_attention.index_add_(0, segment_snv_indices, torch.ones_like(attention_weights))

    # Mean attention.
    mean_attention = (
        (sum_attention / torch.clamp(count_attention, min=1.0))
        .clamp(min=0.0)
        .detach()
        .cpu()
        .numpy()
    )
    return mean_attention


def rank_snvs_by_attention(
    model: nn.Module,
    rna_matrix: torch.Tensor,
    snv_matrix: torch.Tensor,
    adata_snv,
    top_k: int = 2000,
    batch_size: int = 256,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    scores = mean_attention_per_snv(
        model,
        rna_matrix,
        snv_matrix,
        batch_size=batch_size,
        pair_chunk=pair_chunk,
        device=device,
    )
    order = np.argsort(scores)[::-1]
    top_idx = order[:top_k]
    top_attention_df = pd.DataFrame(
        {
            "SNV": np.array(adata_snv.var_names)[top_idx],
            "Attention_Score": scores[top_idx],
        }
    )
    return top_attention_df




def _strip_distributed_prefix(state_dict: dict) -> dict:
    """Remove the ``module.`` prefix added by parallel wrappers such as DDP.

    This makes checkpoints saved with multiple GPUs compatible with single-GPU or
    CPU-only evaluation runs as well as multiprocessing workers that instantiate
    bare ``SNVPerturbationModel`` instances.
    """

    if not state_dict:
        return state_dict

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    return cleaned


def score_by_celltype(
    model,
    adata_rna,
    adata_snv,
    ann_df,
    celltype_key: str = None,
    groups: list = None,
    top_k_attention: int = 2000,
    attn_batch: int = 128,
    device=None,
    cell_batch: int = 128,
    gene_sample_cells: int = None,
    use_mean: bool = True,
    normalize_delta: bool = True,
    eps: float = 1e-6,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    batch_key: str = "batch",
):
    """
    For each cell type:
      1) Rank SNVs by their mean attention weights (optionally keeping the top-K).
      2) Perform counterfactual scoring and summarize the most affected genes.
    Returns a dictionary mapping each group to the tuple
    ``(top_attention_dataframe, counterfactual_scores_dataframe)``.
    """
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    # Align shared cells.
    shared = adata_rna.obs_names.intersection(adata_snv.obs_names)
    adata_rna = adata_rna[shared].copy()
    adata_snv = adata_snv[shared].copy()
    ann_df = _prepare_annotation_dataframe(ann_df, adata_snv.var_names)
    min_cell = 20
    assert celltype_key in adata_rna.obs.columns, f"{celltype_key} not in adata_rna.obs"

    if groups is None:
        groups = (
            list(pd.Categorical(adata_rna.obs[celltype_key]).categories)
            if pd.api.types.is_categorical_dtype(adata_rna.obs[celltype_key])
            else sorted(adata_rna.obs[celltype_key].dropna().unique().tolist())
        )

    results = {}
    log("[INFO] Groups number: %d", len(groups))
    for group_name in groups:
        mask = (adata_rna.obs[celltype_key] == group_name).values
        if mask.sum() < 20:
            log(f"[WARN] group {group_name} has too few cells ({mask.sum()}), skip.")
            continue

        adata_rna_ct = adata_rna[mask]
        adata_snv_ct = adata_snv[mask]

        # Count +1/-1 per SNV.
        snv_matrix = adata_snv_ct.X
        if sparse.isspmatrix(snv_matrix):
            n_pos = np.asarray((snv_matrix == 1).sum(axis=0)).ravel()
            n_neg = np.asarray((snv_matrix == -1).sum(axis=0)).ravel()
        else:
            n_pos = np.asarray(np.sum(snv_matrix == 1, axis=0)).ravel()
            n_neg = np.asarray(np.sum(snv_matrix == -1, axis=0)).ravel()

        valid_mask = (n_pos > min_cell) & (n_neg > min_cell)
        if not np.any(valid_mask):
            log(
                f"[INFO] group {group_name} has no SNVs with >{min_cell} occurrences of both -1 and 1, skip."
            )
            continue

        valid_snv_names = set(adata_snv_ct.var_names[valid_mask].tolist())

        # Build tensors for this cell type.
        (
            X_tensor,
            G_tensor,
            B_tensor,
            gene_names,
            snv_names,
            adata_rna_ct,
            adata_snv_ct,
            _n_batches_ct,
        ) = tensors_from_anndata(adata_rna_ct, adata_snv_ct, batch_key=batch_key)

        # Step 1: rank SNVs and broadcast.
        attn_top_k = adata_snv_ct.n_vars if top_k_attention == -1 else top_k_attention
        top_attn_df = None
        if rank == 0:
            top_attn_df = rank_snvs_by_attention(
                model,
                X_tensor,
                G_tensor,
                adata_snv_ct,
                top_k=attn_top_k,
                batch_size=attn_batch,
                pair_chunk=pair_chunk,
                device=device,
            )
            top_attn_df = top_attn_df[top_attn_df["SNV"].isin(valid_snv_names)].reset_index(
                drop=True
            )
            # Keep attention order, drop duplicates.
            top_snv_names = list(dict.fromkeys(top_attn_df["SNV"].tolist()))
            local_empty = 1 if len(top_snv_names) == 0 else 0
        else:
            top_snv_names = []
            local_empty = 1

        # Step A: sync skip flag.
        if is_dist:
            flag = torch.tensor([local_empty], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            all_empty = flag.item() == dist.get_world_size()
            if all_empty:
                if rank == 0:
                    log(f"[INFO] group {group_name} has no SNVs passing attention filters, skip.")
                continue
        elif local_empty:
            log(f"[INFO] group {group_name} has no SNVs passing attention filters, skip.")
            continue

        # Step B: broadcast SNV list.
        if is_dist:
            obj = [top_snv_names if rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            top_snv_names = obj[0] or []
        # Skip empty SNV list.
        if len(top_snv_names) == 0:
            if (not is_dist) or rank == 0:
                log(f"[WARNING] group {group_name} has no SNVs left after merging across ranks, skip.")
            continue
        # Map SNV names to columns.
        snv_local_idx = adata_snv_ct.var_names.get_indexer(top_snv_names)
        assert (snv_local_idx >= 0).all(), "Some SNVs not found in cell-type matrix."

        # Step 2: counterfactual scoring.
        df_scores = batch_score_all_snvs(
            model,
            X_tensor,  # Cell-type RNA tensor.
            G_tensor,  # Cell-type SNV tensor.
            B_tensor,
            snv_indices=snv_local_idx.tolist(),  # Local SNV column indices.
            snv_names=top_snv_names,
            gene_names=gene_names,
            ann_df=ann_df,
            topk_genes=10,
            device=device,
            cell_batch=cell_batch,
            gene_sample_cells=gene_sample_cells,
            use_mean=use_mean,
            normalize_delta=normalize_delta,
            eps=eps,
            pair_chunk=pair_chunk,
        )
        df_scores.insert(0, "celltype", group_name)
        if rank == 0:
            top_attn_df.insert(0, "celltype", group_name)
            results[group_name] = (top_attn_df, df_scores)
        # Memory cleanup.
        del adata_rna_ct, adata_snv_ct
        del X_tensor, G_tensor, B_tensor
        del top_attn_df, df_scores
        del snv_matrix

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return results


def _align_celltypes_for_scores(
    adata_snv: AnnData, adata_rna: AnnData, celltype_key: str
) -> Tuple[AnnData, AnnData, pd.Series]:
    """Align SNV AnnData with the cell type annotations from RNA AnnData.

    Returns
    -------
    Tuple containing:
        - SNV AnnData restricted to the shared set of cells.
        - RNA AnnData restricted and ordered identically to the SNV AnnData.
        - Cell type annotations aligned with the shared cells.
    """

    if celltype_key not in adata_rna.obs.columns:
        raise KeyError(f"{celltype_key} not found in adata_rna.obs")

    shared_cells = adata_snv.obs_names.intersection(adata_rna.obs_names)
    if len(shared_cells) == 0:
        raise ValueError("No overlapping cells between adata_snv and adata_rna")

    if len(shared_cells) < adata_snv.n_obs:
        log(
            f"[WARN] Subsetting SNV AnnData to {len(shared_cells)} common cells (out of {adata_snv.n_obs})."
        )

    adata_snv_aligned = adata_snv[shared_cells].copy()
    adata_rna_aligned = adata_rna[shared_cells].copy()
    adata_rna_aligned = adata_rna_aligned[adata_snv_aligned.obs_names].copy()

    celltypes = adata_rna_aligned.obs[celltype_key].astype(str)
    celltypes = pd.Series(
        celltypes, index=adata_snv_aligned.obs_names, name=celltype_key
    )
    return adata_snv_aligned, adata_rna_aligned, celltypes


def _positive_carrier_sets_for_subset(
    matrix,
    row_indices: np.ndarray,
    snv_indices: List[int],
) -> Dict[int, Set[int]]:
    """Return local row positions where each SNV has positive carrier value ``1``."""

    carrier_sets: Dict[int, Set[int]] = {}
    if row_indices.size == 0 or len(snv_indices) == 0:
        return carrier_sets

    if sparse.issparse(matrix):
        subset = matrix[row_indices, :][:, snv_indices].tocsc()
        for local_col, snv_idx in enumerate(snv_indices):
            start = subset.indptr[local_col]
            end = subset.indptr[local_col + 1]
            rows = subset.indices[start:end]
            values = subset.data[start:end]
            positive_rows = rows[values == 1]
            carrier_sets[snv_idx] = set(int(row) for row in positive_rows.tolist())
        return carrier_sets

    matrix_array = np.asarray(matrix)
    subset = matrix_array[np.ix_(row_indices, np.asarray(snv_indices, dtype=int))]
    for local_col, snv_idx in enumerate(snv_indices):
        positive_rows = np.flatnonzero(subset[:, local_col] == 1)
        carrier_sets[snv_idx] = set(int(row) for row in positive_rows.tolist())
    return carrier_sets


def _best_explanatory_edge(
    edge_records: List[Dict],
    removed_snv: str,
    kept_snv: str,
) -> Optional[Dict]:
    """Pick the most informative high-cooccurrence edge for an audit row."""

    candidate_edges = [
        edge
        for edge in edge_records
        if removed_snv in {edge["snv_1"], edge["snv_2"]}
    ]
    if not candidate_edges:
        return None

    return max(
        candidate_edges,
        key=lambda edge: (
            kept_snv in {edge["snv_1"], edge["snv_2"]},
            edge["jaccard_index"],
            edge["overlap_coefficient"],
            edge["cooccurrence_count"],
        ),
    )


def filter_highly_cooccurring_snvs_by_celltype(
    all_scores: pd.DataFrame,
    adata_snv: AnnData,
    adata_rna: AnnData,
    celltype_key: str,
    sample_key: str,
    score_column: str = "score_euclidean",
    min_carrier_cells: int = COOCCURRENCE_MIN_CARRIER_CELLS,
    min_shared_cells: int = COOCCURRENCE_MIN_SHARED_CELLS,
    jaccard_threshold: float = COOCCURRENCE_JACCARD_THRESHOLD,
    overlap_threshold: float = COOCCURRENCE_OVERLAP_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove redundant high-cooccurrence SNVs while keeping the top-scoring representative."""

    audit_columns = [
        "celltype",
        "sample",
        "removed_snv",
        "kept_snv",
        "edge_partner_snv",
        "removed_score_euclidean",
        "kept_score_euclidean",
        "jaccard_index",
        "overlap_coefficient",
        "cooccurrence_count",
        "removed_frequency",
        "kept_frequency",
    ]
    audit_index = pd.Index(audit_columns)
    empty_audit = pd.DataFrame(columns=audit_index)
    if all_scores.empty:
        return all_scores.copy(), empty_audit

    required_columns = {"celltype", "SNV", score_column}
    missing_columns = required_columns.difference(all_scores.columns)
    if missing_columns:
        log(
            "[WARN] Skipping SNV co-occurrence de-duplication; score table is missing columns: %s",
            ", ".join(sorted(missing_columns)),
        )
        return all_scores.copy(), empty_audit

    score_df = all_scores.copy()
    score_df["celltype"] = score_df["celltype"].astype(str)
    score_df["SNV"] = score_df["SNV"].astype(str)
    score_df["_vespa_original_order"] = np.arange(score_df.shape[0])
    score_df["_vespa_score_sort"] = pd.to_numeric(
        score_df[score_column], errors="coerce"
    )

    before_dedup = score_df.shape[0]
    score_df = (
        score_df.sort_values(
            ["celltype", "SNV", "_vespa_score_sort", "_vespa_original_order"],
            ascending=[True, True, False, True],
            na_position="last",
        )
        .drop_duplicates(["celltype", "SNV"], keep="first")
        .sort_values("_vespa_original_order")
        .reset_index(drop=True)
    )
    duplicate_rows_removed = before_dedup - score_df.shape[0]
    if duplicate_rows_removed:
        log(
            "[INFO] Removed %d duplicate celltype/SNV score rows before co-occurrence filtering.",
            duplicate_rows_removed,
        )

    deduplicated_scores = score_df.drop(
        columns=["_vespa_original_order", "_vespa_score_sort"], errors="ignore"
    ).reset_index(drop=True)

    if sample_key not in adata_snv.obs.columns:
        log(
            "[WARN] Skipping SNV co-occurrence filtering after duplicate removal; sample key '%s' not found in adata_snv.obs.",
            sample_key,
        )
        return deduplicated_scores, empty_audit

    try:
        adata_snv_aligned, _adata_rna_aligned, celltypes = _align_celltypes_for_scores(
            adata_snv, adata_rna, celltype_key
        )
    except (KeyError, ValueError) as exc:
        log("[WARN] Skipping SNV co-occurrence filtering after duplicate removal: %s", exc)
        return deduplicated_scores, empty_audit

    matrix = adata_snv_aligned.X
    snv_index = pd.Index(adata_snv_aligned.var_names.astype(str))
    sample_values = adata_snv_aligned.obs[sample_key].astype(str)
    celltype_values = celltypes.astype(str)
    sample_array = sample_values.to_numpy()
    celltype_array = celltype_values.to_numpy()

    remove_keys: Set[Tuple[str, str]] = set()
    audit_rows: List[Dict] = []

    for raw_celltype_value, celltype_scores in score_df.groupby("celltype", sort=False):
        celltype_value = str(raw_celltype_value)
        snv_names = list(dict.fromkeys(celltype_scores["SNV"].tolist()))
        snv_positions = snv_index.get_indexer(snv_names)
        present_pairs = [
            (snv_name, int(snv_position))
            for snv_name, snv_position in zip(snv_names, snv_positions)
            if snv_position >= 0
        ]
        if len(present_pairs) < 2:
            continue

        celltype_mask = celltype_array == celltype_value
        if not celltype_mask.any():
            continue

        score_values = celltype_scores["_vespa_score_sort"].fillna(float("-inf"))
        score_by_snv = {
            snv: float(score)
            for snv, score in zip(celltype_scores["SNV"], score_values)
        }
        order_by_snv = {
            snv: int(original_order)
            for snv, original_order in zip(
                celltype_scores["SNV"], celltype_scores["_vespa_original_order"]
            )
        }
        adjacency: Dict[str, Set[str]] = {snv_name: set() for snv_name, _ in present_pairs}
        edge_records: List[Dict] = []

        for sample_value in pd.unique(sample_array[celltype_mask]):
            sample_mask = sample_array == sample_value
            row_indices = np.flatnonzero(celltype_mask & sample_mask)
            if row_indices.size < min_shared_cells:
                continue

            carrier_sets = _positive_carrier_sets_for_subset(
                matrix,
                row_indices,
                [snv_position for _, snv_position in present_pairs],
            )
            active_snvs = [
                (snv_name, snv_position, carrier_sets.get(snv_position, set()))
                for snv_name, snv_position in present_pairs
                if len(carrier_sets.get(snv_position, set())) >= min_carrier_cells
            ]
            if len(active_snvs) < 2:
                continue

            for i in range(len(active_snvs)):
                snv_a, idx_a, cells_a = active_snvs[i]
                for j in range(i + 1, len(active_snvs)):
                    snv_b, idx_b, cells_b = active_snvs[j]
                    cooccur_count = len(cells_a & cells_b)
                    if cooccur_count < min_shared_cells:
                        continue

                    freq_a = len(cells_a)
                    freq_b = len(cells_b)
                    union_count = len(cells_a | cells_b)
                    min_frequency = min(freq_a, freq_b)
                    jaccard = cooccur_count / union_count if union_count else 0.0
                    overlap = cooccur_count / min_frequency if min_frequency else 0.0
                    if jaccard < jaccard_threshold or overlap < overlap_threshold:
                        continue

                    adjacency[snv_a].add(snv_b)
                    adjacency[snv_b].add(snv_a)
                    edge_records.append(
                        {
                            "celltype": celltype_value,
                            "sample": sample_value,
                            "snv_1": snv_a,
                            "snv_2": snv_b,
                            "snv_1_idx": idx_a,
                            "snv_2_idx": idx_b,
                            "snv_1_frequency": freq_a,
                            "snv_2_frequency": freq_b,
                            "cooccurrence_count": cooccur_count,
                            "jaccard_index": jaccard,
                            "overlap_coefficient": overlap,
                        }
                    )

        priority_by_snv = {
            snv: (score_by_snv.get(snv, float("-inf")), -order_by_snv.get(snv, 10**12))
            for snv in adjacency
        }
        kept_snvs: Set[str] = set()
        removed_snvs: Set[str] = set()

        for kept_snv in sorted(
            adjacency,
            key=lambda snv: priority_by_snv[snv],
            reverse=True,
        ):
            if kept_snv in removed_snvs or not adjacency[kept_snv]:
                continue
            kept_snvs.add(kept_snv)

            lower_priority_neighbors = sorted(
                adjacency[kept_snv],
                key=lambda snv: priority_by_snv[snv],
            )
            for removed_snv in lower_priority_neighbors:
                if removed_snv in kept_snvs or removed_snv in removed_snvs:
                    continue
                if priority_by_snv[removed_snv] > priority_by_snv[kept_snv]:
                    continue

                edge = _best_explanatory_edge(edge_records, removed_snv, kept_snv)
                if edge is None:
                    continue
                remove_keys.add((celltype_value, removed_snv))
                removed_snvs.add(removed_snv)
                edge_partner = (
                    edge["snv_2"]
                    if edge["snv_1"] == removed_snv
                    else edge["snv_1"]
                )
                removed_frequency = (
                    edge["snv_1_frequency"]
                    if edge["snv_1"] == removed_snv
                    else edge["snv_2_frequency"]
                )
                kept_frequency = (
                    edge["snv_1_frequency"]
                    if edge["snv_1"] == kept_snv
                    else edge["snv_2_frequency"]
                    if edge["snv_2"] == kept_snv
                    else np.nan
                )
                audit_rows.append(
                    {
                        "celltype": celltype_value,
                        "sample": edge["sample"],
                        "removed_snv": removed_snv,
                        "kept_snv": kept_snv,
                        "edge_partner_snv": edge_partner,
                        "removed_score_euclidean": score_by_snv.get(removed_snv, np.nan),
                        "kept_score_euclidean": score_by_snv.get(kept_snv, np.nan),
                        "jaccard_index": edge["jaccard_index"],
                        "overlap_coefficient": edge["overlap_coefficient"],
                        "cooccurrence_count": edge["cooccurrence_count"],
                        "removed_frequency": removed_frequency,
                        "kept_frequency": kept_frequency,
                    }
                )

    if remove_keys:
        row_keys = pd.MultiIndex.from_arrays(
            [score_df["celltype"], score_df["SNV"]], names=["celltype", "SNV"]
        )
        remove_index = pd.MultiIndex.from_tuples(
            sorted(remove_keys), names=["celltype", "SNV"]
        )
        score_df = score_df.loc[~row_keys.isin(remove_index)].copy()

    filtered_scores = score_df.drop(
        columns=["_vespa_original_order", "_vespa_score_sort"], errors="ignore"
    ).reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows, columns=audit_index)
    log(
        "[INFO] SNV co-occurrence de-duplication kept %d/%d celltype/SNV rows; removed %d highly co-occurring rows.",
        filtered_scores.shape[0],
        before_dedup,
        len(remove_keys),
    )
    return filtered_scores, audit_df


def _normalise_score_column_minmax(
    score_df: pd.DataFrame,
    score_column: str,
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """Add a min-max-normalized [0, 1] version of a score column."""

    if output_column is None:
        output_column = score_column

    if score_column not in score_df.columns:
        log(
            "[WARN] Cannot normalize score column '%s'; column is missing.",
            score_column,
        )
        return score_df

    normalized_df = score_df.copy()
    raw_values = normalized_df.loc[:, [score_column]].iloc[:, 0]
    values_array = np.asarray(
        pd.to_numeric(raw_values.to_list(), errors="coerce"),
        dtype=np.float64,
    )
    valid_mask = ~np.isnan(values_array)
    if not valid_mask.any():
        log(
            "[WARN] Cannot normalize score column '%s'; no numeric values found.",
            score_column,
        )
        normalized_df[output_column] = values_array
        return normalized_df

    valid_values = values_array[valid_mask]
    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))
    value_range = max_value - min_value
    if value_range <= 0.0:
        normalized_values = np.where(valid_mask, 0.0, np.nan)
    else:
        normalized_values = (values_array - min_value) / value_range

    normalized_df[output_column] = normalized_values
    log(
        "[INFO] Added min-max normalized SNV score column '%s' from '%s' "
        "using min=%.6g and max=%.6g.",
        output_column,
        score_column,
        min_value,
        max_value,
    )
    return normalized_df


def _prepare_celltype_score_lookup(
    score_df: pd.DataFrame, score_column: str
) -> Dict[str, pd.DataFrame]:
    """Group SNV score table by cell type for fast lookup."""

    required = {"celltype", "SNV", score_column}
    missing = required.difference(score_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(
            f"Score table is missing required columns: {missing_str}. "
            "Ensure snv_perturbation_scores_by_celltype.csv is available."
        )

    grouped = {
        celltype: group[["SNV", score_column]].reset_index(drop=True)
        for celltype, group in score_df.groupby("celltype", sort=False)
    }
    return grouped


def _compute_single_metric_cell_perturbation(
    adata_snv: AnnData,
    celltypes: pd.Series,
    grouped_scores: Dict[str, pd.DataFrame],
    score_column: str,
    emit_logs: bool = True,
) -> np.ndarray:
    """Compute per-cell perturbation scores for a single SNV score metric."""

    matrix = adata_snv.X
    snv_index = pd.Index(adata_snv.var_names)
    perturbation = np.zeros(adata_snv.n_obs, dtype=float)

    for celltype_value in pd.unique(celltypes):
        mask = (celltypes == celltype_value).fillna(False).astype(bool).to_numpy()
        if mask.sum() == 0:
            continue

        if celltype_value not in grouped_scores:
            if emit_logs:
                log(f"[INFO] Cell type '{celltype_value}' has no SNV scores; skipping.")
            continue

        scores_df = grouped_scores[celltype_value]
        idx = snv_index.get_indexer(scores_df["SNV"])
        valid_mask = idx >= 0

        if not valid_mask.any():
            if emit_logs:
                log(
                    f"[WARN] No SNVs for cell type '{celltype_value}' were found in the AnnData matrix; skipping."
                )
            continue

        if emit_logs and (not valid_mask.all()):
            missing = scores_df.loc[~valid_mask, "SNV"].unique()
            preview = ", ".join(map(str, missing[:5]))
            if len(missing) > 5:
                preview += "..."
            log(
                f"[WARN] Skipping {len(missing)} SNVs for cell type '{celltype_value}' absent in the matrix: {preview}"
            )

        valid_indices = idx[valid_mask]
        snv_scores = scores_df.loc[valid_mask, score_column].to_numpy(dtype=float)

        cell_indices = np.where(mask)[0]
        cell_rows = matrix[cell_indices][:, valid_indices]
        if sparse.isspmatrix(cell_rows):
            cell_rows = cell_rows.tocsr().copy()
            if cell_rows.nnz:
                # Intentional: only cells with value=1 (carrying the SNV) contribute;
                # value=-1 (ref allele) is excluded by design.
                cell_rows.data[cell_rows.data < 0] = 0
                cell_rows.eliminate_zeros()
            contributions = (cell_rows @ snv_scores)
            contributions = np.asarray(contributions).reshape(-1)
            counts = cell_rows.sum(axis=1).A1
        else:
            dense_rows = np.asarray(cell_rows)
            dense_rows = np.maximum(dense_rows, 0)
            contributions = dense_rows @ snv_scores
            counts = dense_rows.sum(axis=1)

        contributions = np.divide(
            contributions,
            counts,
            out=np.zeros_like(contributions, dtype=float),
            where=counts > 0,
        )
        perturbation[cell_indices] = contributions

    if score_column == "score_euclidean":
        perturbation = np.log1p(perturbation)

    return perturbation


def _within_celltype_zscore(
    perturbation_df: pd.DataFrame, celltype_col: str, value_col: str
) -> np.ndarray:
    """Compute z-scores within each cell type for the specified score column."""

    grouped = perturbation_df.groupby(celltype_col)[value_col]
    mean_scores = grouped.transform("mean").to_numpy(dtype=float)
    std_scores = grouped.transform(lambda x: x.std(ddof=0)).to_numpy(dtype=float)
    base_scores = perturbation_df[value_col].to_numpy(dtype=float)
    z_scores = np.zeros_like(base_scores)
    valid_mask = (~np.isnan(std_scores)) & (std_scores != 0)
    z_scores[valid_mask] = (
        base_scores[valid_mask] - mean_scores[valid_mask]
    ) / std_scores[valid_mask]
    return z_scores


def _score_column_alias(score_column: str) -> str:
    """Map internal SNV score column names to compact suffixes."""

    if score_column == "score_euclidean":
        return "euclidean"
    if score_column == "score_cosine_distance":
        return "cosine"
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(score_column)).strip("_").lower()


def compute_cell_perturbation_scores(
    adata_snv: AnnData,
    celltypes: pd.Series,
    grouped_scores: Dict[str, pd.DataFrame],
    score_column: str,
    result_folder: Optional[str] = None,
    grouped_scores_secondary: Optional[Dict[str, pd.DataFrame]] = None,
    secondary_score_column: Optional[str] = None,
) -> pd.DataFrame:
    """Compute per-cell perturbation coefficients from SNV perturbation scores."""

    celltype_col = celltypes.name or "celltype"
    perturbation_primary = _compute_single_metric_cell_perturbation(
        adata_snv=adata_snv,
        celltypes=celltypes,
        grouped_scores=grouped_scores,
        score_column=score_column,
        emit_logs=True,
    )

    perturbation_df = pd.DataFrame(
        {
            "cell_id": adata_snv.obs_names,
            celltype_col: celltypes.values,
            "perturbation_score": perturbation_primary,
        }
    )

    perturbation_df["perturbation_zscore"] = _within_celltype_zscore(
        perturbation_df, celltype_col, "perturbation_score"
    )

    primary_alias = _score_column_alias(score_column)
    primary_score_col = f"perturbation_score_{primary_alias}"
    primary_zscore_col = f"perturbation_zscore_{primary_alias}"
    perturbation_df[primary_score_col] = perturbation_df["perturbation_score"].to_numpy(
        dtype=float
    )
    perturbation_df[primary_zscore_col] = perturbation_df[
        "perturbation_zscore"
    ].to_numpy(dtype=float)

    if (
        grouped_scores_secondary is not None
        and secondary_score_column is not None
        and secondary_score_column != score_column
    ):
        perturbation_secondary = _compute_single_metric_cell_perturbation(
            adata_snv=adata_snv,
            celltypes=celltypes,
            grouped_scores=grouped_scores_secondary,
            score_column=secondary_score_column,
            emit_logs=False,
        )
        secondary_alias = _score_column_alias(secondary_score_column)
        secondary_score_col = f"perturbation_score_{secondary_alias}"
        secondary_zscore_col = f"perturbation_zscore_{secondary_alias}"
        perturbation_df[secondary_score_col] = perturbation_secondary
        perturbation_df[secondary_zscore_col] = _within_celltype_zscore(
            perturbation_df, celltype_col, secondary_score_col
        )

    if result_folder is not None:
        os.makedirs(result_folder, exist_ok=True)
        cell_perturb_path = os.path.join(result_folder, "cell_perturbation_scores.csv")
        perturbation_df.to_csv(cell_perturb_path, index=False)
        log(f"[INFO] Saved per-cell perturbation coefficients to {cell_perturb_path}")

        cell_perturbation_dir = os.path.join(result_folder, "cell_perturbation")
        os.makedirs(cell_perturbation_dir, exist_ok=True)

        high_threshold = 1.0
        low_threshold = -1.0

        for celltype_value, group_df in perturbation_df.groupby(celltype_col):
            high_mask = group_df["perturbation_zscore"] >= high_threshold
            low_mask = group_df["perturbation_zscore"] <= low_threshold
            selected_mask = high_mask | low_mask
            if not selected_mask.any():
                continue

            selected_df = group_df.loc[
                selected_mask,
                ["cell_id", celltype_col, "perturbation_score", "perturbation_zscore"],
            ].copy()
            selected_df["high_perturbation"] = (
                group_df.loc[selected_mask, "perturbation_zscore"] >= high_threshold
            ).astype(int)
            selected_df = selected_df[
                [
                    "cell_id",
                    celltype_col,
                    "high_perturbation",
                    "perturbation_score",
                    "perturbation_zscore",
                ]
            ]

            safe_celltype = (
                re.sub(r"[^0-9A-Za-z._-]+", "_", str(celltype_value)) or "unknown"
            )
            output_filename = f"{safe_celltype}_cell_perturbation.csv"
            output_path = os.path.join(cell_perturbation_dir, output_filename)
            selected_df.to_csv(output_path, index=False)
            log(
                f"[INFO] Saved high/low perturbation cells for '{celltype_value}' to {output_path}"
            )

    return perturbation_df


def plot_cell_perturbation_umap(
    adata_rna: AnnData,
    perturbation_df: pd.DataFrame,
    result_folder: str,
    color_key: str = "perturbation_score",
    cmap: str = "viridis",
    celltype_key: str = "cell_cluster",
    adata_snv: Optional[AnnData] = None,
    rasterized: bool = False,
    output_filename: str = "cell_perturbation_and_celltype_umap.pdf",
) -> None:
    """Generate a set of UMAPs colored by perturbation scores and per-cell metrics."""

    required_cols = {"cell_id", color_key, celltype_key}
    missing_cols = required_cols.difference(perturbation_df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise KeyError(f"Perturbation table missing required columns: {missing_str}")

    perturbation_df = perturbation_df.copy()
    color_key_plot = color_key
    perturbation_title = "Perturbation score"
    if color_key == "perturbation_score":
        top_fraction = 0.1
        if 0 < top_fraction < 1:
            quantile_level = 1.0 - top_fraction
            thresholds = perturbation_df.groupby(celltype_key)[
                "perturbation_score"
            ].quantile(quantile_level)
            thresholds = thresholds.reindex(perturbation_df[celltype_key]).to_numpy(
                dtype=float
            )
            base_scores = perturbation_df["perturbation_score"].to_numpy(dtype=float)
            thresholds = np.where(np.isnan(thresholds), base_scores, thresholds)
            capped_scores = np.minimum(base_scores, thresholds)
            perturbation_df["perturbation_score_capped"] = capped_scores
            color_key_plot = "perturbation_score_capped"
            perturbation_title = "Perturbation score (capped)"

    perturbation_df = perturbation_df.set_index("cell_id")
    shared = adata_rna.obs_names.intersection(perturbation_df.index)
    if shared.empty:
        raise ValueError(
            "No overlapping cells between RNA AnnData and perturbation scores."
        )

    if len(shared) < adata_rna.n_obs:
        log(
            f"[WARN] Restricting RNA AnnData to {len(shared)} cells with perturbation scores (from {adata_rna.n_obs})."
        )

    adata_plot = adata_rna[shared].copy()
    adata_plot.obs[color_key_plot] = perturbation_df.loc[
        adata_plot.obs_names, color_key_plot
    ]
    adata_plot.obs[celltype_key] = perturbation_df.loc[
        adata_plot.obs_names, celltype_key
    ]

    # Optional matched SNV data.
    adata_snv_plot = None
    if adata_snv is not None:
        shared_snv = adata_snv.obs_names.intersection(adata_plot.obs_names)
        if shared_snv.empty:
            log(
                "[WARN] Provided adata_snv has no overlapping cells with RNA AnnData for plotting. Ignoring SNV counts plot."
            )
        else:
            ordered_cells = [
                cell for cell in adata_plot.obs_names if cell in adata_snv.obs_names
            ]
            if ordered_cells:
                adata_snv_plot = adata_snv[ordered_cells].copy()

    def _sum_axis(mat) -> np.ndarray:
        if mat is None:
            return None
        if sparse.issparse(mat):
            return np.asarray(mat.sum(axis=1)).ravel()
        return np.asarray(mat.sum(axis=1)).ravel()

    def _count_nonzero_axis(mat) -> np.ndarray:
        if mat is None:
            return None
        if sparse.issparse(mat):
            return np.asarray((mat != 0).sum(axis=1)).ravel()
        return np.count_nonzero(mat, axis=1)

    # Per-cell UMI counts.
    umi_source = None
    if hasattr(adata_plot, "layers") and "counts" in adata_plot.layers:
        umi_source = adata_plot.layers["counts"]
    elif adata_plot.raw is not None and adata_plot.raw.X is not None:
        umi_source = adata_plot.raw.X
    else:
        umi_source = adata_plot.X
    umi_counts_raw = _sum_axis(umi_source)
    umi_counts = np.log1p(umi_counts_raw)
    adata_plot.obs["umi_counts"] = umi_counts

    snv_counts = None
    if adata_snv_plot is not None:
        snv_counts_raw = _count_nonzero_axis(adata_snv_plot.X)
        snv_counts = np.log1p(snv_counts_raw)
        snv_counts_series = pd.Series(snv_counts, index=adata_snv_plot.obs_names)
        adata_plot.obs["snv_counts"] = (
            snv_counts_series.reindex(adata_plot.obs_names).fillna(0).to_numpy()
        )

    umap_ready = (
        "X_umap" in adata_plot.obsm
        and np.asarray(adata_plot.obsm["X_umap"]).shape[0] == adata_plot.n_obs
    )
    if umap_ready:
        log("[INFO] Reusing existing UMAP coordinates from AnnData.obsm['X_umap'].")
    else:
        log(
            "Preparing RNA data for UMAP visualization using encoder latents ('X_latent') to mitigate batch effects..."
        )
        if "X_latent" not in adata_plot.obsm:
            raise KeyError(
                "'X_latent' missing from AnnData.obsm. Encode RNA profiles before UMAP visualization."
            )
        sc.pp.neighbors(adata_plot, use_rep="X_latent")
        sc.tl.umap(adata_plot)

    log(
        "Saving combined UMAP visualization for perturbation scores, clusters, and per-cell summaries..."
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax in axes:
        ax.set_rasterized(rasterized)

    sc.pl.umap(
        adata_plot,
        color=color_key_plot,
        cmap=cmap,
        ax=axes[0],
        show=False,
        title=perturbation_title,
    )
    sc.pl.umap(
        adata_plot,
        color=celltype_key,
        legend_loc="on data",
        ax=axes[1],
        show=False,
        title="Cell type clusters",
    )

    sc.pl.umap(
        adata_plot,
        color="umi_counts",
        cmap=cmap,
        ax=axes[2],
        show=False,
        title="log1p(UMI count) per cell",
    )

    if snv_counts is not None:
        sc.pl.umap(
            adata_plot,
            color="snv_counts",
            cmap=cmap,
            ax=axes[3],
            show=False,
            title="log1p(SNV count) per cell",
        )
    else:
        axes[3].axis("off")
        axes[3].set_title("SNV data unavailable")

    fig.tight_layout()

    os.makedirs(result_folder, exist_ok=True)
    umap_path = os.path.join(result_folder, output_filename)
    fig.savefig(umap_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"Saved combined UMAP plot to {umap_path}")


def batch_score_all_snvs(
    model,
    rna_matrix: torch.Tensor,
    snv_matrix: torch.Tensor,
    batch_labels: torch.Tensor,
    snv_indices: list,
    snv_names: list,
    gene_names: list,
    ann_df,
    topk_genes: int = 10,
    device: str = None,
    cell_batch: int = 128,
    gene_sample_cells: int = None,
    use_mean: bool = True,
    normalize_delta: bool = True,
    eps: float = 1e-6,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
):
    """
    Distributed & memory-safe batch scoring.
    1. All ranks share a single SNV plan (snv_indices, snv_names) via broadcast.
    2. SNVs are sharded across ranks by stride.
    3. Each rank scores its shard and then all_gather merges results.
    """
    # Prepare annotations.
    ann_df = _prepare_annotation_dataframe(ann_df, snv_names)

    # Detect distributed mode.
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Select device.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")

    # Shard SNVs by rank.
    total_snvs = len(snv_indices)
    log(f"Total SNV count: {total_snvs}.")
    shard_indices = []
    shard_names = []
    for snv_pos in range(rank, total_snvs, world_size):
        shard_indices.append(snv_indices[snv_pos])
        shard_names.append(snv_names[snv_pos])

    log(f"[Rank {rank}] Processing {len(shard_indices)}/{total_snvs} SNVs.")

    # Keep matrices on CPU.
    expression_matrix = rna_matrix.cpu()
    snv_full_matrix = snv_matrix.cpu()
    batch_label_vector = batch_labels.cpu().long()

    model.eval()
    model.to(device)

    n_cells = expression_matrix.shape[0]

    def agg1d(t: torch.Tensor):
        return (t.mean() if use_mean else t.median()).item()

    # Optional cell subsampling.
    if (gene_sample_cells is None) or (gene_sample_cells <= 0) or (gene_sample_cells >= n_cells):
        sampled_cell_indices = torch.arange(n_cells)
    else:
        sampled_cell_indices = torch.arange(min(n_cells, gene_sample_cells))

    local_rows = []

    # Score local SNVs.
    for snv_global_index, snv_name in tqdm(
        list(zip(shard_indices, shard_names)),
        total=len(shard_indices),
        desc=f"Rank {rank} Scoring",
    ):
        # Score one SNV.
        per_cell_scores = score_snv_effects(
            model,
            expression_matrix,
            snv_full_matrix,
            batch_label_vector,
            snv_idx=snv_global_index,
            device=device,
            cell_batch=cell_batch,
            pair_chunk=pair_chunk,
            normalize_delta=normalize_delta,
        )

        if len(per_cell_scores) == 0:
            # No carriers.
            snv_score_cosine_distance = 0.0
            snv_score_euclidean = 0.0
        else:
            snv_score_cosine_distance = float(agg1d(per_cell_scores["cosine_dist"]))
            snv_score_euclidean = float(agg1d(per_cell_scores["euclidean_dist"]))

        # Variant annotation.
        ann_fields = _extract_annotation_metadata(ann_df, snv_name)
        func = ann_fields.get("Func", "")
        gene = ann_fields.get("Gene", "")
        detail = ann_fields.get("GeneDetail", "")
        exonic_func = ann_fields.get("ExonicFunc", "")
        CLNDN = ann_fields.get("CLNDN", "")
        CLNALLELEID = ann_fields.get("CLNALLELEID", "")
        CLNSIG = ann_fields.get("CLNSIG", "")

        local_rows.append(
            {
                "SNV": snv_name,
                "score_euclidean": snv_score_euclidean,
                "score_cosine_distance": snv_score_cosine_distance,
                "Func": func,
                "Gene": gene,
                "CLNALLELEID": CLNALLELEID,
                "CLNDN": CLNDN,
                "CLNSIG": CLNSIG,
                "GeneDetail": detail,
                "ExonicFunc": exonic_func,
            }
        )

    # Gather rows.
    if world_size > 1:
        log(f"[Rank {rank}] Waiting to gather results...")
        dist.barrier()  # Sync ranks.
        gathered_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_rows, local_rows)

        all_rows = []
        for row_list in gathered_rows:
            all_rows.extend(row_list)
    else:
        all_rows = local_rows
    if len(all_rows) == 0:
        # Empty result.
        score_dataframe = pd.DataFrame(
            columns=[
                "SNV",
                "score_euclidean",
                "score_cosine_distance",
                "Func",
                "Gene",
                "CLNALLELEID",
                "CLNDN",
                "CLNSIG",
                "GeneDetail",
                "ExonicFunc",
            ]
        )
        return score_dataframe
    # Sort by score.
    score_dataframe = pd.DataFrame(all_rows).sort_values(
        "score_cosine_distance", ascending=False
    )
    return score_dataframe


def batch_score_snvs_by_cell(
    model,
    rna_matrix: torch.Tensor,
    snv_matrix: torch.Tensor,
    batch_labels: torch.Tensor,
    snv_indices: list,
    snv_names: list,
    cell_ids: list,
    device: str = None,
    cell_batch: int = 128,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    normalize_delta: bool = True,
):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")

    if isinstance(rna_matrix, torch.Tensor):
        expression_matrix = rna_matrix.cpu()
    elif sparse.issparse(rna_matrix):
        # Keep sparse matrix.
        expression_matrix = rna_matrix
    else:
        expression_matrix = torch.from_numpy(np.asarray(rna_matrix, dtype=np.float32))

    if isinstance(snv_matrix, torch.Tensor):
        snv_full_matrix = snv_matrix.cpu()
    elif sparse.issparse(snv_matrix):
        # Keep sparse matrix.
        snv_full_matrix = snv_matrix
    else:
        snv_full_matrix = torch.from_numpy(np.asarray(snv_matrix, dtype=np.int8))
    batch_label_vector = batch_labels.cpu().long()

    model.eval()
    model.to(device)

    total_snvs = len(snv_indices)
    shard_indices = []
    shard_names = []
    for snv_pos in range(rank, total_snvs, world_size):
        shard_indices.append(snv_indices[snv_pos])
        shard_names.append(snv_names[snv_pos])

    log(f"[Rank {rank}] Processing {len(shard_indices)}/{total_snvs} SNVs for per-cell scoring.")

    local_rows = []

    for snv_global_index, snv_name in tqdm(
        list(zip(shard_indices, shard_names)),
        total=len(shard_indices),
        desc=f"Rank {rank} Per-Cell Scoring",
    ):
        # Sparse-safe indexing.
        if sparse.issparse(snv_full_matrix):
            carriers = snv_full_matrix[:, snv_global_index].toarray().flatten() != 0
        else:
            carriers = snv_full_matrix[:, snv_global_index] != 0

        # Convert carriers to NumPy.
        if hasattr(carriers, 'cpu'):  # Torch tensor case.
            carriers = carriers.cpu().numpy()
        elif not isinstance(carriers, np.ndarray):
            carriers = np.asarray(carriers)

        if not carriers.any():
            continue

        per_cell_scores = score_snv_effects(
            model,
            expression_matrix,
            snv_full_matrix,
            batch_label_vector,
            snv_idx=snv_global_index,
            device=device,
            cell_batch=cell_batch,
            pair_chunk=pair_chunk,
            normalize_delta=normalize_delta,
            restrict_to_carriers=True,
        )
        if not per_cell_scores:
            continue

        carrier_indices = torch.nonzero(torch.from_numpy(carriers), as_tuple=False).squeeze(1).tolist()
        if len(carrier_indices) != len(per_cell_scores["euclidean_dist"]):
            raise ValueError(
                "Mismatch between carrier count and per-cell scores for SNV "
                f"{snv_name} (expected {len(carrier_indices)}, got {len(per_cell_scores['euclidean_dist'])})."
            )

        for idx, cell_index in enumerate(carrier_indices):
            local_rows.append(
                {
                    "SNV": snv_name,
                    "cell_id": cell_ids[cell_index],
                    "score_euclidean": float(per_cell_scores["euclidean_dist"][idx]),
                    "score_cosine_distance": float(per_cell_scores["cosine_dist"][idx]),
                }
            )

    if world_size > 1:
        dist.barrier()
        gathered_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_rows, local_rows)
        all_rows = []
        for row_list in gathered_rows:
            all_rows.extend(row_list)
    else:
        all_rows = local_rows

    if len(all_rows) == 0:
        return pd.DataFrame(
            columns=[
                "SNV",
                "cell_id",
                "score_euclidean",
                "score_cosine_distance",
            ]
        )

    return pd.DataFrame(all_rows)




@torch.no_grad()
def score_snv_effects(
    model,
    rna_matrix: torch.Tensor,
    snv_matrix: torch.Tensor,
    batch_labels: torch.Tensor,
    snv_idx: int,
    device: str = "cuda",
    cell_batch: int = 128,
    pair_chunk: int = DEFAULT_PAIR_CHUNK,
    normalize_delta: bool = True,
    eps: float = 1e-6,
    restrict_to_carriers: bool = True,
) -> Dict[str, torch.Tensor]:
    model.eval()
    device = torch.device(device)

    # CPU-side filtering.
    num_cells, num_snvs = snv_matrix.shape
    assert 0 <= snv_idx < num_snvs, f"snv_idx={snv_idx} out scope [0, {num_snvs - 1}]"

    # Sparse-safe indexing.
    if sparse.issparse(snv_matrix):
        carriers = snv_matrix[:, snv_idx].toarray().flatten() != 0
    else:
        carriers = snv_matrix[:, snv_idx] != 0

    # Convert carriers to NumPy.
    if hasattr(carriers, 'cpu'):  # Torch tensor case.
        carriers = carriers.cpu().numpy()
    elif not isinstance(carriers, np.ndarray):
        carriers = np.asarray(carriers)

    batch_label_full = batch_labels.cpu().long()
    if restrict_to_carriers:
        if carriers.sum() == 0:
            return {}
        if sparse.issparse(rna_matrix):
            rna_use = rna_matrix[carriers]
        else:
            rna_use = rna_matrix[carriers]
        if sparse.issparse(snv_matrix):
            snv_use = snv_matrix[carriers]
        else:
            snv_use = snv_matrix[carriers]
        batch_use = batch_label_full[carriers]
    else:
        rna_use = rna_matrix
        snv_use = snv_matrix
        batch_use = batch_label_full
    scores_out = {"cosine_dist": [], "euclidean_dist": []}

    # Filtered cell count.
    N_subset = rna_use.shape[0]
    # Batched scoring.
    for batch_start, batch_end in chunk_iter(N_subset, cell_batch):
        # Expression batch.
        if sparse.issparse(rna_use):
            expression_batch_slice = rna_use[batch_start:batch_end]
            if sparse.issparse(expression_batch_slice):
                expression_batch = torch.from_numpy(
                    expression_batch_slice.toarray().astype(np.float32)
                ).to(device)
            else:
                expression_batch = torch.from_numpy(
                    expression_batch_slice.astype(np.float32)
                ).to(device)
        else:
            expression_batch = rna_use[batch_start:batch_end].to(device).float()

        # SNV batch.
        if sparse.issparse(snv_use):
            snv_batch_slice = snv_use[batch_start:batch_end]
            if sparse.issparse(snv_batch_slice):
                snv_batch = torch.from_numpy(
                    snv_batch_slice.toarray().astype(np.float32)
                ).to(device).clone()
            else:
                snv_batch = torch.from_numpy(
                    snv_batch_slice.astype(np.float32)
                ).to(device).clone()
        else:
            snv_batch = snv_use[batch_start:batch_end].to(device).float().clone()  # Avoid in-place side effects.

        batch_labels_batch = batch_use[batch_start:batch_end].to(device).long()

        latent_mean, latent_log_var = model.encode(expression_batch)
        latent_representation = latent_mean

        # Baseline decode.
        mu_ref, _ = model.decode_sparse(
            latent_representation,
            snv_batch,
            batch_labels_batch,
            pair_chunk=pair_chunk,
        )

        # Guard for empty mutation batch.
        covered = snv_batch[:, snv_idx] != 0
        if not covered.any():
            delta = torch.zeros(expression_batch.shape[0], device=device)
            scores_out["cosine_dist"].append(delta.cpu())
            scores_out["euclidean_dist"].append(delta.cpu())
            continue
        snv_batch[:, snv_idx] *= -1.0
        mu_alt, _ = model.decode_sparse(
            latent_representation,
            snv_batch,
            batch_labels_batch,
            pair_chunk=pair_chunk,
        )

        # Cosine distance.
        cos_sim = F.cosine_similarity(mu_ref, mu_alt, dim=1, eps=eps)  # [B]
        cos_sim_delta_cell = 1.0 - cos_sim
        scores_out["cosine_dist"].append(cos_sim_delta_cell.cpu())
        # Euclidean distance.
        euclidean_dist = torch.norm(mu_alt - mu_ref, p=2, dim=1)
        scores_out["euclidean_dist"].append(euclidean_dist.cpu())
    if not scores_out["euclidean_dist"]:
        return {}
    final_scores = {key: torch.cat(value, dim=0) for key, value in scores_out.items()}
    return final_scores


@torch.no_grad()
def _encode_latent_representation(
    model: nn.Module,
    rna_matrix: Union[torch.Tensor, np.ndarray, sparse.spmatrix],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    if isinstance(rna_matrix, torch.Tensor):
        dataloader = torch.utils.data.DataLoader(
            rna_matrix, batch_size=batch_size, shuffle=False, drop_last=False
        )
        use_lazy = False
    else:
        dataset = LazyAnndataDataset(rna_matrix.shape[0])

        def _collate(indices: List[int]) -> torch.Tensor:
            idx = np.asarray(indices, dtype=np.int64)
            return _dense_batch_from_matrix(
                rna_matrix, idx, torch.float32, device=device
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate,
        )
        use_lazy = True
    latents: List[torch.Tensor] = []

    for expression_batch in dataloader:
        if not use_lazy:
            expression_batch = expression_batch.to(device).float()
        latent_mean, _ = model.encode(expression_batch)
        latents.append(latent_mean.cpu())

    if not latents:
        return torch.empty((0, getattr(model, "latent_dim", 0)))

    return torch.cat(latents, dim=0)

__all__ = [
    "DEFAULT_PAIR_CHUNK",
    "log",
    "log_model_parameter_summary",
    "set_seed",
    "get_device",
    "warmup_cosine_lr",
    "_normalise_chromosome",
    "_read_annotation_table",
    "_prepare_annotation_dataframe",
    "_parse_snv_identifier",
    "STANDARD_CHROMOSOMES",
    "_filter_snv_to_standard_chromosomes",
    "_lookup_annotation_row",
    "_extract_annotation_metadata",
    "_distributed_env_ready",
    "_ANNOTATION_FIELD_MAP",
    "filter_snv_by_counts",
    "tensor_from_anndata_X",
    "align_and_preprocess_adata",
    "remove_high_total_counts_by_batch",
    "remove_high_mt",
    "chunk_iter",
    "cluster_cells",
    "tensors_from_anndata",
    "LazyAnndataDataset",
    "make_lazy_collate_fn",
    "align_adata_rna_with_genes",
    "mean_attention_per_snv",
    "rank_snvs_by_attention",
    "_strip_distributed_prefix",
    "score_by_celltype",
    "_align_celltypes_for_scores",
    "_normalise_score_column_minmax",
    "_prepare_celltype_score_lookup",
    "compute_cell_perturbation_scores",
    "plot_cell_perturbation_umap",
    "batch_score_all_snvs",
    "batch_score_snvs_by_cell",
    "score_snv_effects",
    "_encode_latent_representation",
]
