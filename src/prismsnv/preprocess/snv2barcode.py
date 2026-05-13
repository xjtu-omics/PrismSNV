import logging
import math
import os
import re
import sys

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import yaml

from multiprocessing import Pool, cpu_count, Process, Queue
from scipy.sparse import csr_matrix, lil_matrix, hstack
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

matplotlib.use("Agg")
pysam.set_verbosity(0)
logger = logging.getLogger(__name__)
SNV_PATTERN = re.compile(r"^(chr[^:]+|[^:]+):(\d+)_([^>]+)>([^>]+)$", re.IGNORECASE)
REQUIRED_SAMPLE_FILES = ("bam", "vcf", "cb")
SAMPLE_FILE_LABELS = {
    "bam": "BAM",
    "vcf": "VCF",
    "cb": "CB",
    "annotated_vcf": "annotated_vcf",
}


def read_snv_positions(snv_file):
    """
    Read SNV site list.
    Supports two formats:
      1) VCF style: chrom pos ... ref alt (>=5 columns)
      2) TSV style: chrom pos ref alt (4 columns)
    """
    logger.info("Reading SNV positions from %s", snv_file)
    snv_positions = []
    with open(snv_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            snv_positions.append(parts)
    logger.info("Loaded %d SNV positions", len(snv_positions))
    return snv_positions


def _parse_snv_key(snv_key):
    if isinstance(snv_key, tuple) and len(snv_key) == 4:
        return tuple(str(x) for x in snv_key)
    if not isinstance(snv_key, str):
        return None

    parts = snv_key.split(":")
    if len(parts) != 2:
        return None
    chrom = parts[0]
    rest = parts[1].split("_")
    if len(rest) != 2:
        return None
    pos = rest[0]
    ref_alt = rest[1].split(">")
    if len(ref_alt) != 2:
        return None
    return chrom, pos, ref_alt[0], ref_alt[1]


def _coerce_bool(value, setting_name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"settings.{setting_name} must be a boolean value.")


def _coerce_positive_int(value, setting_name):
    if isinstance(value, bool):
        raise ValueError(f"settings.{setting_name} must be a positive integer.")

    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[1-9]\d*", text):
            raise ValueError(f"settings.{setting_name} must be a positive integer.")
        parsed = int(text)
    else:
        raise ValueError(f"settings.{setting_name} must be a positive integer.")

    if parsed <= 0:
        raise ValueError(f"settings.{setting_name} must be a positive integer.")
    return parsed


def _normalize_required_path(value):
    if value is None:
        return None
    try:
        path = os.fspath(value)
    except TypeError:
        return None
    if isinstance(path, str) and not path.strip():
        return None
    return path


def _validate_config_inputs(samples_cfg, af_filter_enabled, snv_union_cfg):
    if not isinstance(samples_cfg, dict) or not samples_cfg:
        raise ValueError("Config must include a non-empty samples mapping.")

    required_sample_keys = list(REQUIRED_SAMPLE_FILES)
    if af_filter_enabled:
        required_sample_keys.append("annotated_vcf")

    missing_inputs = []
    for sample_name, sconf in samples_cfg.items():
        if not isinstance(sconf, dict):
            missing_inputs.append(f"  [{sample_name}] sample config: <must be a mapping>")
            continue

        for key in required_sample_keys:
            label = SAMPLE_FILE_LABELS.get(key, key)
            path = _normalize_required_path(sconf.get(key))
            if path is None:
                missing_inputs.append(f"  [{sample_name}] {label}: <missing config key or empty path>")
                continue
            if not os.path.exists(path):
                missing_inputs.append(f"  [{sample_name}] {label}: {path}")

    snv_union_path = _normalize_required_path(snv_union_cfg)
    if snv_union_path is None:
        missing_inputs.append("  SNV union: <missing config key or empty path>")
    elif snv_union_path != "auto" and not os.path.exists(snv_union_path):
        missing_inputs.append(f"  SNV union: {snv_union_path}")

    if missing_inputs:
        logger.error("Missing or invalid required input files:")
        for item in missing_inputs:
            logger.error(item)
        raise FileNotFoundError(
            f"Missing or invalid {len(missing_inputs)} required input(s). See log above for details."
        )


def _parse_snv_identifier(snv_name):
    match = SNV_PATTERN.match(str(snv_name).strip())
    if match is None:
        raise ValueError(f"Unsupported SNV format: {snv_name}")
    chrom, pos_text, ref, alt = match.groups()
    return chrom, int(pos_text), ref.upper(), alt.upper()


def _build_snv_key(chrom, pos, ref, alt):
    return f"{chrom}:{int(pos)}_{str(ref).upper()}>{str(alt).upper()}"


def _candidate_snv_keys(chrom, pos, ref, alt):
    primary = _build_snv_key(chrom, pos, ref, alt)
    candidates = [primary]
    chrom_text = str(chrom)
    if chrom_text.startswith("chr"):
        secondary = _build_snv_key(chrom_text[3:], pos, ref, alt)
    else:
        secondary = _build_snv_key(f"chr{chrom_text}", pos, ref, alt)
    if secondary not in candidates:
        candidates.append(secondary)
    return candidates


def _parse_info_field(info_text):
    parsed = {}
    for token in str(info_text).split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            parsed[token] = ""
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def _parse_af_value(raw_value):
    if raw_value is None:
        return float("nan")
    text = str(raw_value).strip()
    if not text or text == ".":
        return float("nan")

    numeric_values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece or piece == ".":
            continue
        try:
            numeric_values.append(float(piece))
        except ValueError:
            continue
    if not numeric_values:
        return float("nan")
    return float(max(numeric_values))


def _load_vcf_af_lookup(vcf_path, af_field, af_threshold, strict_snv_format=False):
    af_lookup = {}
    af_field_seen = False
    stats = {
        "vcf_records_total": 0,
        "vcf_records_with_af_gt_threshold": 0,
        "vcf_records_with_missing_af": 0,
        "vcf_records_malformed": 0,
        "vcf_records_duplicate_keys": 0,
    }

    with open(vcf_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue

            columns = line.rstrip("\n").split("\t")
            if len(columns) < 8:
                stats["vcf_records_malformed"] += 1
                if strict_snv_format:
                    raise ValueError(f"Malformed VCF record with <8 columns: {line.strip()}")
                continue

            chrom, pos_text, _id, ref, alt = columns[:5]
            info_text = columns[7]
            try:
                snv_key = _build_snv_key(chrom, int(pos_text), ref, alt)
            except ValueError:
                stats["vcf_records_malformed"] += 1
                if strict_snv_format:
                    raise
                continue

            info_dict = _parse_info_field(info_text)
            if af_field in info_dict:
                af_field_seen = True
            af_value = _parse_af_value(info_dict.get(af_field))
            if np.isnan(af_value):
                stats["vcf_records_with_missing_af"] += 1
            elif float(af_value) > float(af_threshold):
                stats["vcf_records_with_af_gt_threshold"] += 1

            if snv_key in af_lookup:
                stats["vcf_records_duplicate_keys"] += 1
                previous = af_lookup[snv_key]
                if np.isnan(previous) and not np.isnan(af_value):
                    af_lookup[snv_key] = af_value
                elif not np.isnan(af_value):
                    af_lookup[snv_key] = max(previous, af_value)
            else:
                af_lookup[snv_key] = af_value

            stats["vcf_records_total"] += 1

    if stats["vcf_records_total"] > 0 and not af_field_seen:
        raise KeyError(f"AF field '{af_field}' was not found in VCF INFO records: {vcf_path}")

    return af_lookup, stats


def _merge_af_lookups(af_lookups):
    merged_lookup = {}
    for af_lookup in af_lookups:
        for snv_key, af_value in af_lookup.items():
            if snv_key not in merged_lookup:
                merged_lookup[snv_key] = af_value
                continue

            previous = merged_lookup[snv_key]
            if np.isnan(previous) and not np.isnan(af_value):
                merged_lookup[snv_key] = af_value
            elif not np.isnan(af_value):
                merged_lookup[snv_key] = max(previous, af_value)
    return merged_lookup


def _merge_af_stats(stats_list):
    merged_stats = defaultdict(int)
    for stats in stats_list:
        for key, value in stats.items():
            merged_stats[key] += int(value)
    return dict(merged_stats)


def _build_af_filter_detail_table(var_names, af_lookup, af_threshold, strict_snv_format=False):
    records = []
    malformed = 0
    for snv_name in pd.Index(var_names).astype(str):
        try:
            chrom, pos, ref, alt = _parse_snv_identifier(snv_name)
            candidate_keys = _candidate_snv_keys(chrom, pos, ref, alt)
        except ValueError:
            malformed += 1
            if strict_snv_format:
                raise
            candidate_keys = []

        matched_key = next((key for key in candidate_keys if key in af_lookup), None)
        vcf_record_found = matched_key is not None
        af_value = af_lookup.get(matched_key, float("nan")) if matched_key is not None else float("nan")
        has_numeric_af = bool(np.isfinite(af_value))
        is_common = bool(has_numeric_af and float(af_value) > float(af_threshold))
        records.append(
            {
                "SNV": snv_name,
                "matched_vcf_key": matched_key,
                "annotated_af": af_value,
                "vcf_record_found": vcf_record_found,
                "af_value_is_numeric": has_numeric_af,
                "remove_af_gt_threshold": is_common,
            }
        )
    return pd.DataFrame(records), malformed


def _derive_af_filtered_h5ad_path(h5ad_path):
    base, ext = os.path.splitext(h5ad_path)
    if ext.lower() != ".h5ad":
        return h5ad_path + "_af_filtered"
    return base + "_af_filtered" + ext


def _write_population_af_filtered_h5ad(
    input_h5ad,
    output_h5ad,
    af_lookup,
    af_field,
    af_threshold,
    vcf_label,
    vcf_stats,
    strict_snv_format=False,
):
    logger.info("Applying population AF filter to %s", input_h5ad)
    adata = ad.read_h5ad(input_h5ad)
    adata.var_names = adata.var_names.astype(str)

    detail_df, malformed_h5ad_snv = _build_af_filter_detail_table(
        var_names=adata.var_names,
        af_lookup=af_lookup,
        af_threshold=af_threshold,
        strict_snv_format=strict_snv_format,
    )
    remove_mask = detail_df["remove_af_gt_threshold"].to_numpy(dtype=bool)
    keep_mask = ~remove_mask

    adata.var[af_field] = detail_df["annotated_af"].to_numpy(dtype=float)
    adata.var[f"{af_field}_filter_pass"] = keep_mask

    total_h5ad_snvs = int(adata.n_vars)
    removed_h5ad_snvs = int(remove_mask.sum())
    kept_h5ad_snvs = int(keep_mask.sum())
    matched_h5ad_snvs = int(detail_df["vcf_record_found"].sum())
    unmatched_h5ad_snvs = int(total_h5ad_snvs - matched_h5ad_snvs)
    numeric_af_h5ad_snvs = int(detail_df["af_value_is_numeric"].sum())

    af_filter_metadata = {
        "vcf_path": str(vcf_label),
        "af_field": str(af_field),
        "max_af": float(af_threshold),
        "snvs_before": total_h5ad_snvs,
        "snvs_matched_to_vcf": matched_h5ad_snvs,
        "snvs_unmatched_to_vcf": unmatched_h5ad_snvs,
        "snvs_with_numeric_af": numeric_af_h5ad_snvs,
        "snvs_removed_af_gt_threshold": removed_h5ad_snvs,
        "snvs_kept": kept_h5ad_snvs,
        "malformed_h5ad_snv_ids": int(malformed_h5ad_snv),
    }
    af_filter_metadata.update(vcf_stats)

    filtered = adata[:, keep_mask].copy()
    filtered.uns["population_af_filter"] = af_filter_metadata

    output_dir = os.path.dirname(output_h5ad)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filtered.write_h5ad(output_h5ad)
    logger.info("Saved AF-filtered h5ad to %s", output_h5ad)

    summary_df = pd.DataFrame(
        [
            {
                "vcf_path": str(vcf_label),
                "input_h5ad": str(input_h5ad),
                "output_h5ad": str(output_h5ad),
                "af_field": str(af_field),
                "max_af": float(af_threshold),
                "cells": int(adata.n_obs),
                **af_filter_metadata,
            }
        ]
    )
    summary_path = os.path.splitext(output_h5ad)[0] + ".af_filter_summary.tsv"
    detail_path = os.path.splitext(output_h5ad)[0] + ".af_filter_details.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    detail_df.to_csv(detail_path, sep="\t", index=False)
    logger.info(
        "Saved AF filter details to %s and %s",
        summary_path,
        detail_path,
    )

    return output_h5ad


def _compute_variant_prior(
    snv_key,
    sample_name,
    snv_freq_by_sample,
    default_prior,
    prior_scale,
    prior_cap,
):
    prior = float(default_prior)
    parsed_key = _parse_snv_key(snv_key)
    if parsed_key and snv_freq_by_sample and sample_name:
        freq_map = snv_freq_by_sample.get(parsed_key, {})
        freq = freq_map.get(sample_name, np.nan)
        if not np.isnan(freq):
            prior = float(prior_scale) * float(freq)

    prior = max(float(default_prior), prior)
    prior = min(float(prior_cap), prior)
    prior = min(max(prior, 1e-9), 1.0 - 1e-9)
    return prior


def _format_barcode_counts(counts_dict):
    parts = []
    for barcode, count in sorted(counts_dict.items()):
        count_int = int(count)
        if count_int > 0:
            parts.append(f"{barcode}:{count_int}")
    return ",".join(parts)


def _parse_barcode_counts(counts_text):
    parsed = {}
    if not counts_text:
        return parsed
    for token in counts_text.split(","):
        if not token or ":" not in token:
            continue
        barcode, count_text = token.rsplit(":", 1)
        if not barcode:
            continue
        try:
            count_int = int(count_text)
        except ValueError:
            continue
        if count_int > 0:
            parsed[barcode] = count_int
    return parsed


def _collapse_counts_by_unit(read_counts, umi_sets, count_unit):
    if count_unit == "reads":
        return {barcode: int(count) for barcode, count in read_counts.items() if int(count) > 0}

    collapsed = {}
    all_barcodes = set(read_counts.keys()) | set(umi_sets.keys())
    for barcode in all_barcodes:
        umi_count = len(umi_sets.get(barcode, set()))
        if umi_count > 0:
            collapsed[barcode] = int(umi_count)
        else:
            read_count = int(read_counts.get(barcode, 0))
            if read_count > 0:
                collapsed[barcode] = read_count
    return collapsed


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.size == 0 or values.size != weights.size:
        return np.nan

    positive = weights > 0
    if not np.any(positive):
        return np.nan

    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * float(weights.sum())
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _estimate_sample_p_alt_if_variant(
    snv_alt_counts,
    snv_ref_counts,
    default_p,
    *,
    enabled=False,
    min_alt_count=2,
    min_total_count=2,
    min_observations=25,
    min_snvs=5,
):
    fallback_value = float(default_p)
    metadata = {
        "enabled": bool(enabled),
        "source": "config_default",
        "effective_p_alt_if_variant": fallback_value,
        "configured_p_alt_if_variant": fallback_value,
        "fallback_reason": None,
        "min_alt_count": int(min_alt_count),
        "min_total_count": int(min_total_count),
        "min_observations": int(min_observations),
        "min_snvs": int(min_snvs),
        "n_selected_observations": 0,
        "n_selected_snvs": 0,
    }
    if not enabled:
        metadata["fallback_reason"] = "per_sample_estimation_disabled"
        return fallback_value, metadata

    snv_ratios = []
    snv_weights = []
    n_selected_observations = 0
    all_snvs = sorted(set(snv_alt_counts.keys()) | set(snv_ref_counts.keys()))
    for snv in all_snvs:
        alt_map = snv_alt_counts.get(snv, {})
        ref_map = snv_ref_counts.get(snv, {})
        snv_alt_total = 0
        snv_total = 0
        snv_observations = 0
        for barcode, alt_count_raw in alt_map.items():
            alt_count = int(alt_count_raw)
            if alt_count < int(min_alt_count):
                continue
            ref_count = int(ref_map.get(barcode, 0))
            total_count = alt_count + ref_count
            if total_count < int(min_total_count):
                continue
            snv_alt_total += alt_count
            snv_total += total_count
            snv_observations += 1

        if snv_total > 0 and snv_observations > 0:
            snv_ratios.append(float(snv_alt_total / snv_total))
            snv_weights.append(float(snv_total))
            n_selected_observations += snv_observations

    metadata["n_selected_observations"] = int(n_selected_observations)
    metadata["n_selected_snvs"] = int(len(snv_ratios))
    if n_selected_observations < int(min_observations):
        metadata["fallback_reason"] = "insufficient_selected_observations"
        return fallback_value, metadata
    if len(snv_ratios) < int(min_snvs):
        metadata["fallback_reason"] = "insufficient_selected_snvs"
        return fallback_value, metadata

    estimated_p = _weighted_median(snv_ratios, snv_weights)
    if not np.isfinite(estimated_p):
        metadata["fallback_reason"] = "non_finite_estimate"
        return fallback_value, metadata

    estimated_p = min(max(float(estimated_p), 1e-9), 1.0 - 1e-9)
    metadata["source"] = "estimated_per_sample"
    metadata["effective_p_alt_if_variant"] = estimated_p
    metadata["fallback_reason"] = None
    return estimated_p, metadata


def _save_neg1_posterior_outputs(probabilities, plot_file, summary_file, bins=60):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    summary_dir = os.path.dirname(summary_file)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    plot_dir = os.path.dirname(plot_file)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    if probabilities.size == 0:
        summary_df = pd.DataFrame(
            [
                {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q01": np.nan,
                    "q05": np.nan,
                    "q10": np.nan,
                    "q25": np.nan,
                    "q50": np.nan,
                    "q75": np.nan,
                    "q90": np.nan,
                    "q95": np.nan,
                    "q99": np.nan,
                    "max": np.nan,
                }
            ]
        )
        summary_df.to_csv(summary_file, sep="\t", index=False)
        logger.warning("No -1 posterior probabilities were available; skipped histogram %s", plot_file)
        return

    bins = max(5, int(bins))
    plt.figure(figsize=(8, 5))
    plt.hist(probabilities, bins=bins, range=(0.0, 1.0), color="#4C78A8", edgecolor="white")
    plt.xlabel("Posterior P(ref | ALT=0, REF=r)")
    plt.ylabel("Count of -1 entries")
    plt.title("Distribution of ref posterior for matrix entries initially labelled -1")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    quantiles = np.quantile(probabilities, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    summary_df = pd.DataFrame(
        [
            {
                "count": int(probabilities.size),
                "mean": float(probabilities.mean()),
                "std": float(probabilities.std()),
                "min": float(probabilities.min()),
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q10": float(quantiles[2]),
                "q25": float(quantiles[3]),
                "q50": float(quantiles[4]),
                "q75": float(quantiles[5]),
                "q90": float(quantiles[6]),
                "q95": float(quantiles[7]),
                "q99": float(quantiles[8]),
                "max": float(probabilities.max()),
            }
        ]
    )
    summary_df.to_csv(summary_file, sep="\t", index=False)


def _extract_neg1_probabilities_from_adata(adata_obj):
    if "ref_posterior" not in adata_obj.layers:
        return np.array([], dtype=np.float32)

    matrix = adata_obj.X
    posterior = adata_obj.layers["ref_posterior"]

    if not hasattr(matrix, "nnz") and not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)

    if hasattr(matrix, "tocoo"):
        x_csr = matrix.tocsr()
        if x_csr.nnz == 0:
            return np.array([], dtype=np.float32)
        neg_data_idx = np.flatnonzero(x_csr.data == -1)
        if neg_data_idx.size == 0:
            return np.array([], dtype=np.float32)
        rows = np.searchsorted(x_csr.indptr, neg_data_idx, side="right") - 1
        cols = x_csr.indices[neg_data_idx]

        if hasattr(posterior, "tocsr"):
            posterior_csr = posterior.tocsr().astype(np.float32)
            values = np.zeros(rows.shape[0], dtype=np.float32)
            for row in np.unique(rows):
                mask = rows == row
                row_cols = cols[mask]
                start = posterior_csr.indptr[row]
                end = posterior_csr.indptr[row + 1]
                if start == end:
                    continue
                idx = np.searchsorted(posterior_csr.indices[start:end], row_cols)
                valid = (idx >= 0) & (idx < (end - start))
                if not np.any(valid):
                    continue
                idx_valid = idx[valid] + start
                col_valid = row_cols[valid]
                matched = posterior_csr.indices[idx_valid] == col_valid
                if not np.any(matched):
                    continue
                final_positions = np.flatnonzero(mask)[valid][matched]
                values[final_positions] = posterior_csr.data[idx_valid[matched]].astype(np.float32)
        else:
            posterior_dense = np.asarray(posterior)
            values = posterior_dense[rows, cols]
    else:
        dense_matrix = np.asarray(matrix)
        neg_mask = dense_matrix == -1
        if not np.any(neg_mask):
            return np.array([], dtype=np.float32)
        values = np.asarray(posterior)[neg_mask]

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    return values


def process_snv_chunk(args):
    """
    For a chunk of SNV list, use pileup from BAM to find:
      - CBs supporting alt alleles (mut)
      - CBs supporting ref alleles (no_var)
    """
    bam_path, snv_chunk, count_unit, base_quality_min, mapping_quality_min = args
    logger.debug(
        "Worker processing %d SNVs from BAM file %s", len(snv_chunk), bam_path
    )
    bamfile = pysam.AlignmentFile(bam_path, "rb")

    chunk_results = []
    chunk_no_snv_results = []
    chunk_alt_count_results = []
    chunk_ref_count_results = []

    for snv in snv_chunk:
        # Support both VCF format (>=5 columns) and TSV format (4 columns)
        if len(snv) >= 5:
            chrom, pos, ref, alt = snv[0], int(snv[1]), snv[3], snv[4]
        elif len(snv) >= 4:
            chrom, pos, ref, alt = snv[0], int(snv[1]), snv[2], snv[3]
        else:
            logger.warning("Skipping malformed SNV record: %s", snv)
            continue

        alt_read_counts = defaultdict(int)
        ref_read_counts = defaultdict(int)
        alt_umi_sets = defaultdict(set)
        ref_umi_sets = defaultdict(set)

        pileup_column = bamfile.pileup(chrom, pos - 1, pos)
        for pileup in pileup_column:
            if pileup.pos == pos - 1:
                if pileup.n > 50000:
                    logger.info(
                        f"WARNING: Skipping high-depth site {chrom}:{pos} with depth {pileup.n}"
                    )
                    continue
                for pileup_read in pileup.pileups:
                    if pileup_read.alignment and pileup_read.query_position is not None:
                        query_position = int(pileup_read.query_position)
                        read_base = pileup_read.alignment.query_sequence[query_position]
                        query_qualities = pileup_read.alignment.query_qualities
                        if query_qualities is None:
                            raise ValueError(
                                "BAM read is missing base quality values, so "
                                "base-quality filtering cannot be applied. "
                                f"bam={bam_path}, snv={chrom}:{pos} {ref}>{alt}, "
                                f"read={pileup_read.alignment.query_name}. "
                                "Please provide a BAM with QUAL values or disable/remove "
                                "base-quality filtering explicitly before running this step."
                            )
                        read_quality = query_qualities[query_position]
                        if (
                            read_quality < int(base_quality_min)
                            or pileup_read.alignment.mapping_quality < int(mapping_quality_min)
                        ):
                            continue

                        if pileup_read.alignment.has_tag("CB"):
                            cb_tag = pileup_read.alignment.get_tag("CB")
                            ub_tag = None
                            if pileup_read.alignment.has_tag("UB"):
                                ub_tag = pileup_read.alignment.get_tag("UB")
                            if read_base == alt:
                                alt_read_counts[cb_tag] += 1
                                if ub_tag is not None:
                                    alt_umi_sets[cb_tag].add(ub_tag)
                            elif read_base == ref:
                                ref_read_counts[cb_tag] += 1
                                if ub_tag is not None:
                                    ref_umi_sets[cb_tag].add(ub_tag)

        alt_counts = _collapse_counts_by_unit(alt_read_counts, alt_umi_sets, count_unit)
        ref_counts = _collapse_counts_by_unit(ref_read_counts, ref_umi_sets, count_unit)

        supporting_barcodes = sorted(alt_counts.keys())
        no_var_barcodes = sorted(ref_counts.keys())

        chunk_results.append(
            f"{chrom}\t{pos}\t{ref}\t{alt}\t{','.join(supporting_barcodes)}"
        )
        chunk_no_snv_results.append(
            f"{chrom}\t{pos}\t{ref}\t{alt}\t{','.join(no_var_barcodes)}"
        )
        chunk_alt_count_results.append(
            f"{chrom}\t{pos}\t{ref}\t{alt}\t{_format_barcode_counts(alt_counts)}"
        )
        chunk_ref_count_results.append(
            f"{chrom}\t{pos}\t{ref}\t{alt}\t{_format_barcode_counts(ref_counts)}"
        )

    bamfile.close()
    logger.debug("Worker finished processing %d SNVs", len(snv_chunk))
    return chunk_results, chunk_no_snv_results, chunk_alt_count_results, chunk_ref_count_results


def get_reads_supporting_snv_parallel(
    bam_file,
    snv_positions,
    output_file,
    no_snv_output_file,
    num_processes=4,
    alt_count_output_file=None,
    ref_count_output_file=None,
    count_unit="reads",
    base_quality_min=20,
    mapping_quality_min=20,
):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    else:
        num_processes = _coerce_positive_int(num_processes, "threads")

    if count_unit not in {"reads", "umis"}:
        raise ValueError("count_unit must be either 'reads' or 'umis'.")
    if (alt_count_output_file is None) != (ref_count_output_file is None):
        raise ValueError("alt_count_output_file and ref_count_output_file must be provided together.")

    logger.info(
        (
            "Starting parallel SNV support extraction: %d SNVs, %d processes "
            "(baseQ>=%d, mapQ>=%d)"
        ),
        len(snv_positions),
        num_processes,
        int(base_quality_min),
        int(mapping_quality_min),
    )

    target_chunks = max(num_processes * 20, 1)
    chunk_size = max(1, math.ceil(len(snv_positions) / target_chunks))
    snv_chunks = [
        snv_positions[i : i + chunk_size]
        for i in range(0, len(snv_positions), chunk_size)
    ]
    args_list = [
        (bam_file, chunk, count_unit, int(base_quality_min), int(mapping_quality_min))
        for chunk in snv_chunks
    ]

    if alt_count_output_file is None:
        with Pool(processes=num_processes) as pool, open(output_file, "w") as out, open(
            no_snv_output_file, "w"
        ) as no_snv_out:
            logger.info("Dispatching %d SNV chunks to worker pool", len(args_list))
            for supporting, no_var, _, _ in tqdm(
                pool.imap_unordered(process_snv_chunk, args_list, chunksize=1),
                total=len(args_list),
                desc="Parallel Processing",
            ):
                for line in supporting:
                    out.write(line + "\n")
                for line in no_var:
                    no_snv_out.write(line + "\n")
        logger.info(
            "Finished parallel SNV support extraction. Results saved to %s and %s",
            output_file,
            no_snv_output_file,
        )
        return

    assert alt_count_output_file is not None and ref_count_output_file is not None

    with (
        Pool(processes=num_processes) as pool,
        open(output_file, "w") as out,
        open(no_snv_output_file, "w") as no_snv_out,
        open(alt_count_output_file, "w") as alt_count_out,
        open(ref_count_output_file, "w") as ref_count_out,
    ):
        logger.info("Dispatching %d SNV chunks to worker pool", len(args_list))
        for supporting, no_var, alt_counts, ref_counts in tqdm(
            pool.imap_unordered(process_snv_chunk, args_list, chunksize=1),
            total=len(args_list),
            desc="Parallel Processing",
        ):
            for line in supporting:
                out.write(line + "\n")
            for line in no_var:
                no_snv_out.write(line + "\n")
            for line in alt_counts:
                alt_count_out.write(line + "\n")
            for line in ref_counts:
                ref_count_out.write(line + "\n")

    logger.info(
        "Finished parallel SNV support extraction. Results saved to %s, %s, %s and %s",
        output_file,
        no_snv_output_file,
        alt_count_output_file,
        ref_count_output_file,
    )


def read_cb_list_spatial(cb_file):
    df = pd.read_csv(cb_file, header=None)
    df.columns = ["barcode", "in_tissue", "x", "y", "px", "py"]
    return list(df[df["in_tissue"] == 1]["barcode"])


def read_cb_df_spatial(cb_file):
    df = pd.read_csv(cb_file, header=None)
    df.columns = ["barcode", "in_tissue", "x", "y", "px", "py"]
    return df[df["in_tissue"] == 1]


def read_cb_list_sc(cb_file):
    df = pd.read_csv(cb_file, header=None)
    df.columns = ["barcode"]
    return list(df["barcode"])


def count_snvs_per_cb(snv_file, cb_list, output_file):
    logger.info(
        "Counting SNVs per barcode from %s for %d barcodes",
        snv_file,
        len(cb_list),
    )
    snv_counts = {cb: {"count": 0, "snvs": []} for cb in cb_list}
    with open(snv_file, "r") as f:
        for line in tqdm(f, desc="counting snv supported barcodes..."):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, ref, alt, cb_tags = parts
            cb_tags = cb_tags.split(",")

            for cb in cb_tags:
                if cb in snv_counts:
                    snv_counts[cb]["count"] += 1
                    snv_counts[cb]["snvs"].append(f"{chrom}:{pos}({ref}->{alt})")

    with open(output_file, "w") as out:
        for cb, info in snv_counts.items():
            snv_list = ",".join(info["snvs"])
            out.write(f"{cb}\t{info['count']}\t{snv_list}\n")
    logger.info("Finished counting SNVs per barcode. Output saved to %s", output_file)


_GLOBAL_SNV_SUPPORT = None
_GLOBAL_SNV_NO_VAR = None
_GLOBAL_SNV_ALT_COUNTS = None
_GLOBAL_SNV_REF_COUNTS = None
_GLOBAL_SNV_PRIORS = None
_GLOBAL_BARCODE_TO_INDEX = None
_GLOBAL_N_BARCODES = None
_GLOBAL_P_ALT_IF_VARIANT = None
_GLOBAL_ALT_ERROR_RATE = None
_GLOBAL_NEG1_PROB_THRESHOLD = None


def _read_snv_support_worker(file_path, cb_set, queue):
    logger.debug("Reading SNV support data from %s", file_path)
    support = defaultdict(dict)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            chrom, pos, ref, alt, barcodes = parts
            snv = f"{chrom}:{pos}_{ref}>{alt}"
            for barcode in barcodes.split(","):
                if not barcode:
                    continue
                if barcode in cb_set:
                    support[snv][barcode] = 1
    queue.put({snv: dict(barcodes) for snv, barcodes in support.items()})


def _read_snv_no_var_worker(file_path, cb_set, queue):
    logger.debug("Reading SNV reference-support data from %s", file_path)
    no_var = defaultdict(set)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            chrom, pos, ref, alt, barcodes = parts
            snv = f"{chrom}:{pos}_{ref}>{alt}"
            for barcode in barcodes.split(","):
                if not barcode:
                    continue
                if barcode in cb_set:
                    no_var[snv].add(barcode)
    queue.put({snv: set(barcodes) for snv, barcodes in no_var.items()})


def _read_snv_count_worker(file_path, cb_set, queue):
    logger.debug("Reading SNV count data from %s", file_path)
    counts = defaultdict(dict)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            chrom, pos, ref, alt, counts_text = parts
            snv = f"{chrom}:{pos}_{ref}>{alt}"
            parsed = _parse_barcode_counts(counts_text)
            if not parsed:
                continue
            for barcode, count in parsed.items():
                if barcode in cb_set and int(count) > 0:
                    counts[snv][barcode] = int(count)
    queue.put({snv: dict(barcodes) for snv, barcodes in counts.items()})


def _compute_ref_posterior(prior, ref_count, p_alt_if_variant, alt_error_rate):
    prior = min(max(float(prior), 1e-9), 1.0 - 1e-9)
    ref_count = max(1, int(ref_count))
    p_alt_if_variant = min(max(float(p_alt_if_variant), 1e-9), 1.0 - 1e-9)
    alt_error_rate = min(max(float(alt_error_rate), 1e-9), 1.0 - 1e-9)

    numerator = (1.0 - prior) * ((1.0 - alt_error_rate) ** ref_count)
    denominator = numerator + prior * ((1.0 - p_alt_if_variant) ** ref_count)
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _init_partial_matrix_worker(
    snv_support,
    snv_no_var,
    snv_alt_counts,
    snv_ref_counts,
    snv_priors,
    barcode_to_index,
    n_barcodes,
    p_alt_if_variant,
    alt_error_rate,
    neg1_prob_threshold,
):
    global _GLOBAL_SNV_SUPPORT, _GLOBAL_SNV_NO_VAR
    global _GLOBAL_SNV_ALT_COUNTS, _GLOBAL_SNV_REF_COUNTS, _GLOBAL_SNV_PRIORS
    global _GLOBAL_BARCODE_TO_INDEX, _GLOBAL_N_BARCODES
    global _GLOBAL_P_ALT_IF_VARIANT, _GLOBAL_ALT_ERROR_RATE, _GLOBAL_NEG1_PROB_THRESHOLD
    _GLOBAL_SNV_SUPPORT = snv_support
    _GLOBAL_SNV_NO_VAR = snv_no_var
    _GLOBAL_SNV_ALT_COUNTS = snv_alt_counts
    _GLOBAL_SNV_REF_COUNTS = snv_ref_counts
    _GLOBAL_SNV_PRIORS = snv_priors
    _GLOBAL_BARCODE_TO_INDEX = barcode_to_index
    _GLOBAL_N_BARCODES = n_barcodes
    _GLOBAL_P_ALT_IF_VARIANT = p_alt_if_variant
    _GLOBAL_ALT_ERROR_RATE = alt_error_rate
    _GLOBAL_NEG1_PROB_THRESHOLD = neg1_prob_threshold
    logger.debug("Initialized partial matrix worker with %d barcodes", n_barcodes)


def _build_partial_matrix(snv_list):
    support_global = _GLOBAL_SNV_SUPPORT or {}
    no_var_global = _GLOBAL_SNV_NO_VAR or {}
    alt_count_global = _GLOBAL_SNV_ALT_COUNTS or {}
    ref_count_global = _GLOBAL_SNV_REF_COUNTS or {}
    prior_global = _GLOBAL_SNV_PRIORS or {}
    barcode_to_index = _GLOBAL_BARCODE_TO_INDEX or {}
    n_barcodes = _GLOBAL_N_BARCODES or 0

    part_matrix = lil_matrix((n_barcodes, len(snv_list)), dtype=np.int8)
    part_alt_count = lil_matrix((n_barcodes, len(snv_list)), dtype=np.int32)
    part_ref_count = lil_matrix((n_barcodes, len(snv_list)), dtype=np.int32)
    part_ref_posterior = lil_matrix((n_barcodes, len(snv_list)), dtype=np.float32)

    for local_j, snv in enumerate(snv_list):
        support_map = support_global.get(snv, {})
        no_var_map = no_var_global.get(snv, {})
        alt_count_map = alt_count_global.get(snv, {})
        ref_count_map = ref_count_global.get(snv, {})
        prior = prior_global.get(snv, 1e-3)

        for barcode in support_map:
            if barcode in barcode_to_index:
                i = barcode_to_index[barcode]
                part_matrix[i, local_j] = 1
                alt_count = int(alt_count_map.get(barcode, 0))
                ref_count = int(ref_count_map.get(barcode, 0))
                if alt_count > 0:
                    part_alt_count[i, local_j] = alt_count
                if ref_count > 0:
                    part_ref_count[i, local_j] = ref_count

        for barcode in no_var_map:
            if (
                barcode in barcode_to_index
                and barcode not in support_map
            ):
                i = barcode_to_index[barcode]
                ref_count = int(ref_count_map.get(barcode, 0))
                if ref_count <= 0:
                    ref_count = 1
                part_ref_count[i, local_j] = ref_count

                ref_posterior = _compute_ref_posterior(
                    prior=prior,
                    ref_count=ref_count,
                    p_alt_if_variant=_GLOBAL_P_ALT_IF_VARIANT,
                    alt_error_rate=_GLOBAL_ALT_ERROR_RATE,
                )
                if not np.isnan(ref_posterior):
                    part_ref_posterior[i, local_j] = ref_posterior

                if part_matrix[i, local_j] == 0:
                    if (
                        _GLOBAL_NEG1_PROB_THRESHOLD is None
                        or np.isnan(ref_posterior)
                        or ref_posterior >= _GLOBAL_NEG1_PROB_THRESHOLD
                    ):
                        part_matrix[i, local_j] = -1
    logger.debug("Built partial matrix for %d SNVs across %d barcodes", len(snv_list), n_barcodes)
    return (
        snv_list,
        part_matrix.tocsr(),
        part_alt_count.tocsr(),
        part_ref_count.tocsr(),
        part_ref_posterior.tocsr(),
    )


def generate_barcode_snv_matrix(
    snv_output_file,
    no_snv_output_file,
    alt_count_file,
    ref_count_file,
    barcodes_file,
    matrix_file,
    data_type,
    sample_name=None,
    snv_freq_by_sample=None,
    threads=4,
    p_alt_if_variant=0.5,
    alt_error_rate=0.001,
    default_prior=0.001,
    prior_scale=2.0,
    prior_cap=0.95,
    neg1_prob_threshold=None,
    neg1_prob_plot_file=None,
    neg1_prob_summary_file=None,
    neg1_prob_hist_bins=60,
    estimate_p_per_sample=False,
    p_estimation_min_alt_count=2,
    p_estimation_min_total_count=2,
    p_estimation_min_observations=25,
    p_estimation_min_snvs=5,
):
    threads = _coerce_positive_int(threads, "threads")
    logger.info(
        "Building barcode-SNV matrix from %s and %s using %s barcodes file",
        snv_output_file,
        no_snv_output_file,
        barcodes_file,
    )
    if data_type == "spatial":
        cb_list = read_cb_list_spatial(barcodes_file)
    elif data_type == "sc":
        cb_list = read_cb_list_sc(barcodes_file)
    else:
        raise ValueError("Invalid data_type specified in config. Must be 'spatial' or 'sc'.")

    cb_set = set(cb_list)
    logger.info("Loaded %d valid barcodes", len(cb_list))

    support_queue = Queue()
    no_var_queue = Queue()
    alt_count_queue = Queue()
    ref_count_queue = Queue()

    support_process = Process(
        target=_read_snv_support_worker,
        args=(snv_output_file, cb_set, support_queue),
    )
    no_var_process = Process(
        target=_read_snv_no_var_worker,
        args=(no_snv_output_file, cb_set, no_var_queue),
    )
    alt_count_process = Process(
        target=_read_snv_count_worker,
        args=(alt_count_file, cb_set, alt_count_queue),
    )
    ref_count_process = Process(
        target=_read_snv_count_worker,
        args=(ref_count_file, cb_set, ref_count_queue),
    )

    support_process.start()
    no_var_process.start()
    alt_count_process.start()
    ref_count_process.start()

    snv_support_raw = support_queue.get()
    snv_no_var_candidates = no_var_queue.get()
    snv_alt_counts_raw = alt_count_queue.get()
    snv_ref_counts_raw = ref_count_queue.get()
    logger.debug(
        "Collected support=%d, no_var=%d, alt_counts=%d, ref_counts=%d entries",
        len(snv_support_raw),
        len(snv_no_var_candidates),
        len(snv_alt_counts_raw),
        len(snv_ref_counts_raw),
    )

    support_process.join()
    no_var_process.join()
    alt_count_process.join()
    ref_count_process.join()

    snv_support = defaultdict(dict, snv_support_raw)
    snv_alt_counts = defaultdict(dict, snv_alt_counts_raw)
    snv_ref_counts = defaultdict(dict, snv_ref_counts_raw)
    snv_no_var = defaultdict(dict)
    for snv, barcodes in snv_no_var_candidates.items():
        support_barcodes = snv_support.get(snv, {})
        filtered_barcodes = {
            barcode: -1 for barcode in barcodes if barcode not in support_barcodes
        }
        if filtered_barcodes:
            snv_no_var[snv] = filtered_barcodes

    effective_p_alt_if_variant, p_estimation_metadata = _estimate_sample_p_alt_if_variant(
        snv_alt_counts,
        snv_ref_counts,
        p_alt_if_variant,
        enabled=estimate_p_per_sample,
        min_alt_count=p_estimation_min_alt_count,
        min_total_count=p_estimation_min_total_count,
        min_observations=p_estimation_min_observations,
        min_snvs=p_estimation_min_snvs,
    )
    logger.info(
        "Using p_alt_if_variant=%.6f for sample %s (%s)",
        effective_p_alt_if_variant,
        sample_name if sample_name is not None else "<unknown>",
        p_estimation_metadata["source"],
    )
    if p_estimation_metadata.get("fallback_reason"):
        logger.info(
            "Per-sample p estimation fallback for sample %s: %s",
            sample_name if sample_name is not None else "<unknown>",
            p_estimation_metadata["fallback_reason"],
        )

    all_snvs = sorted(
        set(snv_support.keys())
        | set(snv_no_var.keys())
        | set(snv_alt_counts.keys())
        | set(snv_ref_counts.keys())
    )
    barcode_to_index = {barcode: i for i, barcode in enumerate(cb_list)}
    logger.info("Total SNVs considered for this sample: %d", len(all_snvs))
    logger.info("Total barcodes in matrix: %d", len(cb_list))

    snv_priors = {
        snv: _compute_variant_prior(
            snv_key=snv,
            sample_name=sample_name,
            snv_freq_by_sample=snv_freq_by_sample,
            default_prior=default_prior,
            prior_scale=prior_scale,
            prior_cap=prior_cap,
        )
        for snv in all_snvs
    }

    n_barcodes = len(cb_list)
    if len(all_snvs) == 0:
        logger.warning("No SNVs found for this sample, creating empty matrix.")
        merged_matrix = lil_matrix((n_barcodes, 0), dtype=np.int8).tocsr()
        merged_alt_count = lil_matrix((n_barcodes, 0), dtype=np.int32).tocsr()
        merged_ref_count = lil_matrix((n_barcodes, 0), dtype=np.int32).tocsr()
        merged_ref_posterior = lil_matrix((n_barcodes, 0), dtype=np.float32).tocsr()
        merged_columns = np.array([], dtype=str)
    else:
        num_chunks = min(threads, len(all_snvs))
        snv_chunks = [
            chunk.tolist()
            for chunk in np.array_split(all_snvs, num_chunks)
            if len(chunk) > 0
        ]
        logger.info("Splitting SNVs into %d chunks for matrix construction", len(snv_chunks))

        with Pool(
            processes=threads,
            initializer=_init_partial_matrix_worker,
            initargs=(
                snv_support,
                snv_no_var,
                snv_alt_counts,
                snv_ref_counts,
                snv_priors,
                barcode_to_index,
                n_barcodes,
                effective_p_alt_if_variant,
                alt_error_rate,
                neg1_prob_threshold,
            ),
        ) as pool:
            results = pool.map(_build_partial_matrix, snv_chunks)

        partial_matrices = [m for _, m, _, _, _ in results]
        partial_alt_counts = [m for _, _, m, _, _ in results]
        partial_ref_counts = [m for _, _, _, m, _ in results]
        partial_ref_posteriors = [m for _, _, _, _, m in results]
        merged_matrix = hstack(partial_matrices).tocsr()
        merged_alt_count = hstack(partial_alt_counts).tocsr()
        merged_ref_count = hstack(partial_ref_counts).tocsr()
        merged_ref_posterior = hstack(partial_ref_posteriors).tocsr()
        merged_columns = np.concatenate([chunk for chunk, *_ in results])

    adata = ad.AnnData(
        X=merged_matrix,
        obs=pd.DataFrame(index=cb_list),
        var=pd.DataFrame(index=merged_columns),
    )
    adata.layers["alt_count"] = merged_alt_count
    adata.layers["ref_count"] = merged_ref_count
    adata.layers["ref_posterior"] = merged_ref_posterior

    freq_values = []
    for snv_key in merged_columns:
        if snv_freq_by_sample and sample_name:
            if isinstance(snv_key, tuple):
                key = snv_key
            else:
                parts = snv_key.split(":")
                key = None
                if len(parts) == 2:
                    chrom = parts[0]
                    rest = parts[1].split("_")
                    if len(rest) == 2:
                        pos = rest[0]
                        ref_alt = rest[1].split(">")
                        if len(ref_alt) == 2:
                            key = (chrom, pos, ref_alt[0], ref_alt[1])

            if key and key in snv_freq_by_sample:
                freq = snv_freq_by_sample[key].get(sample_name, np.nan)
            else:
                freq = np.nan
        else:
            freq = np.nan
        freq_values.append(freq)
    adata.var["FREQ"] = freq_values
    adata.var["variant_prior"] = [float(snv_priors.get(snv, default_prior)) for snv in merged_columns]
    adata.uns["ref_posterior_params"] = {
        "p_alt_if_variant": float(effective_p_alt_if_variant),
        "alt_error_rate": float(alt_error_rate),
        "default_prior": float(default_prior),
        "prior_scale": float(prior_scale),
        "prior_cap": float(prior_cap),
        "neg1_prob_threshold": None if neg1_prob_threshold is None else float(neg1_prob_threshold),
    }
    adata.uns["p_alt_if_variant_estimation"] = p_estimation_metadata
    valid_count = len([f for f in freq_values if not np.isnan(f)])
    logger.info("Added FREQ column to adata.var with %d valid values", valid_count)

    neg1_probabilities = []
    if merged_matrix.nnz > 0 and merged_ref_posterior.nnz > 0:
        x_coo = merged_matrix.tocoo()
        neg_mask = x_coo.data == -1
        if np.any(neg_mask):
            rows = x_coo.row[neg_mask]
            cols = x_coo.col[neg_mask]
            sampled = np.asarray(merged_ref_posterior[rows, cols]).reshape(-1)
            sampled = sampled[np.isfinite(sampled)]
            sampled = sampled[sampled > 0]
            neg1_probabilities = sampled.tolist()

    if neg1_prob_plot_file and neg1_prob_summary_file:
        _save_neg1_posterior_outputs(
            neg1_probabilities,
            neg1_prob_plot_file,
            neg1_prob_summary_file,
            bins=neg1_prob_hist_bins,
        )

    adata.write_h5ad(matrix_file)
    logger.info("Matrix saved to %s", matrix_file)


def extract_frequency_from_vcf_line(columns):
    """
    Extract allele frequency from VCF data line.

    Priority:
    1. FREQ field in sample (VarScan format, column 9, 7th field after ':')
    2. Calculate from AD/DP in sample

    Returns:
        float or None: allele frequency (0-1 scale)
    """
    if len(columns) < 10:
        return None

    sample_field = columns[9]
    format_field = columns[8]

    format_keys = format_field.split(':')
    sample_values = sample_field.split(':')
    sample_dict = dict(zip(format_keys, sample_values))

    if 'FREQ' in sample_dict:
        freq_str = sample_dict['FREQ'].rstrip('%')
        if freq_str and freq_str != '.':
            try:
                return float(freq_str) / 100.0
            except ValueError:
                pass

    if 'AD' in sample_dict and 'DP' in sample_dict:
        ad_str = sample_dict['AD']
        dp_str = sample_dict['DP']

        if not ad_str or ad_str == '.' or not dp_str or dp_str == '.':
            return None

        try:
            dp = int(dp_str)
            if dp <= 0:
                return None

            if ',' in ad_str:
                ad_values = ad_str.split(',')
                ad = 0
                for val in ad_values[1:]:
                    if val and val != '.':
                        ad += int(val)
            else:
                ad = int(ad_str)

            return ad / dp
        except (ValueError, ZeroDivisionError, IndexError):
            pass

    return None


def filter_vcf(vcf_file, output_vcf_file, percentage):
    """
    Filter VCF by allele frequency.
    Exactly the same as the original: only retain SNVs with freq < percentage.
    """
    input_file = vcf_file
    output_file = output_vcf_file

    logger.info(
        "Filtering VCF %s for variants with frequency below %.2f%%",
        vcf_file,
        percentage,
    )

    if percentage is None or percentage <= 0:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                if not line.strip():
                    continue
                outfile.write(line)
        logger.info("Filtered VCF written to %s", output_file)
        return

    kept_count = 0
    dropped_count = 0
    unparsable_count = 0
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            if line.startswith("#"):
                outfile.write(line)
                continue

            columns = line.strip().split("\t")
            if len(columns) < 10:
                dropped_count += 1
                continue

            freq = extract_frequency_from_vcf_line(columns)
            if freq is None or not np.isfinite(freq):
                unparsable_count += 1
                dropped_count += 1
                logger.warning(
                    "Could not parse allele frequency from line; dropping variant: %s",
                    line.strip(),
                )
                continue

            if float(freq) * 100.0 < float(percentage):
                outfile.write(line)
                kept_count += 1
            else:
                dropped_count += 1

    logger.info(
        "Filtered VCF written to %s (kept=%d, dropped=%d, unparsable=%d)",
        output_file,
        kept_count,
        dropped_count,
        unparsable_count,
    )


def build_snv_union_tsv(filtered_vcfs, union_tsv_path, sample_names):
    """
    Extract unique union of chr/pos/ref/alt from multiple samples' filtered VCFs,
    and write to SNV_union.tsv (no header, 4 columns).

    Args:
        filtered_vcfs: list of filtered VCF file paths
        union_tsv_path: output path for SNV union TSV
        sample_names: list of sample names corresponding to filtered_vcfs

    Returns:
        dict: SNV key -> {sample_name: frequency}
    """
    logger.info("Building SNV union TSV from %d filtered VCFs", len(filtered_vcfs))
    seen = set()
    snv_freq_by_sample = defaultdict(dict)
    count = 0
    with open(union_tsv_path, "w") as out:
        for vcf, sample_name in zip(filtered_vcfs, sample_names):
            logger.info("  Including variants from %s (sample: %s)", vcf, sample_name)
            with open(vcf, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if line.startswith("#"):
                        continue
                    cols = line.strip().split("\t")
                    if len(cols) < 5:
                        continue
                    chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
                    key = (chrom, pos, ref, alt)
                    if key not in seen:
                        seen.add(key)
                        out.write("\t".join([chrom, pos, ref, alt]) + "\n")
                        count += 1

                    freq = extract_frequency_from_vcf_line(cols)
                    if freq is not None:
                        snv_freq_by_sample[key][sample_name] = freq
    logger.info("SNV union written to %s with %d unique SNVs", union_tsv_path, count)
    return dict(snv_freq_by_sample)


def main(config_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting SNV to barcode pipeline with config %s", config_path)
    start_time = datetime.now()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    settings = config.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Config must include a settings mapping.")

    samples_cfg = config.get("samples")

    output_dir = settings["output_dir"]
    percentage = settings["percentage"]
    threads = _coerce_positive_int(settings.get("threads"), "threads")
    data_type = settings["data_type"]
    count_unit = settings.get("count_unit", "reads")
    p_alt_if_variant = float(settings.get("p_alt_if_variant", 0.5))
    alt_error_rate = float(settings.get("alt_error_rate", 0.001))
    default_prior = float(settings.get("default_prior", 0.001))
    prior_scale = float(settings.get("prior_scale", 2.0))
    prior_cap = float(settings.get("prior_cap", 0.95))
    estimate_p_per_sample = _coerce_bool(
        settings.get("estimate_p_per_sample", False),
        "estimate_p_per_sample",
    )
    p_estimation_min_alt_count = int(settings.get("p_estimation_min_alt_count", 2))
    p_estimation_min_total_count = int(settings.get("p_estimation_min_total_count", 2))
    p_estimation_min_observations = int(settings.get("p_estimation_min_observations", 25))
    p_estimation_min_snvs = int(settings.get("p_estimation_min_snvs", 5))
    pileup_base_quality_min = int(settings.get("pileup_base_quality_min", 20))
    pileup_mapping_quality_min = int(settings.get("pileup_mapping_quality_min", 20))
    neg1_prob_hist_bins = int(settings.get("neg1_prob_hist_bins", 60))
    af_filter_enabled = _coerce_bool(settings.get("af_filter_enabled", False), "af_filter_enabled")
    af_field = str(settings.get("af_field", "gnomad41_genome_AF"))
    af_max = float(settings.get("af_max", 0.001))
    af_strict_snv_format = _coerce_bool(
        settings.get("af_strict_snv_format", False),
        "af_strict_snv_format",
    )
    neg1_prob_threshold_cfg = settings.get("neg1_prob_threshold", None)
    if neg1_prob_threshold_cfg in (None, "", "none", "None"):
        neg1_prob_threshold = None
    else:
        neg1_prob_threshold = float(neg1_prob_threshold_cfg)

    if count_unit not in {"reads", "umis"}:
        raise ValueError("settings.count_unit must be 'reads' or 'umis'.")
    if not (0.0 < p_alt_if_variant < 1.0):
        raise ValueError("settings.p_alt_if_variant must be between 0 and 1.")
    if not (0.0 < alt_error_rate < 1.0):
        raise ValueError("settings.alt_error_rate must be between 0 and 1.")
    if not (0.0 < default_prior < 1.0):
        raise ValueError("settings.default_prior must be between 0 and 1.")
    if not (0.0 < prior_scale):
        raise ValueError("settings.prior_scale must be > 0.")
    if not (0.0 < prior_cap < 1.0):
        raise ValueError("settings.prior_cap must be between 0 and 1.")
    if p_estimation_min_alt_count < 1:
        raise ValueError("settings.p_estimation_min_alt_count must be >= 1.")
    if p_estimation_min_total_count < 1:
        raise ValueError("settings.p_estimation_min_total_count must be >= 1.")
    if p_estimation_min_observations < 1:
        raise ValueError("settings.p_estimation_min_observations must be >= 1.")
    if p_estimation_min_snvs < 1:
        raise ValueError("settings.p_estimation_min_snvs must be >= 1.")
    if pileup_base_quality_min < 0:
        raise ValueError("settings.pileup_base_quality_min must be >= 0.")
    if pileup_mapping_quality_min < 0:
        raise ValueError("settings.pileup_mapping_quality_min must be >= 0.")
    if af_max < 0.0:
        raise ValueError("settings.af_max must be >= 0.")
    if neg1_prob_threshold is not None and not (0.0 <= neg1_prob_threshold <= 1.0):
        raise ValueError("settings.neg1_prob_threshold must be in [0,1] when set.")

    file_paths = config.get("file_paths", {})
    if file_paths is None:
        file_paths = {}
    if not isinstance(file_paths, dict):
        raise ValueError("Config file_paths must be a mapping when provided.")
    snv_union_cfg = file_paths.get("snv_union", "auto")

    _validate_config_inputs(samples_cfg, af_filter_enabled, snv_union_cfg)

    logger.info("All input files verified successfully")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory set to %s", output_dir)
    if af_filter_enabled:
        logger.info(
            "Population AF filtering enabled: field=%s, max_af=%.6f",
            af_field,
            af_max,
        )

    # ===== 1. Filter VCF for each sample and record list of filtered VCFs =====
    filtered_vcfs = []
    sample_names_ordered = []
    for sample_name, sconf in samples_cfg.items():
        vcf_file = sconf["vcf"]
        filtered_vcf = os.path.join(
            output_dir, f"{sample_name}_filtered_less{percentage}.vcf"
        )
        if os.path.exists(filtered_vcf):
            logger.info(
                "Filtered VCF already exists for sample %s, skipping: %s",
                sample_name,
                filtered_vcf,
            )
        else:
            logger.info(
                "Filtering VCF for sample %s: %s -> %s",
                sample_name,
                vcf_file,
                filtered_vcf,
            )
            filter_vcf(vcf_file, filtered_vcf, percentage)
        filtered_vcfs.append(filtered_vcf)
        sample_names_ordered.append(sample_name)

    # ===== 2. Generate global SNV_union.tsv =====
    if snv_union_cfg == "auto":
        snv_union_file = os.path.join(output_dir, "SNV_union.tsv")
        snv_freq_by_sample = build_snv_union_tsv(filtered_vcfs, snv_union_file, sample_names_ordered)
    else:
        snv_union_file = snv_union_cfg
        snv_freq_by_sample = {}
        logger.warning("Using custom SNV union file, FREQ values will be NaN")

    # Read SNV_union.tsv as unified site list
    snv_positions = read_snv_positions(snv_union_file)

    # ===== 3. For each sample, generate matrix based on SNV_union.tsv + its own BAM/CB =====
    per_sample_matrix_files = []
    all_neg1_probabilities = []
    per_sample_p_params = {}
    per_sample_af_lookups = {}
    per_sample_af_stats = {}
    per_sample_annotated_vcfs = {}

    for sample_name, sconf in samples_cfg.items():
        logger.info("Processing sample: %s", sample_name)
        bam_file = sconf["bam"]
        cb_file = sconf["cb"]

        sample_outdir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_outdir, exist_ok=True)

        snv_output_file = os.path.join(
            sample_outdir, f"{sample_name}_reads_supporting_snvs.txt"
        )
        no_snv_output_file = os.path.join(
            sample_outdir, f"{sample_name}_reads_supporting_ref.txt"
        )
        cb_output_file = os.path.join(
            sample_outdir, f"{sample_name}_cb_snvs_count.txt"
        )
        matrix_file = os.path.join(
            sample_outdir, f"{sample_name}_barcode_snv_binary_matrix.h5ad"
        )
        alt_count_output_file = os.path.join(
            sample_outdir, f"{sample_name}_alt_counts.txt"
        )
        ref_count_output_file = os.path.join(
            sample_outdir, f"{sample_name}_ref_counts.txt"
        )
        neg1_prob_plot_file = os.path.join(
            sample_outdir, f"{sample_name}_neg1_ref_posterior_hist.png"
        )
        neg1_prob_summary_file = os.path.join(
            sample_outdir, f"{sample_name}_neg1_ref_posterior_summary.tsv"
        )

        support_step_outputs = [
            snv_output_file,
            no_snv_output_file,
            alt_count_output_file,
            ref_count_output_file,
        ]
        if all(os.path.exists(path) for path in support_step_outputs):
            logger.info(
                "SNV support/ref/count files already exist for sample %s, skipping BAM extraction",
                sample_name,
            )
        else:
            logger.info(
                "Extracting SNV-supporting reads for sample %s from BAM %s",
                sample_name,
                bam_file,
            )
            get_reads_supporting_snv_parallel(
                bam_file,
                snv_positions,
                snv_output_file,
                no_snv_output_file,
                threads,
                alt_count_output_file=alt_count_output_file,
                ref_count_output_file=ref_count_output_file,
                count_unit=count_unit,
                base_quality_min=pileup_base_quality_min,
                mapping_quality_min=pileup_mapping_quality_min,
            )

        if os.path.exists(cb_output_file):
            logger.info(
                "Per-barcode SNV count file already exists for sample %s, skipping: %s",
                sample_name,
                cb_output_file,
            )
        else:
            if data_type == "spatial":
                cb_list = read_cb_list_spatial(cb_file)
            elif data_type == "sc":
                cb_list = read_cb_list_sc(cb_file)
            else:
                raise ValueError(
                    "Invalid data_type specified in config. Must be 'spatial' or 'sc'."
                )

            logger.info("Counting SNVs per barcode for sample %s", sample_name)
            count_snvs_per_cb(snv_output_file, cb_list, cb_output_file)

        if os.path.exists(matrix_file):
            logger.info(
                "Matrix file already exists for sample %s, skipping matrix generation: %s",
                sample_name,
                matrix_file,
            )
        else:
            logger.info("Generating barcode-SNV matrix for sample %s", sample_name)
            generate_barcode_snv_matrix(
                snv_output_file,
                no_snv_output_file,
                alt_count_output_file,
                ref_count_output_file,
                cb_file,
                matrix_file,
                data_type,
                sample_name=sample_name,
                snv_freq_by_sample=snv_freq_by_sample,
                threads=threads,
                p_alt_if_variant=p_alt_if_variant,
                alt_error_rate=alt_error_rate,
                default_prior=default_prior,
                prior_scale=prior_scale,
                prior_cap=prior_cap,
                estimate_p_per_sample=estimate_p_per_sample,
                p_estimation_min_alt_count=p_estimation_min_alt_count,
                p_estimation_min_total_count=p_estimation_min_total_count,
                p_estimation_min_observations=p_estimation_min_observations,
                p_estimation_min_snvs=p_estimation_min_snvs,
                neg1_prob_threshold=neg1_prob_threshold,
                neg1_prob_plot_file=neg1_prob_plot_file,
                neg1_prob_summary_file=neg1_prob_summary_file,
                neg1_prob_hist_bins=neg1_prob_hist_bins,
            )

        if af_filter_enabled:
            annotated_vcf = sconf["annotated_vcf"]
            logger.info(
                "Loading annotated AF VCF for sample %s: %s",
                sample_name,
                annotated_vcf,
            )
            af_lookup, vcf_stats = _load_vcf_af_lookup(
                annotated_vcf,
                af_field,
                af_max,
                strict_snv_format=af_strict_snv_format,
            )
            per_sample_af_lookups[sample_name] = af_lookup
            per_sample_af_stats[sample_name] = vcf_stats
            per_sample_annotated_vcfs[sample_name] = annotated_vcf

            filtered_matrix_file = _derive_af_filtered_h5ad_path(matrix_file)
            if os.path.exists(filtered_matrix_file):
                logger.info(
                    "AF-filtered matrix already exists for sample %s, skipping: %s",
                    sample_name,
                    filtered_matrix_file,
                )
            else:
                _write_population_af_filtered_h5ad(
                    input_h5ad=matrix_file,
                    output_h5ad=filtered_matrix_file,
                    af_lookup=af_lookup,
                    af_field=af_field,
                    af_threshold=af_max,
                    vcf_label=annotated_vcf,
                    vcf_stats=vcf_stats,
                    strict_snv_format=af_strict_snv_format,
                )

        sample_adata = ad.read_h5ad(matrix_file)
        per_sample_p_params[sample_name] = {
            "ref_posterior_params": dict(sample_adata.uns.get("ref_posterior_params", {})),
            "p_alt_if_variant_estimation": dict(sample_adata.uns.get("p_alt_if_variant_estimation", {})),
        }
        sample_probs = _extract_neg1_probabilities_from_adata(sample_adata)
        if (not os.path.exists(neg1_prob_plot_file)) or (not os.path.exists(neg1_prob_summary_file)):
            logger.info(
                "Neg1 posterior outputs missing for sample %s, regenerating from existing matrix",
                sample_name,
            )
            _save_neg1_posterior_outputs(
                sample_probs,
                neg1_prob_plot_file,
                neg1_prob_summary_file,
                bins=neg1_prob_hist_bins,
            )
        if sample_probs.size > 0:
            all_neg1_probabilities.append(sample_probs)

        per_sample_matrix_files.append((sample_name, matrix_file))

    # ===== 4. Merge all sample matrices (axis=0, outer join var, fill_value=0) =====
    if per_sample_matrix_files:
        logger.info("Merging per-sample matrices into a single AnnData object")
        adatas = []
        for sample_name, mfile in per_sample_matrix_files:
            logger.info("  Loading matrix for sample %s: %s", sample_name, mfile)
            a = ad.read_h5ad(mfile)
            # Add a sample field to obs
            a.obs["sample"] = sample_name
            # Prevent CB name collisions between samples: add sample prefix
            a.obs_names = [f"{sample_name}_{bc}" for bc in a.obs_names]
            adatas.append(a)

        merged = ad.concat(adatas, axis=0, join="outer", fill_value=0)
        if per_sample_p_params:
            merged.uns["ref_posterior_params_by_sample"] = per_sample_p_params

        if snv_freq_by_sample:
            merged_freq_values = []
            for snv_key in merged.var_names:
                if isinstance(snv_key, tuple):
                    key = snv_key
                else:
                    parts = snv_key.split(":")
                    key = None
                    if len(parts) == 2:
                        chrom = parts[0]
                        rest = parts[1].split("_")
                        if len(rest) == 2:
                            pos = rest[0]
                            ref_alt = rest[1].split(">")
                            if len(ref_alt) == 2:
                                key = (chrom, pos, ref_alt[0], ref_alt[1])

                if key and key in snv_freq_by_sample:
                    freq_parts = []
                    for sname in sorted(snv_freq_by_sample[key].keys()):
                        freq_val = snv_freq_by_sample[key][sname]
                        freq_parts.append(f"{sname}:{freq_val:.4f}")
                    merged_freq = ";".join(freq_parts)
                else:
                    merged_freq = ""
                merged_freq_values.append(merged_freq)
            merged.var["FREQ"] = merged_freq_values
            logger.info("Updated FREQ column with multi-sample frequencies")

        merged_file = os.path.join(
            output_dir, "all_samples_merged_barcode_snv_matrix.h5ad"
        )
        merged.write_h5ad(merged_file)
        logger.info("Merged matrix saved to %s", merged_file)

        if af_filter_enabled:
            merged_af_lookup = _merge_af_lookups(list(per_sample_af_lookups.values()))
            merged_af_stats = _merge_af_stats(list(per_sample_af_stats.values()))
            merged_af_label = ";".join(
                str(per_sample_annotated_vcfs[sample_name])
                for sample_name in sorted(per_sample_annotated_vcfs.keys())
            )
            merged_filtered_file = _derive_af_filtered_h5ad_path(merged_file)
            _write_population_af_filtered_h5ad(
                input_h5ad=merged_file,
                output_h5ad=merged_filtered_file,
                af_lookup=merged_af_lookup,
                af_field=af_field,
                af_threshold=af_max,
                vcf_label=merged_af_label,
                vcf_stats=merged_af_stats,
                strict_snv_format=af_strict_snv_format,
            )

        if all_neg1_probabilities:
            combined_probs = np.concatenate(all_neg1_probabilities).astype(np.float32)
        else:
            combined_probs = np.array([], dtype=np.float32)
        merged_neg1_hist = os.path.join(output_dir, "all_samples_neg1_ref_posterior_hist.png")
        merged_neg1_summary = os.path.join(output_dir, "all_samples_neg1_ref_posterior_summary.tsv")
        _save_neg1_posterior_outputs(
            combined_probs,
            merged_neg1_hist,
            merged_neg1_summary,
            bins=neg1_prob_hist_bins,
        )
        logger.info(
            "Saved merged -1 posterior distribution outputs to %s and %s",
            merged_neg1_hist,
            merged_neg1_summary,
        )
    else:
        logger.warning("No per-sample matrices were generated; merged matrix not created.")

    elapsed = datetime.now() - start_time
    logger.info("Completed SNV to barcode pipeline in %s", elapsed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python snv2barcode.py <path_to_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
