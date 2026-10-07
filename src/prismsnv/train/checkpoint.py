"""Versioned SNV checkpoints with explicit feature and batch contracts."""

import os
import tempfile
from typing import Iterable

import torch


CHECKPOINT_FORMAT = "prismsnv.snv_perturbation"
CHECKPOINT_VERSION = 1
MODEL_CONFIG_KEYS = (
    "n_genes", "n_snvs", "latent_dim", "snv_emb_dim", "n_batches", "batch_emb_dim"
)


def _validate_names(names: object, label: str, allow_empty: bool = False) -> None:
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError(f"Checkpoint {label} must be a list of non-empty strings.")
    if not names and not allow_empty:
        raise ValueError(f"Checkpoint {label} must not be empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"Checkpoint {label} contains duplicate identifiers.")


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary.")
    for key in ("gene_names", "snv_names", "batch_categories"):
        _validate_names(metadata.get(key), key, allow_empty=key == "batch_categories")
    if not isinstance(metadata.get("batch_key"), str) or not metadata["batch_key"]:
        raise ValueError("Checkpoint batch_key must be a non-empty string.")
    config = metadata.get("model_config")
    if not isinstance(config, dict) or set(config) != set(MODEL_CONFIG_KEYS):
        raise ValueError("Checkpoint model_config is missing or has unsupported fields.")
    for key in MODEL_CONFIG_KEYS:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Checkpoint model_config.{key} must be a positive integer.")
    for key, expected in (
        ("n_genes", len(metadata["gene_names"])),
        ("n_snvs", len(metadata["snv_names"])),
        ("n_batches", max(1, len(metadata["batch_categories"]))),
    ):
        if config[key] != expected:
            raise ValueError(f"Checkpoint {key} does not match its saved metadata.")


def make_checkpoint_metadata(
    gene_names: Iterable[str],
    snv_names: Iterable[str],
    batch_key: str,
    batch_categories: Iterable[str],
    model_config: dict,
) -> dict:
    """Build the ordered metadata that gives model tensor indices their meaning."""
    metadata = {
        "gene_names": list(gene_names),
        "snv_names": list(snv_names),
        "batch_key": batch_key,
        "batch_categories": list(batch_categories),
        "model_config": dict(model_config),
    }
    _validate_metadata(metadata)
    return metadata


def validate_checkpoint_metadata(saved: dict, current: dict) -> None:
    """Reject feature identity/order and model-configuration mismatches."""
    _validate_metadata(saved)
    _validate_metadata(current)
    for key in ("gene_names", "snv_names", "batch_categories"):
        expected, actual = saved[key], current[key]
        if expected != actual:
            mismatch = next(
                (i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                min(len(expected), len(actual)),
            )
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {len(expected)} entries, "
                f"got {len(actual)}; first difference at index {mismatch}: "
                f"saved={expected[mismatch:mismatch + 1]!r}, "
                f"current={actual[mismatch:mismatch + 1]!r}. "
                "Use the same feature identities and order as training."
            )
    for key in ("batch_key", "model_config"):
        if saved[key] != current[key]:
            raise ValueError(
                f"Checkpoint {key} mismatch: saved={saved[key]!r}, current={current[key]!r}."
            )


def load_snv_checkpoint(path: str) -> dict:
    """Load a verifiable checkpoint without inferring metadata from current inputs."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint {path!r} has no verifiable training metadata (legacy state_dict "
            "checkpoints are not supported for evaluation). Retrain with the current "
            "checkpoint format. Do not reconstruct training metadata from evaluation "
            "inputs or an unverified .snvs.npy sidecar."
        )
    if type(checkpoint.get("version")) is not int or checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported SNV checkpoint version: {checkpoint.get('version')!r}.")
    _validate_metadata(checkpoint.get("metadata"))
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state_dict is missing or empty.")
    return checkpoint


def save_snv_checkpoint(path: str, state_dict: dict, metadata: dict) -> None:
    """Atomically replace weights and their metadata together without changing dtypes."""
    _validate_metadata(metadata)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".snv_checkpoint_", suffix=".pt", dir=parent)
    os.close(fd)
    try:
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "version": CHECKPOINT_VERSION,
                "model_state_dict": state_dict,
                "metadata": metadata,
            },
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
