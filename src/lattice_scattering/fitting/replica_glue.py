"""Replica and resampling glue for joint fit results.

ReplicaTable and fit_replicas helpers preserve sample identity
and record per-replica failures."""

from __future__ import annotations

import numpy as np

from lattice_scattering.data.replicas import ReplicaTable
from lattice_scattering.fitting.replicas import fit_replicas

__all__ = ["ReplicaTable", "fit_replicas", "replica_summary"]


def replica_summary(result: dict) -> dict:
    """Summarise a :func:`fit_replicas` result without hiding failures.

    ``central_estimate`` is the replica mean **only** when every sample
    succeeded; with failed samples it is ``None`` and the failure identities are
    listed, because a mean over a success-only subset is not the ensemble mean.
    """
    if not isinstance(result, dict) or "records" not in result:
        raise ValueError("a fit_replicas result dictionary is required")
    records = result["records"]
    succeeded = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") == "failed"]
    pending = [r for r in records if r.get("status") == "pending"]
    summary = {
        "n_samples": len(records),
        "n_success": len(succeeded),
        "n_failed": len(failed),
        "n_pending": len(pending),
        "complete": bool(result.get("complete", False)),
        "failed_samples": [r.get("sample_id") for r in failed],
        "failure_reasons": {
            r.get("sample_id"): f"{r.get('error_type')}: {r.get('error')}" for r in failed
        },
        "parameter_ids": list(result.get("parameter_ids", ())),
        "input_fingerprint": result.get("input_fingerprint"),
        "run_id": result.get("run_id"),
        "replica_mean": None,
        "replica_covariance": None,
        "scope": (
            "replica mean/covariance reported only for a complete sample set; "
            "failed samples are never replaced by the central estimate"
        ),
    }
    if result.get("complete"):
        values = np.asarray([r["parameters"] for r in records], dtype=float)
        summary["replica_mean"] = values.mean(axis=0).tolist()
        summary["replica_covariance"] = np.cov(values, rowvar=False, ddof=1).tolist()
    return summary
