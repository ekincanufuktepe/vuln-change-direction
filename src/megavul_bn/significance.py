from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from .constants import INTRODUCING, METADATA_COLUMNS, OUTCOME


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def mutual_information_report(
    dataset_csv: str | Path,
    output_csv: str | Path,
    permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features with pair-aware permutation p-values.

    Under each permutation, the two outcome labels are either retained or swapped
    independently within every pair. This preserves the paired design and balance.
    """
    frame = pd.read_csv(dataset_csv)
    features = [column for column in frame.columns if column not in METADATA_COLUMNS]
    truth = (frame[OUTCOME] == INTRODUCING).astype(int).to_numpy()
    observed = np.array([mutual_info_score(frame[feature], truth) for feature in features])
    exceedances = np.ones(len(features), dtype=int)
    rng = np.random.default_rng(random_state)
    pair_indices = [group.index.to_numpy() for _, group in frame.groupby("pair_id", sort=False)]

    for _ in range(permutations):
        permuted = truth.copy()
        for indices in pair_indices:
            if len(indices) == 2 and rng.integers(0, 2):
                permuted[indices] = permuted[indices[::-1]]
        scores = np.array([mutual_info_score(frame[feature], permuted) for feature in features])
        exceedances += scores >= observed

    p_values = exceedances / (permutations + 1)
    q_values = _benjamini_hochberg(p_values)
    support = np.array([(frame[feature] != 0).sum() for feature in features])
    report = pd.DataFrame(
        {
            "feature": features,
            "mutual_information_nats": observed,
            "support": support,
            "permutation_p": p_values,
            "bh_q": q_values,
        }
    ).sort_values(["mutual_information_nats", "support"], ascending=[False, False])
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return report

