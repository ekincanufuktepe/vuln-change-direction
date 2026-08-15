from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .bn import learn_model
from .constants import METADATA_COLUMNS, OUTCOME
from .io import write_json


def bootstrap_edges(
    dataset_csv: str | Path,
    output_json: str | Path,
    repetitions: int = 100,
    random_state: int = 42,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
) -> dict:
    """Pair-preserving bootstrap frequencies for directed edges and adjacencies."""
    frame = pd.read_csv(dataset_csv)
    features = [column for column in frame.columns if column not in METADATA_COLUMNS]
    pair_groups = {pair_id: group for pair_id, group in frame.groupby("pair_id", sort=False)}
    pair_ids = np.array(list(pair_groups))
    if not len(pair_ids):
        raise ValueError("Dataset has no complete pairs")
    rng = np.random.default_rng(random_state)
    directed: Counter[tuple[str, str]] = Counter()
    adjacency: Counter[tuple[str, str]] = Counter()
    failures: list[dict] = []
    completed = 0

    for repetition in range(repetitions):
        sampled_ids = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled = pd.concat([pair_groups[pair_id] for pair_id in sampled_ids], ignore_index=True)
        active_features = [feature for feature in features if sampled[feature].nunique(dropna=False) > 1]
        try:
            model = learn_model(
                sampled[active_features + [OUTCOME]],
                score=score,
                max_indegree=max_indegree,
                equivalent_sample_size=equivalent_sample_size,
            )
            completed += 1
            for source, target in model.edges():
                directed[(source, target)] += 1
                adjacency[tuple(sorted((source, target)))] += 1
        except Exception as exc:
            failures.append({"repetition": repetition, "error": str(exc)})

    if completed == 0:
        raise RuntimeError("Every bootstrap structure-learning run failed")
    report = {
        "requested_repetitions": repetitions,
        "completed_repetitions": completed,
        "directed_edges": [
            {"source": source, "target": target, "frequency": count / completed, "count": count}
            for (source, target), count in directed.most_common()
        ],
        "adjacencies": [
            {"node_a": a, "node_b": b, "frequency": count / completed, "count": count}
            for (a, b), count in adjacency.most_common()
        ],
        "failures": failures,
        "note": "Adjacency stability is more robust than direction when structures are Markov equivalent.",
    }
    write_json(output_json, report)
    return report

