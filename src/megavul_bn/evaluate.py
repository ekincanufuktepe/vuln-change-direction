from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit

from .bn import learn_model
from .constants import FIXING, INTRODUCING, METADATA_COLUMNS, OUTCOME
from .io import write_json


def _probability_of_introducing(inference, row: pd.Series, features: list[str]) -> float:
    evidence = {feature: int(row[feature]) for feature in features}
    query = inference.query([OUTCOME], evidence=evidence, show_progress=False)
    states = list(query.state_names[OUTCOME])
    return float(query.values[states.index(INTRODUCING)])


def evaluate_holdout(
    dataset_csv: str | Path,
    output_json: str | Path,
    group_column: str = "project",
    test_size: float = 0.2,
    random_state: int = 42,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
) -> dict:
    frame = pd.read_csv(dataset_csv)
    if group_column not in frame:
        raise ValueError(f"Unknown split group {group_column!r}")
    all_features = [column for column in frame.columns if column not in METADATA_COLUMNS]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, groups=frame[group_column]))
    train, test = frame.iloc[train_index], frame.iloc[test_index]
    features = [feature for feature in all_features if train[feature].nunique(dropna=False) > 1]
    if not features:
        raise ValueError("No nonconstant change-type features remain in the training partition")
    model = learn_model(
        train[features + [OUTCOME]],
        score=score,
        max_indegree=max_indegree,
        equivalent_sample_size=equivalent_sample_size,
    )
    from pgmpy.inference import VariableElimination

    inference = VariableElimination(model)
    cache: dict[tuple[int, ...], float] = {}
    probabilities_list = []
    for _, row in test.iterrows():
        key = tuple(int(row[feature]) for feature in features)
        if key not in cache:
            cache[key] = _probability_of_introducing(inference, row, features)
        probabilities_list.append(cache[key])
    probabilities = np.array(probabilities_list)
    truth = (test[OUTCOME] == INTRODUCING).astype(int).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    observed, predicted = calibration_curve(truth, probabilities, n_bins=10, strategy="uniform")
    result = {
        "split_group": group_column,
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "train_groups": int(train[group_column].nunique()),
        "test_groups": int(test[group_column].nunique()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "brier": float(brier_score_loss(truth, probabilities)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "calibration": [
            {"mean_predicted": float(pred), "observed_fraction": float(obs)}
            for pred, obs in zip(predicted, observed)
        ],
        "features_used": len(features),
        "outcome_parents": sorted(model.get_parents(OUTCOME)),
        "isolated_nodes_with_marginal_cpds": model.graph.get("isolated_cpd_nodes", []),
        "warning": "Synthetic inverse performance does not establish validity on naturally authored inducing commits.",
    }
    write_json(output_json, result)
    return result
