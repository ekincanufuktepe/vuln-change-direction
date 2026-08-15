from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .constants import FIXING, INTRODUCING, METADATA_COLUMNS, OUTCOME
from .io import write_json
# new addition to pause and continue training
from .checkpoint_hillclimb import CheckpointHillClimbSearch


def forbidden_sink_edges(columns: Iterable[str], outcome: str = OUTCOME) -> set[tuple[str, str]]:
    return {(outcome, column) for column in columns if column != outcome}


def _load_pgmpy():
    try:
        from pgmpy.estimators import BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch
        from pgmpy.models import DiscreteBayesianNetwork
    except ImportError as exc:
        raise RuntimeError("Bayesian-network commands require the project dependencies") from exc
    return BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch, DiscreteBayesianNetwork


def _bdeu_marginal(series: pd.Series, equivalent_sample_size: float) -> tuple[list, list[float]]:
    """Estimate a BDeu-smoothed marginal distribution for an isolated node."""
    if equivalent_sample_size <= 0:
        raise ValueError("equivalent_sample_size must be greater than zero")
    if series.isna().any():
        raise ValueError(f"Missing values are not supported for isolated node {series.name!r}")
    if isinstance(series.dtype, pd.CategoricalDtype):
        states = list(series.cat.categories)
    else:
        states = sorted(series.unique().tolist(), key=str)
    if not states:
        raise ValueError(f"Cannot estimate a CPD for empty node {series.name!r}")
    counts = series.value_counts(sort=False).reindex(states, fill_value=0).astype(float)
    alpha = equivalent_sample_size / len(states)
    probabilities = (counts.to_numpy() + alpha) / (float(counts.sum()) + equivalent_sample_size)
    return states, probabilities.tolist()


def _add_missing_isolated_cpds(model, data: pd.DataFrame, equivalent_sample_size: float) -> list[str]:
    """Work around pgmpy 1.0 dropping isolated nodes during Bayesian estimation.

    An isolated node has no parents, so its CPD is its marginal distribution.
    The calculation below uses the same BDeu equivalent-sample-size prior as
    the parameter estimator used for the connected part of the network.
    """
    from pgmpy.factors.discrete import TabularCPD

    missing = [node for node in model.nodes() if model.get_cpds(node) is None]
    for node in missing:
        if model.degree(node) != 0:
            raise ValueError(f"Connected node {node!r} is missing its CPD; this is not an isolated-node case")
        states, probabilities = _bdeu_marginal(data[node], equivalent_sample_size)
        cpd = TabularCPD(
            variable=node,
            variable_card=len(states),
            values=np.asarray(probabilities, dtype=float).reshape(len(states), 1),
            state_names={node: states},
        )
        model.add_cpds(cpd)
    model.graph["isolated_cpd_nodes"] = sorted(missing)
    return missing


def learn_model(
    data: pd.DataFrame,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
    show_progress: bool = False,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
):
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if  max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch, DiscreteBayesianNetwork = _load_pgmpy()
    model_data = data.copy()
    for column in model_data.columns:
        model_data[column] = model_data[column].astype("category")
    forbidden = forbidden_sink_edges(model_data.columns)
    knowledge = ExpertKnowledge(forbidden_edges=forbidden)
    if score == "bdeu":
        structure_score = BDeu(
            model_data,
            equivalent_sample_size=equivalent_sample_size,
        )
    else:
        structure_score = score
    '''
    dag = HillClimbSearch(model_data).estimate(
        #scoring_method=score,
        scoring_method=structure_score,
        expert_knowledge=knowledge,
        max_indegree=max_indegree,
        show_progress=show_progress,
    )
    '''
    # Change made to see log, use cache, and set an iteration limit for bdeu
    #dag = HillClimbSearch(model_data, use_cache=True).estimate(
    dag = CheckpointHillClimbSearch(model_data).estimate(
        scoring_method=structure_score,
        expert_knowledge=knowledge,
        max_indegree=max_indegree,
        show_progress=True,
        max_iter=2000,
        epsilon=epsilon,
        # new params for checkpoint hill climb
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    if any(source == OUTCOME for source, _ in dag.edges()):
        raise AssertionError("Sink constraint was violated by structure learning")
    model = DiscreteBayesianNetwork(dag.edges())
    # for checkpoint hill climbing
    model.graph.update(dag.graph)
    
    model.add_nodes_from(model_data.columns)
    model.fit(
        model_data,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=equivalent_sample_size,
    )
    _add_missing_isolated_cpds(model, model_data, equivalent_sample_size)
    model.check_model()
    return model


def train_from_csv(
    dataset_csv: str | Path,
    model_path: str | Path,
    summary_path: str | Path,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    frame = pd.read_csv(dataset_csv)
    feature_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError("Dataset has no change-type features")
    model_data = frame[feature_columns + [OUTCOME]].copy()
    model = learn_model(model_data, score, max_indegree, equivalent_sample_size, epsilon=epsilon, max_iter=max_iter, checkpoint_path=checkpoint_path, resume=resume)
    bundle = {
        "model": model,
        "features": feature_columns,
        "outcome": OUTCOME,
        "outcome_states": [FIXING, INTRODUCING],
        "score": score,
        "max_indegree": max_indegree,
        "equivalent_sample_size": equivalent_sample_size,
        "epsilon": epsilon,
        "max_iter": max_iter,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    edges = sorted([list(edge) for edge in model.edges()])
    summary = {
        "nodes": sorted(model.nodes()),
        "edges": edges,
        "outcome_parents": sorted(model.get_parents(OUTCOME)),
        "outcome_children": sorted(model.get_children(OUTCOME)),
        "sink_constraint_satisfied": len(model.get_children(OUTCOME)) == 0,
        "isolated_nodes_with_marginal_cpds": model.graph.get("isolated_cpd_nodes", []),
        "interpretation": "Learned directions are statistical, not causal or temporal claims.",
        "epsilon": epsilon,
        "max_iter": max_iter,
    }
    write_json(summary_path, summary)
    return summary


def predict_present_features(model_path: str | Path, present_features: Iterable[str]) -> dict:
    from pgmpy.inference import VariableElimination

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    present = set(present_features)
    unknown = sorted(present - set(features))
    evidence = {feature: int(feature in present) for feature in features}
    query = VariableElimination(model).query([OUTCOME], evidence=evidence, show_progress=False)
    states = query.state_names[OUTCOME]
    probabilities = {str(state): float(query.values[index]) for index, state in enumerate(states)}
    return {"probabilities": probabilities, "unknown_features": unknown, "evidence_features": len(features)}


def write_prediction(model_path: str | Path, evidence_json: str | Path, output_json: str | Path) -> dict:
    with Path(evidence_json).open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    result = predict_present_features(model_path, evidence.get("present_features", []))
    write_json(output_json, result)
    return result
