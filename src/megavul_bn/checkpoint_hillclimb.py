from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import networkx as nx
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch
from pgmpy.estimators.StructureScore import get_scoring_method


class CheckpointHillClimbSearch(HillClimbSearch):
    """Hill-climbing search with progress reporting and resumable checkpoints."""

    @staticmethod
    def _save_checkpoint(
        checkpoint_path: str | Path | None,
        current_model: DAG,
        tabu_list,
        completed_iterations: int,
        status: str,
        best_operation=None,
        best_score_delta=None,
    ) -> None:
        if checkpoint_path is None:
            return

        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dag": current_model.copy(),
            "tabu_list": list(tabu_list),
            "completed_iterations": completed_iterations,
            "status": status,
            "best_operation": best_operation,
            "best_score_delta": (
                None
                if best_score_delta is None
                else float(best_score_delta)
            ),
        }

        temporary = destination.with_name(destination.name + ".tmp")
        joblib.dump(payload, temporary)
        temporary.replace(destination)

    def estimate(
        self,
        scoring_method=None,
        start_dag=None,
        tabu_length: int = 100,
        max_indegree: int | None = None,
        expert_knowledge=None,
        epsilon: float = 1e-4,
        max_iter: int = 1_000_000,
        show_progress: bool = True,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ):
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        score, cached_score = get_scoring_method(
            scoring_method,
            self.data,
            self.use_cache,
        )
        score_fn = cached_score.local_score

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if resume:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when resume=True"
                )

            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint does not exist: {checkpoint}"
                )

            payload = joblib.load(checkpoint)
            current_model = payload["dag"]
            completed_iterations = int(
                payload.get("completed_iterations", 0)
            )
            restored_tabu = payload.get("tabu_list", [])
        else:
            completed_iterations = 0
            restored_tabu = []

            if start_dag is None:
                current_model = DAG()
                current_model.add_nodes_from(self.variables)
            else:
                if not isinstance(start_dag, DAG):
                    raise ValueError("start_dag must be a pgmpy DAG")

                if set(start_dag.nodes()) != set(self.variables):
                    raise ValueError(
                        "start_dag must contain the same variables as the data"
                    )

                current_model = start_dag.copy()

        if set(current_model.nodes()) != set(self.variables):
            raise ValueError(
                "Checkpoint DAG does not contain the same variables as the data"
            )

        search_space = getattr(
            expert_knowledge,
            "search_space",
            None,
        )

        limit_search_space = getattr(
            expert_knowledge,
            "limit_search_space",
            None,
        )

        if search_space and callable(limit_search_space):
            limit_search_space(self.data.columns)

        if resume:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when resume=True"
                )

        current_model.add_edges_from(expert_knowledge.required_edges)

        if not nx.is_directed_acyclic_graph(current_model):
            raise ValueError(
                "Required edges or checkpoint structure contain a cycle"
            )

        expert_knowledge._orient_temporal_forbidden_edges(
            current_model,
            only_edges=False,
        )
        current_model.remove_edges_from(
            expert_knowledge.forbidden_edges
        )

        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(restored_tabu, maxlen=tabu_length)

        status = "max_iter"
        last_operation = None
        last_delta = None

        progress = tqdm(
            range(completed_iterations, int(max_iter)),
            total=int(max_iter),
            initial=completed_iterations,
            disable=not (
                show_progress and config.SHOW_PROGRESS
            ),
            desc="Hill-climbing",
        )

        try:
            for _ in progress:
                best_operation, best_score_delta = max(
                    self._legal_operations(
                        current_model,
                        score_fn,
                        score.structure_prior_ratio,
                        tabu_list,
                        max_indegree,
                        expert_knowledge.forbidden_edges,
                        expert_knowledge.required_edges,
                    ),
                    key=lambda candidate: candidate[1],
                    default=(None, None),
                )

                last_operation = best_operation
                last_delta = best_score_delta

                operation_text = (
                    "none"
                    if best_operation is None
                    else str(best_operation)
                )

                delta_text = (
                    "none"
                    if best_score_delta is None
                    else f"{best_score_delta:.6f}"
                )

                progress.set_postfix(
                    {
                        "edges": current_model.number_of_edges(),
                        "best_delta": delta_text,
                        "epsilon": f"{epsilon:.6f}",
                        "operation": operation_text,
                    },
                    refresh=True,
                )

                if best_operation is None:
                    status = "no_legal_operation"
                    break

                best_score_delta = float(best_score_delta)
                epsilon = float(epsilon)

                print("Best score delta: " + str(best_score_delta))
                progress.write(
                    f"Stop check: delta={best_score_delta:.9f}, "
                    f"epsilon={epsilon:.9f}, "
                    f"stop={best_score_delta < epsilon}"
                )

                if best_score_delta < epsilon:
                    status = "epsilon"
                    break

                operation_type, edge = best_operation

                if operation_type == "+":
                    current_model.add_edge(*edge)
                    tabu_list.append(("-", edge))

                elif operation_type == "-":
                    current_model.remove_edge(*edge)
                    tabu_list.append(("+", edge))

                elif operation_type == "flip":
                    source, target = edge
                    current_model.remove_edge(source, target)
                    current_model.add_edge(target, source)
                    tabu_list.append(best_operation)

                else:
                    raise ValueError(
                        f"Unknown graph operation: {operation_type}"
                    )

                completed_iterations += 1

                self._save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    current_model=current_model,
                    tabu_list=tabu_list,
                    completed_iterations=completed_iterations,
                    status="running",
                    best_operation=best_operation,
                    best_score_delta=best_score_delta,
                )

        except KeyboardInterrupt:
            status = "interrupted"

            self._save_checkpoint(
                checkpoint_path=checkpoint_path,
                current_model=current_model,
                tabu_list=tabu_list,
                completed_iterations=completed_iterations,
                status=status,
                best_operation=last_operation,
                best_score_delta=last_delta,
            )

            print(
                "\nSearch interrupted. The latest completed DAG was saved.",
                flush=True,
            )

        finally:
            progress.close()

        current_model.graph["search_status"] = status
        current_model.graph["search_iterations"] = completed_iterations
        current_model.graph["search_epsilon"] = epsilon
        current_model.graph["search_max_iter"] = max_iter
        current_model.graph["last_best_score_delta"] = (
            None if last_delta is None else float(last_delta)
        )

        self._save_checkpoint(
            checkpoint_path=checkpoint_path,
            current_model=current_model,
            tabu_list=tabu_list,
            completed_iterations=completed_iterations,
            status=status,
            best_operation=last_operation,
            best_score_delta=last_delta,
        )

        return current_model