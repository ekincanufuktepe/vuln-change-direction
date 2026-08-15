from __future__ import annotations

import argparse
import json

from .bn import train_from_csv, write_prediction
from .bootstrap import bootstrap_edges
from .coming import run_coming
from .dataset import build_dataset
from .evaluate import evaluate_holdout
from .graph import write_dot
from .manifest import write_paired_manifest
from .materialize import materialize_samples
from .significance import mutual_information_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="megavul-bn")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="Build paired fixing and synthetic-inverse samples")
    manifest.add_argument("--megavul", required=True)
    manifest.add_argument("--output", required=True)

    materialize = commands.add_parser("materialize", help="Materialize full Java source pairs")
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--repositories", required=True)
    materialize.add_argument("--pairs", required=True)
    materialize.add_argument("--clone-missing", action="store_true")

    coming = commands.add_parser("coming", help="Run Coming once per labeled revision")
    coming.add_argument("--manifest", required=True)
    coming.add_argument("--pairs", required=True)
    coming.add_argument("--output", required=True)
    coming.add_argument("--coming-jar", required=True)
    coming.add_argument("--java", default="java")
    coming.add_argument("--workers", type=int, default=1)

    dataset = commands.add_parser("dataset", help="Build the discrete change-type matrix")
    dataset.add_argument("--manifest", required=True)
    dataset.add_argument("--coming-output", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--metadata", required=True)
    dataset.add_argument("--representation", choices=["presence", "count", "binned"], default="presence")
    dataset.add_argument("--min-support", type=int, default=1)

    learn = commands.add_parser("learn", help="Learn a sink-constrained Bayesian network")
    learn.add_argument("--dataset", required=True)
    learn.add_argument("--model", required=True)
    learn.add_argument("--summary", required=True)
    learn.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    learn.add_argument("--max-indegree", type=int, default=4)
    learn.add_argument("--equivalent-sample-size", type=float, default=5.0)
    learn.add_argument("--epsilon", type=float, default=1e-4, help="Minimum score improvement required to accept a graph operation")
    learn.add_argument("--max-iterations", type=int, default=1_000_000, help="Maximum number of hill-climbing operations")
    # for checkpoint hill climb
    learn.add_argument(
        "--checkpoint",
        help="Path for the resumable structure-learning checkpoint",
    )
    learn.add_argument(
        "--resume",
        action="store_true",
        help="Resume structure learning from the checkpoint",
    )

    evaluate = commands.add_parser("evaluate", help="Grouped holdout evaluation")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--group", choices=["project", "pair_id"], default="project")
    evaluate.add_argument("--test-size", type=float, default=0.2)
    evaluate.add_argument("--random-state", type=int, default=42)
    evaluate.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    evaluate.add_argument("--max-indegree", type=int, default=4)
    evaluate.add_argument("--equivalent-sample-size", type=float, default=5.0)

    predict = commands.add_parser("predict", help="Infer outcome probabilities for extracted change types")
    predict.add_argument("--model", required=True)
    predict.add_argument("--evidence", required=True)
    predict.add_argument("--output", required=True)

    significance = commands.add_parser("mi", help="Pair-aware mutual-information significance analysis")
    significance.add_argument("--dataset", required=True)
    significance.add_argument("--output", required=True)
    significance.add_argument("--permutations", type=int, default=1000)
    significance.add_argument("--random-state", type=int, default=42)

    bootstrap = commands.add_parser("bootstrap", help="Pair-preserving edge-stability bootstrap")
    bootstrap.add_argument("--dataset", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--repetitions", type=int, default=100)
    bootstrap.add_argument("--random-state", type=int, default=42)
    bootstrap.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    bootstrap.add_argument("--max-indegree", type=int, default=4)
    bootstrap.add_argument("--equivalent-sample-size", type=float, default=5.0)

    dot = commands.add_parser("dot", help="Export a learned structure summary as Graphviz DOT")
    dot.add_argument("--summary", required=True)
    dot.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        result = write_paired_manifest(args.megavul, args.output)
        message = {"samples": len(result), "pairs": len(result) // 2}
    elif args.command == "materialize":
        message = materialize_samples(args.manifest, args.repositories, args.pairs, args.clone_missing)
    elif args.command == "coming":
        message = run_coming(args.manifest, args.pairs, args.output, args.coming_jar, args.workers, args.java)
    elif args.command == "dataset":
        frame = build_dataset(
            args.manifest,
            args.coming_output,
            args.output,
            args.metadata,
            args.representation,
            args.min_support,
        )
        message = {"samples": len(frame), "columns": len(frame.columns)}
    elif args.command == "learn":
        message = train_from_csv(
            args.dataset,
            args.model,
            args.summary,
            args.score,
            args.max_indegree,
            args.equivalent_sample_size,
            epsilon=args.epsilon,
            max_iter=args.max_iterations,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        message = evaluate_holdout(
            args.dataset,
            args.output,
            args.group,
            args.test_size,
            args.random_state,
            args.score,
            args.max_indegree,
            args.equivalent_sample_size,
        )
    elif args.command == "predict":
        message = write_prediction(args.model, args.evidence, args.output)
    elif args.command == "mi":
        report = mutual_information_report(
            args.dataset, args.output, args.permutations, args.random_state
        )
        message = {"features": len(report), "output": args.output}
    elif args.command == "bootstrap":
        message = bootstrap_edges(
            args.dataset,
            args.output,
            args.repetitions,
            args.random_state,
            args.score,
            args.max_indegree,
            args.equivalent_sample_size,
        )
    elif args.command == "dot":
        write_dot(args.summary, args.output)
        message = {"output": args.output}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(message, indent=2, default=str))


if __name__ == "__main__":
    main()
