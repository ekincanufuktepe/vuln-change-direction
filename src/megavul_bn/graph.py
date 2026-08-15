from __future__ import annotations

from pathlib import Path

from .constants import OUTCOME
from .io import read_json


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_dot(summary_json: str | Path, output_dot: str | Path) -> None:
    summary = read_json(summary_json)
    lines = ["digraph BayesianNetwork {", "  rankdir=LR;", "  node [shape=box];"]
    for node in summary["nodes"]:
        if node == OUTCOME:
            lines.append(f"  {_quote(node)} [shape=doublecircle, style=filled, fillcolor=lightgoldenrod1];")
        else:
            lines.append(f"  {_quote(node)};")
    for source, target in summary["edges"]:
        lines.append(f"  {_quote(source)} -> {_quote(target)};")
    lines.append("}")
    destination = Path(output_dot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

