#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import graphviz
import plotly.graph_objects as go
from plotly.colors import qualitative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a sempipes trajectory JSON as a tree."
    )
    parser.add_argument("trajectory_json", type=Path, help="Path to trajectory JSON file.")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("trajectory_tree.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional PNG output path (requires kaleido).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional chart title.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=("score", "reward", "both"),
        default="score",
        help="Node label: CV score, bandit reward (Δ vs parent), or both.",
    )
    return parser.parse_args()


def load_outcomes(trajectory_path: Path) -> list[dict[str, Any]]:
    with trajectory_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    outcomes = payload.get("outcomes", [])
    if not outcomes:
        raise ValueError(f"No outcomes found in {trajectory_path}")
    return outcomes


def build_nodes(outcomes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for outcome in outcomes:
        search_node = outcome.get("search_node", {})
        trial = search_node.get("trial")
        if trial is None:
            continue

        operator = search_node.get("operator_to_evolve")
        score = outcome.get("score")
        parent_trial = search_node.get("parent_trial")
        parent_score = search_node.get("parent_score")
        reward = None
        if score is not None and parent_score is not None:
            reward = float(score) - float(parent_score)
        nodes[int(trial)] = {
            "trial": int(trial),
            "parent_trial": None if parent_trial is None else int(parent_trial),
            "operator": operator if operator is not None else "ROOT",
            "score": score,
            "parent_score": parent_score,
            "reward": reward,
        }
    if not nodes:
        raise ValueError("No valid trial nodes were found in outcomes.")
    return nodes


def compute_layout(
    nodes: dict[int, dict[str, Any]],
) -> tuple[dict[int, float], dict[int, float], list[tuple[int, int]], list[list[tuple[float, float]]]]:
    edges: list[tuple[int, int]] = []
    for trial, node in nodes.items():
        parent_trial = node["parent_trial"]
        if parent_trial is not None:
            edges.append((parent_trial, trial))

    graph = graphviz.Digraph(engine="dot")
    graph.attr(rankdir="TB", splines="true", overlap="false", nodesep="0.6", ranksep="0.8")
    graph.attr("node", shape="circle", width="0.35", height="0.35", fixedsize="true")
    for trial in sorted(nodes):
        graph.node(str(trial))
    for parent_trial, child_trial in edges:
        graph.edge(str(parent_trial), str(child_trial))

    try:
        plain = graph.pipe(format="plain").decode("utf-8")
    except graphviz.ExecutableNotFound as exc:
        raise RuntimeError(
            "Graphviz executable not found. Install system graphviz (dot) to compute non-overlapping layout."
        ) from exc

    x_pos: dict[int, float] = {}
    y_pos: dict[int, float] = {}
    edge_paths: list[list[tuple[float, float]]] = []
    for line in plain.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node":
            trial = int(parts[1])
            x_pos[trial] = float(parts[2])
            y_pos[trial] = float(parts[3])
        elif parts[0] == "edge":
            points_count = int(parts[3])
            points: list[tuple[float, float]] = []
            for idx in range(points_count):
                x = float(parts[4 + 2 * idx])
                y = float(parts[5 + 2 * idx])
                points.append((x, y))
            edge_paths.append(points)

    return x_pos, y_pos, edges, edge_paths


def _node_label(node: dict[str, Any], metric: str) -> str:
    score = node["score"]
    reward = node["reward"]
    score_str = "n/a" if score is None else f"{float(score):.4f}"
    if reward is None:
        reward_str = "—"
    else:
        reward_str = f"{reward:+.4f}"
    if metric == "score":
        return score_str
    if metric == "reward":
        return reward_str
    return f"{score_str}<br>{reward_str}"


def create_figure(
    nodes: dict[int, dict[str, Any]],
    x_pos: dict[int, float],
    y_pos: dict[int, float],
    edge_paths: list[list[tuple[float, float]]],
    title: str,
    metric: str = "score",
) -> go.Figure:
    operators = sorted({node["operator"] for node in nodes.values()})
    operator_counts = Counter(node["operator"] for node in nodes.values())
    palette = qualitative.Plotly + qualitative.D3 + qualitative.Set3
    color_map = {op: palette[idx % len(palette)] for idx, op in enumerate(operators)}

    edge_x: list[float] = []
    edge_y: list[float] = []
    for points in edge_paths:
        for x, y in points:
            edge_x.append(x)
            edge_y.append(y)
        edge_x.append(None)
        edge_y.append(None)

    valid_scores = [(trial, node["score"]) for trial, node in nodes.items() if node["score"] is not None]
    best_trial = max(valid_scores, key=lambda item: float(item[1]))[0] if valid_scores else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.0, "color": "rgba(140,140,140,0.55)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for operator in operators:
        op_trials = [trial for trial, node in nodes.items() if node["operator"] == operator]
        op_trials.sort()
        score_text = []
        custom_data = []
        for trial in op_trials:
            node = nodes[trial]
            score_str = "n/a" if node["score"] is None else f"{float(node['score']):.4f}"
            reward_str = "—" if node["reward"] is None else f"{float(node['reward']):+.4f}"
            score_text.append(_node_label(node, metric))
            custom_data.append((trial, operator, score_str, reward_str))

        fig.add_trace(
            go.Scatter(
                x=[x_pos[trial] for trial in op_trials],
                y=[y_pos[trial] for trial in op_trials],
                mode="markers+text",
                text=score_text,
                textposition="top center",
                marker={
                    "size": [24 if trial == best_trial else 16 for trial in op_trials],
                    "color": color_map[operator],
                    "line": {
                        "width": [4 if trial == best_trial else 1 for trial in op_trials],
                        "color": ["#111111" if trial == best_trial else "#ffffff" for trial in op_trials],
                    },
                },
                name=f"{operator} ({operator_counts[operator]})",
                customdata=custom_data,
                hovertemplate=(
                    "Trial: %{customdata[0]}<br>"
                    "Optimized operator: %{customdata[1]}<br>"
                    "CV score: %{customdata[2]}<br>"
                    "Reward (Δ vs parent): %{customdata[3]}<extra></extra>"
                ),
            )
        )

    if best_trial is not None:
        best_score = float(nodes[best_trial]["score"])
        fig.add_annotation(
            x=x_pos[best_trial],
            y=y_pos[best_trial],
            text=f"Best: t{best_trial} ({best_score:.4f})",
            showarrow=True,
            arrowhead=2,
            ay=-35,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#111111",
            borderwidth=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"title": {"text": "Optimized operator (count)"}},
    )
    return fig


def main() -> None:
    args = parse_args()
    outcomes = load_outcomes(args.trajectory_json)
    nodes = build_nodes(outcomes)
    x_pos, y_pos, _, edge_paths = compute_layout(nodes)

    title = args.title or f"Trajectory Tree - {args.trajectory_json.name}"
    figure = create_figure(nodes, x_pos, y_pos, edge_paths, title, metric=args.metric)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(args.output_html), include_plotlyjs="cdn")
    print(f"Wrote interactive tree: {args.output_html}")

    if args.output_png is not None:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(str(args.output_png), width=1800, height=1000, scale=2)
        print(f"Wrote PNG: {args.output_png}")


if __name__ == "__main__":
    main()
