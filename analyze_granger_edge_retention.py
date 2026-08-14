import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from count_granger_main import load_config, safe_name
from count_granger_model import CountGrangerConfig, CountGrangerDetector
from count_series_preprocessing import load_count_series, preprocess_count_series_pair

DEFAULT_DATASETS = {
    "BGL": "dataset/BGL/BGL.log",
    "Thunderbird": "dataset/Thunderbird/Thunderbird.log",
    "Spirit": "dataset/Spirit/Spirit1G.log",
}


def parse_dataset_items(items: Optional[List[str]]) -> Dict[str, str]:
    if not items:
        return dict(DEFAULT_DATASETS)
    datasets: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Dataset item must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        datasets[name.strip()] = path.strip()
    return datasets


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normal_lag_window_count(labels: np.ndarray, max_lag: int) -> int:
    lag = int(max(1, max_lag))
    labels = labels.astype(np.int64)
    if len(labels) <= lag:
        return 0
    return int(sum(np.all(labels[index - lag : index + 1] == 0) for index in range(lag, len(labels))))


def fit_normal_graph(
    name: str,
    counts: np.ndarray,
    labels: np.ndarray,
    granger_cfg: Dict[str, Any],
) -> Tuple[CountGrangerDetector, Dict[str, Any]]:
    detector_cfg = dict(granger_cfg)
    detector_cfg["edge_selection"] = "none"
    detector = CountGrangerDetector(CountGrangerConfig(**detector_cfg))
    fit_mode = "normal_only"
    try:
        fit_info = detector.fit_many([counts], [labels.astype(np.int64)])
    except ValueError as error:
        if "No normal lagged samples" not in str(error):
            raise
        relaxed_labels = np.zeros(len(labels), dtype=np.int64)
        fit_info = detector.fit_many([counts], [relaxed_labels])
        fit_mode = "relaxed_all_windows"
    context = {
        "dataset": name,
        "fit_mode": fit_mode,
        "bins": int(len(labels)),
        "normal_bins": int((labels == 0).sum()),
        "anomaly_bins": int((labels != 0).sum()),
        "normal_lag_windows": normal_lag_window_count(labels, int(detector_cfg.get("max_lag", 5))),
    }
    return detector, {**fit_info, **context}


def ranked_edges(detector: CountGrangerDetector, threshold: float) -> List[Dict[str, Any]]:
    features = detector.selected_features.astype(int).tolist() if detector.selected_features is not None else []
    adjacency = detector.adjacency if detector.adjacency is not None else np.zeros((0, 0), dtype=np.float64)
    ranked: List[Dict[str, Any]] = []
    for child_index, parent_index in zip(*np.where(adjacency > threshold)):
        if child_index == parent_index:
            continue
        ranked.append(
            {
                "child_feature": int(features[int(child_index)]),
                "parent_feature": int(features[int(parent_index)]),
                "weight": float(adjacency[int(child_index), int(parent_index)]),
            }
        )
    ranked.sort(key=lambda item: (-item["weight"], item["child_feature"], item["parent_feature"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return ranked


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def retention_for_k(
    source_edges: List[Dict[str, Any]],
    target_edges: List[Dict[str, Any]],
    source_features: Iterable[int],
    target_features: Iterable[int],
    k: int,
) -> Dict[str, Any]:
    source_top = source_edges[: max(1, int(k))]
    target_by_edge = {
        (edge["child_feature"], edge["parent_feature"]): edge for edge in target_edges
    }
    target_feature_set = set(target_features)
    source_feature_set = set(source_features)
    target_edge_count = len(target_edges)
    endpoint_eligible = [
        edge
        for edge in source_top
        if edge["child_feature"] in target_feature_set and edge["parent_feature"] in target_feature_set
    ]
    retained = [
        edge
        for edge in source_top
        if (edge["child_feature"], edge["parent_feature"]) in target_by_edge
    ]
    retained_endpoint = [
        edge
        for edge in endpoint_eligible
        if (edge["child_feature"], edge["parent_feature"]) in target_by_edge
    ]
    target_ranks = [
        target_by_edge[(edge["child_feature"], edge["parent_feature"])]["rank"]
        for edge in retained
    ]
    target_rank_percentiles = [
        1.0 - (rank - 1) / max(1, target_edge_count - 1)
        for rank in target_ranks
    ]
    source_weight_total = float(sum(edge["weight"] for edge in source_top))
    retained_source_weight = float(sum(edge["weight"] for edge in retained))
    retained_target_weight = float(
        sum(target_by_edge[(edge["child_feature"], edge["parent_feature"])]["weight"] for edge in retained)
    )
    return {
        "k": int(k),
        "source_top_k_count": len(source_top),
        "source_endpoint_eligible_count": len(endpoint_eligible),
        "retained_edge_count": len(retained),
        "retained_endpoint_edge_count": len(retained_endpoint),
        "raw_retention": safe_ratio(len(retained), len(source_top)),
        "endpoint_conditioned_retention": safe_ratio(len(retained_endpoint), len(endpoint_eligible)),
        "weighted_retention": safe_ratio(retained_source_weight, source_weight_total),
        "target_weight_over_source_weight": safe_ratio(retained_target_weight, retained_source_weight),
        "mean_target_rank": float(np.mean(target_ranks)) if target_ranks else float("nan"),
        "mean_target_rank_percentile": float(np.mean(target_rank_percentiles)) if target_rank_percentiles else float("nan"),
        "source_node_count": len(source_feature_set),
        "target_node_count": len(target_feature_set),
        "shared_node_count": len(source_feature_set & target_feature_set),
        "target_edge_count": target_edge_count,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "source", "target", "k", "raw_retention", "endpoint_conditioned_retention",
        "weighted_retention", "target_weight_over_source_weight", "mean_target_rank_percentile",
        "source_endpoint_eligible_count", "retained_edge_count", "retained_endpoint_edge_count",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(key, "")) for key in headers) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Top-K Granger edge retention from source normal data to target normal data")
    parser.add_argument("--datasets", nargs="*", help="Dataset mapping items: NAME=PATH")
    parser.add_argument("--config", default="configs/count_granger_config.yaml")
    parser.add_argument("--output_root", default="results/granger_edge_retention")
    parser.add_argument("--cache_root", default="results/cache/count_granger")
    parser.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100, 200])
    args = parser.parse_args()

    datasets = parse_dataset_items(args.datasets)
    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, stamp)
    ensure_dir(output_dir)
    all_rows: List[Dict[str, Any]] = []
    run_context: Dict[str, Any] = {"datasets": datasets, "config": args.config, "ks": args.ks, "graph_edge_selection": "none"}

    for source_name, source_path in datasets.items():
        for target_name, target_path in datasets.items():
            if source_name == target_name:
                continue
            pair = f"{safe_name(source_path)}_to_{safe_name(target_path)}"
            cache_dir = os.path.join(args.cache_root, pair)
            print(f"[retention] processing {source_name} -> {target_name}; cache={cache_dir}", flush=True)
            paths = preprocess_count_series_pair(source_path, target_path, cache_dir, **config["preprocess"])
            source = load_count_series(paths["source"])
            target = load_count_series(paths["target"])
            source_detector, source_fit = fit_normal_graph(source_name, source["counts"], source["labels"], config["granger"])
            target_detector, target_fit = fit_normal_graph(target_name, target["counts"], target["labels"], config["granger"])
            source_features = source_detector.selected_features.astype(int).tolist()
            target_features = target_detector.selected_features.astype(int).tolist()
            threshold = float(config["granger"].get("edge_threshold", 0.01))
            source_edges = ranked_edges(source_detector, threshold)
            target_edges = ranked_edges(target_detector, threshold)
            detail_rows: List[Dict[str, Any]] = []
            for k in args.ks:
                metrics = retention_for_k(source_edges, target_edges, source_features, target_features, k)
                row = {
                    "source": source_name,
                    "target": target_name,
                    "pair": pair,
                    "source_fit_mode": source_fit["fit_mode"],
                    "target_fit_mode": target_fit["fit_mode"],
                    "source_normal_bins": source_fit["normal_bins"],
                    "target_normal_bins": target_fit["normal_bins"],
                    "source_normal_lag_windows": source_fit["normal_lag_windows"],
                    "target_normal_lag_windows": target_fit["normal_lag_windows"],
                    "source_graph_nodes": len(source_features),
                    "target_graph_nodes": len(target_features),
                    "source_graph_edges": len(source_edges),
                    "target_graph_edges": len(target_edges),
                    "edge_threshold": threshold,
                    **metrics,
                }
                all_rows.append(row)
                detail_rows.append(row)
            write_csv(os.path.join(output_dir, f"{source_name}_to_{target_name}_retention.csv"), detail_rows)
            edge_rows = []
            target_by_edge = {(edge["child_feature"], edge["parent_feature"]): edge for edge in target_edges}
            target_feature_set = set(target_features)
            for edge in source_edges[: max(args.ks)]:
                key = (edge["child_feature"], edge["parent_feature"])
                target_match = target_by_edge.get(key)
                edge_rows.append({
                    "source_rank": edge["rank"],
                    "child_feature": edge["child_feature"],
                    "parent_feature": edge["parent_feature"],
                    "source_weight": edge["weight"],
                    "target_present": int(target_match is not None),
                    "target_rank": "" if target_match is None else target_match["rank"],
                    "target_weight": "" if target_match is None else target_match["weight"],
                    "endpoints_shared": int(edge["child_feature"] in target_feature_set and edge["parent_feature"] in target_feature_set),
                })
            write_csv(os.path.join(output_dir, f"{source_name}_to_{target_name}_top_edges.csv"), edge_rows)
            run_context[pair] = {
                "source_fit": source_fit,
                "target_fit": target_fit,
                "source_graph_edges": len(source_edges),
                "target_graph_edges": len(target_edges),
            }

    write_csv(os.path.join(output_dir, "edge_retention_summary.csv"), all_rows)
    write_markdown(os.path.join(output_dir, "edge_retention_summary.md"), all_rows)
    with open(os.path.join(output_dir, "run_context.json"), "w", encoding="utf-8") as file:
        json.dump(run_context, file, ensure_ascii=False, indent=2)
    print(f"[retention] summary: {os.path.join(output_dir, 'edge_retention_summary.csv')}", flush=True)
    print(f"[retention] markdown: {os.path.join(output_dir, 'edge_retention_summary.md')}", flush=True)


if __name__ == "__main__":
    main()
