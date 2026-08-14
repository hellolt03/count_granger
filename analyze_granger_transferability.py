
import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from count_granger_main import evaluation_labels, load_config, safe_name, split_indices, split_target_indices
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


def fit_graph(counts: np.ndarray, labels: np.ndarray, granger_cfg: Dict[str, Any]) -> Tuple[CountGrangerDetector, Dict[str, Any]]:
    detector = CountGrangerDetector(CountGrangerConfig(**granger_cfg))
    fit_info = detector.fit_many([counts], [labels.astype(np.int64)])
    return detector, fit_info


def normal_lag_window_count(labels: np.ndarray, max_lag: int) -> int:
    lag = int(max(1, max_lag))
    labels = labels.astype(np.int64)
    if len(labels) <= lag:
        return 0
    count = 0
    for index in range(lag, len(labels)):
        if np.all(labels[index - lag : index + 1] == 0):
            count += 1
    return count


def fit_graph_with_fallback(
    name: str,
    counts: np.ndarray,
    labels: np.ndarray,
    granger_cfg: Dict[str, Any],
) -> Tuple[CountGrangerDetector, Dict[str, Any], Dict[str, Any]]:
    normal_bins = int((labels == 0).sum())
    normal_lag_windows = normal_lag_window_count(labels, int(granger_cfg.get("max_lag", 5)))
    try:
        detector, fit_info = fit_graph(counts, labels, granger_cfg)
        fit_mode = "normal_only"
    except ValueError as error:
        if "No normal lagged samples" not in str(error):
            raise
        relaxed_labels = np.zeros(len(labels), dtype=np.int64)
        detector, fit_info = fit_graph(counts, relaxed_labels, granger_cfg)
        fit_mode = "relaxed_all_windows"
    context = {
        "name": name,
        "fit_mode": fit_mode,
        "bins": int(len(labels)),
        "normal_bins": normal_bins,
        "anomaly_bins": int((labels != 0).sum()),
        "normal_lag_windows": int(normal_lag_windows),
    }
    return detector, fit_info, context


def edge_set(adjacency: np.ndarray, threshold: float) -> set[Tuple[int, int]]:
    rows, cols = np.where(adjacency > threshold)
    return set(zip(rows.tolist(), cols.tolist()))


def topk_edge_set(adjacency: np.ndarray, k: int) -> set[Tuple[int, int]]:
    result: set[Tuple[int, int]] = set()
    if adjacency.size == 0:
        return result
    for target_idx in range(adjacency.shape[0]):
        row = adjacency[target_idx].copy()
        row[target_idx] = 0.0
        positive = np.where(row > 0)[0]
        if len(positive) == 0:
            continue
        top = positive[np.argsort(-row[positive])[: max(1, int(k))]]
        for source_idx in top:
            result.add((int(target_idx), int(source_idx)))
    return result


def jaccard(left: set, right: set) -> float:
    union = len(left | right)
    if union == 0:
        return 1.0
    return len(left & right) / union


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(left, right).statistic)
        return float(spearmanr(left, right).statistic)
    except Exception:
        return float("nan")


def compare_graphs(
    source_detector: CountGrangerDetector,
    target_detector: CountGrangerDetector,
    *,
    edge_threshold: float,
    top_k: int,
) -> Dict[str, Any]:
    source_features = source_detector.selected_features.astype(int).tolist() if source_detector.selected_features is not None else []
    target_features = target_detector.selected_features.astype(int).tolist() if target_detector.selected_features is not None else []
    source_node_set = set(source_features)
    target_node_set = set(target_features)
    shared_nodes = sorted(source_node_set & target_node_set)
    source_local = {feature: idx for idx, feature in enumerate(source_features)}
    target_local = {feature: idx for idx, feature in enumerate(target_features)}
    source_adj = source_detector.adjacency if source_detector.adjacency is not None else np.zeros((0, 0), dtype=np.float64)
    target_adj = target_detector.adjacency if target_detector.adjacency is not None else np.zeros((0, 0), dtype=np.float64)

    source_edges_local = edge_set(source_adj, edge_threshold)
    target_edges_local = edge_set(target_adj, edge_threshold)
    source_edges_global = {(source_features[i], source_features[j]) for i, j in source_edges_local}
    target_edges_global = {(target_features[i], target_features[j]) for i, j in target_edges_local}
    source_topk_global = {(source_features[i], source_features[j]) for i, j in topk_edge_set(source_adj, top_k)}
    target_topk_global = {(target_features[i], target_features[j]) for i, j in topk_edge_set(target_adj, top_k)}

    shared_edges = sorted(source_edges_global & target_edges_global)
    source_weights = []
    target_weights = []
    for edge in shared_edges:
        child_feature, parent_feature = edge
        source_weights.append(float(source_adj[source_local[child_feature], source_local[parent_feature]]))
        target_weights.append(float(target_adj[target_local[child_feature], target_local[parent_feature]]))
    source_weights_arr = np.asarray(source_weights, dtype=np.float64)
    target_weights_arr = np.asarray(target_weights, dtype=np.float64)

    reversed_overlap = sum(1 for child, parent in source_edges_global if (parent, child) in target_edges_global)
    shared_or_reversed = len(shared_edges) + reversed_overlap
    direction_consistency = len(shared_edges) / shared_or_reversed if shared_or_reversed > 0 else 0.0

    parent_jaccards = []
    for feature in shared_nodes:
        source_idx = source_local[feature]
        target_idx = target_local[feature]
        source_row = source_adj[source_idx].copy()
        target_row = target_adj[target_idx].copy()
        if len(source_row):
            source_row[source_idx] = 0.0
        if len(target_row):
            target_row[target_idx] = 0.0
        source_positive = np.where(source_row > 0)[0]
        target_positive = np.where(target_row > 0)[0]
        source_parents = {source_features[i] for i in source_positive[np.argsort(-source_row[source_positive])[:top_k]]}
        target_parents = {target_features[i] for i in target_positive[np.argsort(-target_row[target_positive])[:top_k]]}
        parent_jaccards.append(jaccard(source_parents, target_parents))

    node_jaccard = jaccard(source_node_set, target_node_set)
    topk_jaccard = jaccard(source_topk_global, target_topk_global)
    spearman = safe_corr(source_weights_arr, target_weights_arr, "spearman")
    pearson = safe_corr(source_weights_arr, target_weights_arr, "pearson")
    transferability = (
        0.3 * node_jaccard
        + 0.3 * topk_jaccard
        + 0.2 * (0.0 if math.isnan(spearman) else max(0.0, spearman))
        + 0.2 * direction_consistency
    )
    return {
        "source_num_nodes": len(source_node_set),
        "target_num_nodes": len(target_node_set),
        "shared_nodes": len(shared_nodes),
        "node_jaccard": node_jaccard,
        "source_num_edges": len(source_edges_global),
        "target_num_edges": len(target_edges_global),
        "shared_edges": len(shared_edges),
        "edge_jaccard": jaccard(source_edges_global, target_edges_global),
        "topk_edge_jaccard": topk_jaccard,
        "mean_topk_parent_jaccard": float(np.mean(parent_jaccards)) if parent_jaccards else float("nan"),
        "shared_edge_weight_pearson": pearson,
        "shared_edge_weight_spearman": spearman,
        "reversed_edge_overlap": reversed_overlap,
        "direction_consistency": direction_consistency,
        "transferability_score": transferability,
        "source_edges_global": source_edges_global,
        "target_edges_global": target_edges_global,
    }


def write_csv(path: str, rows: List[Dict[str, Any]], exclude: Optional[set[str]] = None) -> None:
    exclude = exclude or set()
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key in exclude:
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "source", "target", "node_jaccard", "edge_jaccard", "topk_edge_jaccard",
        "mean_topk_parent_jaccard", "shared_edge_weight_spearman", "direction_consistency", "transferability_score",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                value = "nan" if math.isnan(value) else f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze source/target Granger graph transferability for all directed pairs")
    parser.add_argument("--datasets", nargs="*", help="Dataset mapping items: NAME=PATH")
    parser.add_argument("--config", default="configs/count_granger_config.yaml")
    parser.add_argument("--output_root", default="results/granger_transferability")
    parser.add_argument("--cache_root", default="results/cache/count_granger")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    datasets = parse_dataset_items(args.datasets)
    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, stamp)
    ensure_dir(output_dir)
    rows: List[Dict[str, Any]] = []

    for source_name, source_path in datasets.items():
        for target_name, target_path in datasets.items():
            if source_name == target_name:
                continue
            pair = f"{safe_name(source_path)}_to_{safe_name(target_path)}"
            cache_dir = os.path.join(args.cache_root, pair)
            print(f"[transfer] processing {source_name} -> {target_name}; cache={cache_dir}", flush=True)
            paths = preprocess_count_series_pair(source_path, target_path, cache_dir, **config["preprocess"])
            source = load_count_series(paths["source"])
            target = load_count_series(paths["target"])
            source_train_idx, _, _ = split_indices(len(source["counts"]), config["split"], is_target=False)
            target_eval_labels = evaluation_labels(target, config["split"])
            target_train_idx, _, _ = split_target_indices(len(target["counts"]), target_eval_labels, config["split"])
            source_detector, source_fit, source_context = fit_graph_with_fallback(source_name, source["counts"], source["labels"], config["granger"])
            target_detector, target_fit, target_context = fit_graph_with_fallback(target_name, target["counts"], target["labels"], config["granger"])
            metrics = compare_graphs(source_detector, target_detector, edge_threshold=float(config["granger"].get("edge_threshold", 0.01)), top_k=args.top_k)
            row = {
                "source": source_name,
                "target": target_name,
                "pair": pair,
                "top_k": args.top_k,
                "source_split_train_bins": len(source_train_idx),
                "target_split_train_bins": len(target_train_idx),
                "source_graph_fit_mode": source_context["fit_mode"],
                "target_graph_fit_mode": target_context["fit_mode"],
                "source_normal_bins": source_context["normal_bins"],
                "target_normal_bins": target_context["normal_bins"],
                "source_normal_lag_windows": source_context["normal_lag_windows"],
                "target_normal_lag_windows": target_context["normal_lag_windows"],
                "source_fit_features": source_fit.get("num_features"),
                "target_fit_features": target_fit.get("num_features"),
                **{key: value for key, value in metrics.items() if not key.endswith("_global")},
            }
            rows.append(row)
            edge_rows = []
            for child, parent in sorted(metrics["source_edges_global"] | metrics["target_edges_global"]):
                edge_rows.append({
                    "child_feature": child,
                    "parent_feature": parent,
                    "in_source_graph": int((child, parent) in metrics["source_edges_global"]),
                    "in_target_graph": int((child, parent) in metrics["target_edges_global"]),
                })
            write_csv(os.path.join(output_dir, f"{source_name}_to_{target_name}_edges.csv"), edge_rows)

    write_csv(os.path.join(output_dir, "graph_transfer_summary.csv"), rows)
    write_markdown(os.path.join(output_dir, "graph_transfer_summary.md"), rows)
    with open(os.path.join(output_dir, "run_context.json"), "w", encoding="utf-8") as file:
        json.dump({"datasets": datasets, "config": args.config, "top_k": args.top_k}, file, ensure_ascii=False, indent=2)
    print(f"[transfer] summary: {os.path.join(output_dir, 'graph_transfer_summary.csv')}", flush=True)
    print(f"[transfer] markdown: {os.path.join(output_dir, 'graph_transfer_summary.md')}", flush=True)


if __name__ == "__main__":
    main()

