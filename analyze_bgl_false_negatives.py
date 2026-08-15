import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from count_granger_main import evaluation_labels, load_config, split_target_indices


def robust_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0) / 0.6745
    std = values.std(axis=0)
    scale = np.maximum.reduce([mad, std, np.full(values.shape[1], 1e-6)])
    return center, scale


def read_scores(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result: Dict[str, List[float]] = {key: [] for key in rows[0].keys()} if rows else {}
    for row in rows:
        for key, value in row.items():
            if key in {"label", "prediction", "bin"}:
                result[key].append(int(value))
            elif key == "selected_component":
                continue
            else:
                result[key].append(float(value))
    return {key: np.asarray(value) for key, value in result.items()}


def summarize_scores(scores: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = scores[mask]
    if len(values) == 0:
        return {"count": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def boundary_summary(global_indices: np.ndarray, fn_global: np.ndarray) -> Dict[str, Any]:
    anomaly_runs = contiguous_runs(global_indices)
    if len(fn_global) == 0:
        return {"fn_count": 0, "anomaly_runs": len(anomaly_runs)}
    distances = []
    run_lengths = []
    for idx in fn_global:
        containing = [(start, end) for start, end in anomaly_runs if start <= idx <= end]
        if not containing:
            continue
        start, end = containing[0]
        distances.append(min(int(idx - start), int(end - idx)))
        run_lengths.append(int(end - start + 1))
    distances_arr = np.asarray(distances, dtype=np.float64)
    run_lengths_arr = np.asarray(run_lengths, dtype=np.float64)
    return {
        "fn_count": int(len(fn_global)),
        "anomaly_runs": int(len(anomaly_runs)),
        "fn_boundary_distance_p50": float(np.percentile(distances_arr, 50)) if len(distances_arr) else None,
        "fn_boundary_distance_p75": float(np.percentile(distances_arr, 75)) if len(distances_arr) else None,
        "fn_within_1_bin_boundary": float(np.mean(distances_arr <= 1)) if len(distances_arr) else None,
        "fn_within_2_bin_boundary": float(np.mean(distances_arr <= 2)) if len(distances_arr) else None,
        "fn_run_length_p50": float(np.percentile(run_lengths_arr, 50)) if len(run_lengths_arr) else None,
    }


def neighbor_detection_summary(test_idx: np.ndarray, preds: np.ndarray, fn: np.ndarray, radius: int = 1) -> Dict[str, Any]:
    fn_global = test_idx[fn]
    if len(fn_global) == 0:
        return {"fn_count": 0, "radius": int(radius)}
    pred_by_global = {int(idx): int(pred) for idx, pred in zip(test_idx, preds)}
    covered = 0
    for idx in fn_global:
        idx = int(idx)
        has_neighbor_hit = any(pred_by_global.get(idx + offset, 0) == 1 for offset in range(-radius, radius + 1) if offset != 0)
        covered += int(has_neighbor_hit)
    return {
        "fn_count": int(len(fn_global)),
        "radius": int(radius),
        "fn_with_predicted_neighbor": int(covered),
        "fn_with_predicted_neighbor_rate": float(covered / len(fn_global)),
    }


def high_component_rate(scores: np.ndarray, normal_mask: np.ndarray, query_mask: np.ndarray) -> Dict[str, Any]:
    normal = scores[normal_mask]
    query = scores[query_mask]
    if len(normal) == 0 or len(query) == 0:
        return {"query_count": int(len(query))}
    p95 = float(np.percentile(normal, 95))
    p99 = float(np.percentile(normal, 99))
    return {
        "query_count": int(len(query)),
        "normal_p95": p95,
        "normal_p99": p99,
        "above_normal_p95": int(np.sum(query > p95)),
        "above_normal_p95_rate": float(np.mean(query > p95)),
        "above_normal_p99": int(np.sum(query > p99)),
        "above_normal_p99_rate": float(np.mean(query > p99)),
    }


def selected_feature_coverage_summary(
    counts: np.ndarray,
    test_idx: np.ndarray,
    selected_features: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    test_counts = counts[test_idx].astype(np.float64)
    total_counts = test_counts.sum(axis=1)
    selected_counts = test_counts[:, selected_features].sum(axis=1)
    active_selected = (test_counts[:, selected_features] > 0).sum(axis=1).astype(np.float64)
    coverage = np.divide(selected_counts, np.maximum(total_counts, 1.0))
    result: Dict[str, Any] = {}
    for name, mask in masks.items():
        result[name] = {
            "total_counts": summarize_scores(total_counts, mask),
            "selected_counts": summarize_scores(selected_counts, mask),
            "active_selected_features": summarize_scores(active_selected, mask),
            "selected_count_coverage": summarize_scores(coverage, mask),
        }
    return result


def top_feature_diagnostics(
    counts: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mask: np.ndarray,
    selected_features: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    normal_train_idx = train_idx[labels[train_idx] == 0]
    selected_counts = counts[:, selected_features].astype(np.float64)
    values = np.log1p(selected_counts)
    center, scale = robust_stats(values[normal_train_idx])
    z_scores = np.abs(values[test_idx] - center) / scale
    selected_rows = z_scores[mask]
    selected_counts_rows = selected_counts[test_idx][mask]
    if len(selected_rows) == 0:
        return []
    top_k = min(max(1, top_k), selected_rows.shape[1])
    feature_hits: Dict[int, Dict[str, float]] = {}
    top_indices = np.argpartition(selected_rows, -top_k, axis=1)[:, -top_k:]
    for row_idx, feature_positions in enumerate(top_indices):
        ordered = feature_positions[np.argsort(selected_rows[row_idx, feature_positions])[::-1]]
        for rank, feature_pos in enumerate(ordered, start=1):
            feature_id = int(selected_features[feature_pos])
            item = feature_hits.setdefault(
                feature_id,
                {"feature": float(feature_id), "hits": 0.0, "rank_sum": 0.0, "z_sum": 0.0, "count_sum": 0.0},
            )
            item["hits"] += 1.0
            item["rank_sum"] += float(rank)
            item["z_sum"] += float(selected_rows[row_idx, feature_pos])
            item["count_sum"] += float(selected_counts_rows[row_idx, feature_pos])
    rows = []
    for item in feature_hits.values():
        hits = max(item["hits"], 1.0)
        rows.append(
            {
                "feature": int(item["feature"]),
                "hits": int(item["hits"]),
                "hit_rate": float(item["hits"] / len(selected_rows)),
                "avg_rank": float(item["rank_sum"] / hits),
                "avg_z": float(item["z_sum"] / hits),
                "avg_count": float(item["count_sum"] / hits),
            }
        )
    return sorted(rows, key=lambda row: (-row["hits"], row["avg_rank"]))[:25]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(run_dir: str, cache_dir: str, output_dir: str, top_k: int) -> Dict[str, Any]:
    score_dir = os.path.join(run_dir, "target_scores")
    summary = load_json(os.path.join(score_dir, "summary.json"))
    model = load_json(os.path.join(score_dir, "count_granger_model.json"))
    config = summary["config"]
    scores = read_scores(os.path.join(score_dir, "target_test_scores.csv"))
    pair_prefix = os.path.basename(cache_dir)
    target_series_path = os.path.join(cache_dir, f"{pair_prefix}_target_count_series.npz")
    target = dict(np.load(target_series_path, allow_pickle=True))
    eval_labels = evaluation_labels(target, config["split"])
    train_idx, _, test_idx = split_target_indices(len(target["counts"]), eval_labels, config["split"])
    labels = scores["label"].astype(np.int64)
    preds = scores["prediction"].astype(np.int64)
    if len(test_idx) != len(labels):
        raise ValueError(f"test score length {len(labels)} does not match split length {len(test_idx)}")

    tp = (labels == 1) & (preds == 1)
    fn = (labels == 1) & (preds == 0)
    fp = (labels == 0) & (preds == 1)
    tn = (labels == 0) & (preds == 0)
    anomaly = labels == 1
    normal = labels == 0
    threshold = float(summary["threshold"])
    selected = str(summary["selected_score_component"])
    selected_scores = scores["selected_score"]

    selected_features = np.asarray(model["selected_features"], dtype=np.int64)
    fn_top_features = top_feature_diagnostics(
        target["counts"],
        eval_labels,
        train_idx,
        test_idx,
        fn,
        selected_features,
        top_k,
    )
    tp_top_features = top_feature_diagnostics(
        target["counts"],
        eval_labels,
        train_idx,
        test_idx,
        tp,
        selected_features,
        top_k,
    )

    anomaly_global = np.flatnonzero(eval_labels == 1)
    fn_global = test_idx[fn]
    anomaly_counts = target.get("anomaly_counts", eval_labels)
    test_anomaly_counts = anomaly_counts[test_idx]
    payload: Dict[str, Any] = {
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "selected_component": selected,
        "threshold": threshold,
        "confusion": {
            "tp": int(tp.sum()),
            "fn": int(fn.sum()),
            "fp": int(fp.sum()),
            "tn": int(tn.sum()),
            "precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
            "recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
        },
        "selected_score_summary": {
            "normal": summarize_scores(selected_scores, normal),
            "anomaly": summarize_scores(selected_scores, anomaly),
            "tp": summarize_scores(selected_scores, tp),
            "fn": summarize_scores(selected_scores, fn),
            "fp": summarize_scores(selected_scores, fp),
            "tn": summarize_scores(selected_scores, tn),
        },
        "component_summary": {},
        "fn_boundary_summary": boundary_summary(anomaly_global, fn_global),
        "fn_neighbor_detection_radius1": neighbor_detection_summary(test_idx, preds, fn, radius=1),
        "fn_neighbor_detection_radius2": neighbor_detection_summary(test_idx, preds, fn, radius=2),
        "anomaly_line_count_summary": {
            "tp": summarize_scores(test_anomaly_counts.astype(np.float64), tp),
            "fn": summarize_scores(test_anomaly_counts.astype(np.float64), fn),
        },
        "selected_feature_coverage_summary": selected_feature_coverage_summary(
            target["counts"],
            test_idx,
            selected_features,
            {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        ),
        "fn_top_features": fn_top_features,
        "tp_top_features": tp_top_features,
    }
    for component in ["residual", "edge", "level", "combined"]:
        if component in scores:
            payload["component_summary"][component] = {
                "normal": summarize_scores(scores[component], normal),
                "anomaly": summarize_scores(scores[component], anomaly),
                "tp": summarize_scores(scores[component], tp),
                "fn": summarize_scores(scores[component], fn),
                "fp": summarize_scores(scores[component], fp),
                "tn": summarize_scores(scores[component], tn),
            }
    payload["fn_high_component_rates"] = {
        component: high_component_rate(scores[component], normal, fn)
        for component in ["residual", "edge", "level", "combined"]
        if component in scores
    }
    os.makedirs(output_dir, exist_ok=True)
    save_json(os.path.join(output_dir, "bgl_false_negative_diagnostics.json"), payload)
    save_csv(os.path.join(output_dir, "fn_top_features.csv"), fn_top_features)
    save_csv(os.path.join(output_dir, "tp_top_features.csv"), tp_top_features)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose false negatives for BGL target Count-Granger runs")
    parser.add_argument("--run_dir", required=True, help="Run directory containing target_scores outputs")
    parser.add_argument("--cache_dir", required=True, help="Count-series cache directory for the same source-target pair")
    parser.add_argument("--output_dir", required=True, help="Directory for diagnostic outputs")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    payload = analyze_run(args.run_dir, args.cache_dir, args.output_dir, args.top_k)
    print(json.dumps({
        "selected_component": payload["selected_component"],
        "threshold": payload["threshold"],
        "confusion": payload["confusion"],
        "fn_boundary_summary": payload["fn_boundary_summary"],
        "fn_neighbor_detection_radius1": payload["fn_neighbor_detection_radius1"],
        "anomaly_line_count_summary": payload["anomaly_line_count_summary"],
        "selected_feature_coverage_summary": payload["selected_feature_coverage_summary"],
        "fn_high_component_rates": payload["fn_high_component_rates"],
        "fn_selected_score": payload["selected_score_summary"]["fn"],
        "fn_edge": payload["component_summary"].get("edge", {}).get("fn", {}),
        "fn_level": payload["component_summary"].get("level", {}).get("fn", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
