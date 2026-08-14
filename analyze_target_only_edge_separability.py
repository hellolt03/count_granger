import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from count_granger_main import (
    apply_score_postprocess,
    evaluation_labels,
    load_config,
    metrics_at_with_postprocess,
    normal_bin_count,
    safe_name,
    score_diagnostics,
    select_threshold,
    split_indices,
    split_target_indices,
    threshold_curve,
)
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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fit_train_calibration_info(detector: CountGrangerDetector, *, reason: str) -> Dict[str, Any]:
    return {
        "normal_bins": None,
        "residual_center": float(detector.train_residual_center),
        "residual_scale": float(detector.train_residual_scale),
        "edge_center": float(detector.train_edge_center),
        "edge_scale": float(detector.train_edge_scale),
        "strategy": "fit_train",
        "reason": reason,
    }


def calibrate_for_target(
    detector: CountGrangerDetector,
    source_train_counts: Optional[np.ndarray],
    source_train_labels: Optional[np.ndarray],
    target_train_counts: np.ndarray,
    target_train_labels: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    run_cfg = config.get("run", {})
    min_target_normal_bins = int(run_cfg.get("min_target_calibration_normal_bins", 100))
    calibration_fallback = str(run_cfg.get("target_calibration_fallback", "pooled")).lower()
    target_normal_bins = normal_bin_count(target_train_labels)
    source_normal_bins = normal_bin_count(source_train_labels)
    reason = "target normal bins are sufficient"
    if target_normal_bins >= min_target_normal_bins:
        calibration = detector.calibrate_normal(target_train_counts, target_train_labels)
        strategy = "target"
    else:
        reason = f"target normal bins {target_normal_bins} < min_target_calibration_normal_bins {min_target_normal_bins}"
        calibration = None
        strategy = "fit_train"
        if calibration_fallback == "source" and source_train_counts is not None and source_normal_bins > 0:
            calibration = detector.calibrate_many_normal([source_train_counts], [source_train_labels])
            strategy = "source"
        elif calibration_fallback == "pooled":
            series = []
            labels = []
            if source_train_counts is not None and source_normal_bins > 0:
                series.append(source_train_counts)
                labels.append(source_train_labels)
            if target_normal_bins > 0 and source_normal_bins > 0:
                series.append(target_train_counts)
                labels.append(target_train_labels)
            if series:
                calibration = detector.calibrate_many_normal(series, labels)
                strategy = "pooled" if len(series) > 1 else "source"
        if calibration is None:
            calibration = fit_train_calibration_info(detector, reason=reason)
    calibration.update(
        {
            "strategy": strategy,
            "fallback": calibration_fallback,
            "reason": reason,
            "target_normal_bins": int(target_normal_bins),
            "source_normal_bins": int(source_normal_bins),
            "min_target_normal_bins": int(min_target_normal_bins),
        }
    )
    return calibration


def fit_mode_detector(
    mode: str,
    source: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Tuple[CountGrangerDetector, Dict[str, Any], Dict[str, np.ndarray]]:
    labels_all = evaluation_labels(target, config["split"])
    target_train_idx, target_val_idx, target_test_idx = split_target_indices(len(target["counts"]), labels_all, config["split"])
    source_train_idx, _, _ = split_indices(len(source["counts"]), config["split"], is_target=False)
    source_train_counts = source["counts"][source_train_idx]
    source_train_labels = source["labels"][source_train_idx]
    target_train_counts = target["counts"][target_train_idx]
    target_train_labels = target["labels"][target_train_idx]

    fit_scope = "split_train"
    if mode == "full":
        train_series = [source_train_counts, target_train_counts]
        train_labels = [source_train_labels, target_train_labels]
        calibration_source_counts = source_train_counts
        calibration_source_labels = source_train_labels
    elif mode == "target_only":
        train_series = [target_train_counts]
        train_labels = [target_train_labels]
        calibration_source_counts = None
        calibration_source_labels = None
    else:
        raise ValueError("mode must be one of {'full', 'target_only'}")

    detector = CountGrangerDetector(CountGrangerConfig(**config["granger"]))
    try:
        fit_info = detector.fit_many(train_series, train_labels)
    except ValueError as error:
        if mode != "target_only" or "No normal lagged samples" not in str(error):
            raise
        fit_scope = "all_target_timeline_normal_fallback"
        train_series = [target["counts"]]
        train_labels = [target["labels"]]
        fit_info = detector.fit_many(train_series, train_labels)
    calibration = calibrate_for_target(
        detector,
        calibration_source_counts,
        calibration_source_labels,
        target_train_counts,
        target_train_labels,
        config,
    )
    fit_info["target_normal_calibration"] = calibration
    fit_info["fit_mode"] = mode
    fit_info["fit_scope"] = fit_scope
    indices = {
        "target_train_idx": target_train_idx,
        "target_val_idx": target_val_idx,
        "target_test_idx": target_test_idx,
        "eval_labels_all": labels_all,
        "source_train_idx": source_train_idx,
    }
    return detector, fit_info, indices


def oracle_metrics(scores: np.ndarray, labels: np.ndarray, postprocess_cfg: Dict[str, Any]) -> Dict[str, Any]:
    processed = apply_score_postprocess(scores, postprocess_cfg)
    precision, recall, thresholds, f1 = threshold_curve(processed, labels)
    if len(thresholds) == 0:
        threshold = float(processed.max() + 1e-6) if len(processed) else 0.0
        return {"threshold": threshold, **metrics_at_with_postprocess(scores, labels, threshold, postprocess_cfg)}
    order = np.lexsort((-recall, -precision, -f1))
    idx = int(order[0])
    threshold = float(thresholds[idx])
    return {"threshold": threshold, **metrics_at_with_postprocess(scores, labels, threshold, postprocess_cfg)}


def distribution_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    result = dict(score_diagnostics(scores, labels))
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if len(normal) and len(anomaly):
        normal_p95 = float(np.percentile(normal, 95))
        anomaly_p05 = float(np.percentile(anomaly, 5))
        normal_p99 = float(np.percentile(normal, 99))
        anomaly_p01 = float(np.percentile(anomaly, 1))
        result.update(
            {
                "separation_gap_p05_p95": anomaly_p05 - normal_p95,
                "separation_gap_p01_p99": anomaly_p01 - normal_p99,
                "anomaly_below_normal_p95_rate": float(np.mean(anomaly <= normal_p95)),
                "normal_above_anomaly_p05_rate": float(np.mean(normal >= anomaly_p05)),
                "normal_p99": normal_p99,
                "anomaly_p01": anomaly_p01,
                "anomaly_ratio": float(labels.mean()) if len(labels) else 0.0,
            }
        )
    return result


def evaluate_edge(
    source_name: str,
    target_name: str,
    mode: str,
    detector: CountGrangerDetector,
    fit_info: Dict[str, Any],
    target: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    postprocess_cfg = config.get("postprocess", {})
    detect_cfg = dict(config["detect"])
    detect_cfg["score_component"] = "edge"
    detect_cfg["candidate_components"] = ["edge"]
    labels_all = indices["eval_labels_all"]
    val_idx = indices["target_val_idx"]
    test_idx = indices["target_test_idx"]
    val_labels = labels_all[val_idx]
    test_labels = labels_all[test_idx]
    val_edge = detector.score(target["counts"][val_idx])["edge"]
    test_edge = detector.score(target["counts"][test_idx])["edge"]
    threshold, val_metrics = select_threshold(apply_score_postprocess(val_edge, postprocess_cfg), val_labels, detect_cfg)
    test_metrics = metrics_at_with_postprocess(test_edge, test_labels, threshold, postprocess_cfg)
    test_oracle = oracle_metrics(test_edge, test_labels, postprocess_cfg)
    val_dist = distribution_metrics(val_edge, val_labels)
    test_dist = distribution_metrics(test_edge, test_labels)
    num_features = int(fit_info.get("num_features", 0) or 0)
    num_edges = int(fit_info.get("num_edges", 0) or 0)
    return {
        "source": source_name,
        "target": target_name,
        "mode": mode,
        "component": "edge",
        "num_features": num_features,
        "num_edges": num_edges,
        "edge_density": float(num_edges / max(num_features * num_features, 1)),
        "fit_scope": fit_info.get("fit_scope", ""),
        "calibration_strategy": fit_info.get("target_normal_calibration", {}).get("strategy"),
        "calibration_reason": fit_info.get("target_normal_calibration", {}).get("reason"),
        "target_val_bins": int(len(val_idx)),
        "target_test_bins": int(len(test_idx)),
        "target_val_anomaly_ratio": float(val_labels.mean()) if len(val_labels) else 0.0,
        "target_test_anomaly_ratio": float(test_labels.mean()) if len(test_labels) else 0.0,
        "threshold_from_val": float(threshold),
        "val_precision": safe_float(val_metrics.get("precision")),
        "val_recall": safe_float(val_metrics.get("recall")),
        "val_f1": safe_float(val_metrics.get("f1")),
        "val_constraint_satisfied": bool(val_metrics.get("constraint_satisfied", True)),
        "test_precision": safe_float(test_metrics.get("precision")),
        "test_recall": safe_float(test_metrics.get("recall")),
        "test_f1": safe_float(test_metrics.get("f1")),
        "test_roc_auc": safe_float(test_metrics.get("roc_auc")),
        "test_pr_auc": safe_float(test_metrics.get("pr_auc")),
        "test_oracle_precision": safe_float(test_oracle.get("precision")),
        "test_oracle_recall": safe_float(test_oracle.get("recall")),
        "test_oracle_f1": safe_float(test_oracle.get("f1")),
        "threshold_transfer_f1_gap": safe_float(test_oracle.get("f1")) - safe_float(test_metrics.get("f1")),
        **{f"val_{key}": value for key, value in val_dist.items()},
        **{f"test_{key}": value for key, value in test_dist.items()},
    }


def comparison_row(full: Dict[str, Any], target_only: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": full["source"],
        "target": full["target"],
        "full_f1": full["test_f1"],
        "target_only_f1": target_only["test_f1"],
        "delta_f1_target_only_minus_full": target_only["test_f1"] - full["test_f1"],
        "full_precision": full["test_precision"],
        "target_only_precision": target_only["test_precision"],
        "delta_precision_target_only_minus_full": target_only["test_precision"] - full["test_precision"],
        "full_recall": full["test_recall"],
        "target_only_recall": target_only["test_recall"],
        "delta_recall_target_only_minus_full": target_only["test_recall"] - full["test_recall"],
        "full_roc_auc": full["test_roc_auc"],
        "target_only_roc_auc": target_only["test_roc_auc"],
        "delta_roc_auc_target_only_minus_full": target_only["test_roc_auc"] - full["test_roc_auc"],
        "full_pr_auc": full["test_pr_auc"],
        "target_only_pr_auc": target_only["test_pr_auc"],
        "delta_pr_auc_target_only_minus_full": target_only["test_pr_auc"] - full["test_pr_auc"],
        "full_gap_p05_p95": full.get("test_separation_gap_p05_p95", float("nan")),
        "target_only_gap_p05_p95": target_only.get("test_separation_gap_p05_p95", float("nan")),
        "delta_gap_target_only_minus_full": target_only.get("test_separation_gap_p05_p95", float("nan")) - full.get("test_separation_gap_p05_p95", float("nan")),
        "full_anomaly_below_normal_p95_rate": full.get("test_anomaly_below_normal_p95_rate", float("nan")),
        "target_only_anomaly_below_normal_p95_rate": target_only.get("test_anomaly_below_normal_p95_rate", float("nan")),
        "delta_anomaly_below_rate_target_only_minus_full": target_only.get("test_anomaly_below_normal_p95_rate", float("nan")) - full.get("test_anomaly_below_normal_p95_rate", float("nan")),
        "full_num_features": full["num_features"],
        "target_only_num_features": target_only["num_features"],
        "full_num_edges": full["num_edges"],
        "target_only_num_edges": target_only["num_edges"],
        "target_only_better_f1": target_only["test_f1"] > full["test_f1"],
        "target_only_better_auc": target_only["test_roc_auc"] > full["test_roc_auc"],
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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "source",
        "target",
        "full_f1",
        "target_only_f1",
        "delta_f1_target_only_minus_full",
        "full_roc_auc",
        "target_only_roc_auc",
        "delta_roc_auc_target_only_minus_full",
        "full_pr_auc",
        "target_only_pr_auc",
        "delta_pr_auc_target_only_minus_full",
        "full_anomaly_below_normal_p95_rate",
        "target_only_anomaly_below_normal_p95_rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in headers) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full vs target-only edge separability for Count-Granger pairs")
    parser.add_argument("--datasets", nargs="*", help="Dataset mapping items: NAME=PATH")
    parser.add_argument("--config", default="configs/count_granger_config.yaml")
    parser.add_argument("--output_root", default="results/target_only_edge_separability")
    parser.add_argument("--cache_root", default="results/cache/count_granger")
    args = parser.parse_args()

    datasets = parse_dataset_items(args.datasets)
    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, stamp)
    ensure_dir(output_dir)
    mode_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    run_context: Dict[str, Any] = {"datasets": datasets, "config": args.config, "component": "edge"}

    for source_name, source_path in datasets.items():
        for target_name, target_path in datasets.items():
            if source_name == target_name:
                continue
            pair = f"{safe_name(source_path)}_to_{safe_name(target_path)}"
            cache_dir = os.path.join(args.cache_root, pair)
            print(f"[target-edge] processing {source_name} -> {target_name}; cache={cache_dir}", flush=True)
            paths = preprocess_count_series_pair(source_path, target_path, cache_dir, **config["preprocess"])
            source = load_count_series(paths["source"])
            target = load_count_series(paths["target"])
            pair_results: Dict[str, Dict[str, Any]] = {}
            pair_context: Dict[str, Any] = {}
            for mode in ["full", "target_only"]:
                detector, fit_info, indices = fit_mode_detector(mode, source, target, config)
                row = evaluate_edge(source_name, target_name, mode, detector, fit_info, target, indices, config)
                mode_rows.append(row)
                pair_results[mode] = row
                pair_context[mode] = fit_info
            comparison = comparison_row(pair_results["full"], pair_results["target_only"])
            comparison_rows.append(comparison)
            write_csv(os.path.join(output_dir, f"{source_name}_to_{target_name}_edge_modes.csv"), [pair_results["full"], pair_results["target_only"]])
            run_context[pair] = pair_context

    write_csv(os.path.join(output_dir, "target_only_edge_modes.csv"), mode_rows)
    write_csv(os.path.join(output_dir, "target_only_edge_comparison.csv"), comparison_rows)
    write_markdown(os.path.join(output_dir, "target_only_edge_comparison.md"), comparison_rows)
    with open(os.path.join(output_dir, "run_context.json"), "w", encoding="utf-8") as file:
        json.dump(run_context, file, ensure_ascii=False, indent=2)
    print(f"[target-edge] modes: {os.path.join(output_dir, 'target_only_edge_modes.csv')}", flush=True)
    print(f"[target-edge] comparison: {os.path.join(output_dir, 'target_only_edge_comparison.csv')}", flush=True)


if __name__ == "__main__":
    main()