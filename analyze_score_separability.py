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
    select_score_component,
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


def fit_detector_for_pair(source: Dict[str, np.ndarray], target: Dict[str, np.ndarray], config: Dict[str, Any]) -> Tuple[CountGrangerDetector, Dict[str, Any], Dict[str, np.ndarray]]:
    eval_labels_all = evaluation_labels(target, config["split"])
    target_train_idx, target_val_idx, target_test_idx = split_target_indices(len(target["counts"]), eval_labels_all, config["split"])
    train_series: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    source_train_idx, _, _ = split_indices(len(source["counts"]), config["split"], is_target=False)
    source_train_counts = source["counts"][source_train_idx]
    source_train_labels = source["labels"][source_train_idx]
    train_series.append(source_train_counts)
    train_labels.append(source_train_labels)
    target_train_counts = target["counts"][target_train_idx]
    target_train_labels = target["labels"][target_train_idx]
    use_target_train = bool(config.get("run", {}).get("use_target_train", True))
    if use_target_train:
        train_series.append(target_train_counts)
        train_labels.append(target_train_labels)

    detector = CountGrangerDetector(CountGrangerConfig(**config["granger"]))
    fit_info = detector.fit_many(train_series, train_labels)

    run_cfg = config.get("run", {})
    min_target_normal_bins = int(run_cfg.get("min_target_calibration_normal_bins", 100))
    calibration_fallback = str(run_cfg.get("target_calibration_fallback", "pooled")).lower()
    target_normal_bins = normal_bin_count(target_train_labels)
    source_normal_bins = normal_bin_count(source_train_labels)
    calibration_reason = "target normal bins are sufficient"
    if target_normal_bins >= min_target_normal_bins:
        target_calibration = detector.calibrate_normal(target_train_counts, target_train_labels)
        applied_calibration = "target"
    else:
        calibration_reason = f"target normal bins {target_normal_bins} < min_target_calibration_normal_bins {min_target_normal_bins}"
        target_calibration = None
        applied_calibration = "fit_train"
        if calibration_fallback == "source" and source_normal_bins > 0:
            target_calibration = detector.calibrate_many_normal([source_train_counts], [source_train_labels])
            applied_calibration = "source"
        elif calibration_fallback == "pooled":
            calibration_series = []
            calibration_labels = []
            if source_normal_bins > 0:
                calibration_series.append(source_train_counts)
                calibration_labels.append(source_train_labels)
            if target_normal_bins > 0 and source_normal_bins > 0:
                calibration_series.append(target_train_counts)
                calibration_labels.append(target_train_labels)
            if calibration_series:
                target_calibration = detector.calibrate_many_normal(calibration_series, calibration_labels)
                applied_calibration = "pooled" if len(calibration_series) > 1 else "source"
        if target_calibration is None:
            target_calibration = fit_train_calibration_info(detector, reason=calibration_reason)
    target_calibration.update(
        {
            "strategy": applied_calibration,
            "fallback": calibration_fallback,
            "reason": calibration_reason,
            "target_normal_bins": int(target_normal_bins),
            "source_normal_bins": int(source_normal_bins),
            "min_target_normal_bins": int(min_target_normal_bins),
        }
    )
    fit_info["target_normal_calibration"] = target_calibration
    indices = {
        "target_train_idx": target_train_idx,
        "target_val_idx": target_val_idx,
        "target_test_idx": target_test_idx,
        "eval_labels_all": eval_labels_all,
        "source_train_idx": source_train_idx,
    }
    return detector, fit_info, indices


def oracle_threshold_metrics(scores: np.ndarray, labels: np.ndarray, postprocess_cfg: Dict[str, Any]) -> Dict[str, Any]:
    processed = apply_score_postprocess(scores, postprocess_cfg)
    precision, recall, thresholds, f1 = threshold_curve(processed, labels)
    if len(thresholds) == 0:
        threshold = float(processed.max() + 1e-6) if len(processed) else 0.0
        metrics_item = metrics_at_with_postprocess(scores, labels, threshold, postprocess_cfg)
        return {"threshold": threshold, **metrics_item}
    order = np.lexsort((-recall, -precision, -f1))
    idx = int(order[0])
    threshold = float(thresholds[idx])
    metrics_item = metrics_at_with_postprocess(scores, labels, threshold, postprocess_cfg)
    return {"threshold": threshold, **metrics_item}


def distribution_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    diagnostics = score_diagnostics(scores, labels)
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    result: Dict[str, Any] = dict(diagnostics)
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
                "label_anomaly_ratio": float(labels.mean()) if len(labels) else 0.0,
            }
        )
    return result


def component_rows(
    pair_info: Dict[str, Any],
    val_scores_all: Dict[str, np.ndarray],
    test_scores_all: Dict[str, np.ndarray],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    postprocess_cfg = config.get("postprocess", {})
    components = [name for name in ["score", "residual", "edge"] if name in val_scores_all and name in test_scores_all]
    for component in components:
        detect_cfg = dict(config["detect"])
        detect_cfg["score_component"] = component
        detect_cfg["candidate_components"] = [component]
        val_for_threshold = apply_score_postprocess(val_scores_all[component], postprocess_cfg)
        threshold, val_at_threshold = select_threshold(val_for_threshold, val_labels, detect_cfg)
        test_at_val_threshold = metrics_at_with_postprocess(test_scores_all[component], test_labels, threshold, postprocess_cfg)
        val_oracle = oracle_threshold_metrics(val_scores_all[component], val_labels, postprocess_cfg)
        test_oracle = oracle_threshold_metrics(test_scores_all[component], test_labels, postprocess_cfg)
        val_dist = distribution_metrics(val_scores_all[component], val_labels)
        test_dist = distribution_metrics(test_scores_all[component], test_labels)
        rows.append(
            {
                **pair_info,
                "component": component,
                "threshold_from_val": float(threshold),
                "val_precision_at_threshold": safe_float(val_at_threshold.get("precision")),
                "val_recall_at_threshold": safe_float(val_at_threshold.get("recall")),
                "val_f1_at_threshold": safe_float(val_at_threshold.get("f1")),
                "val_constraint_satisfied": bool(val_at_threshold.get("constraint_satisfied", True)),
                "test_precision_at_val_threshold": safe_float(test_at_val_threshold.get("precision")),
                "test_recall_at_val_threshold": safe_float(test_at_val_threshold.get("recall")),
                "test_f1_at_val_threshold": safe_float(test_at_val_threshold.get("f1")),
                "test_roc_auc": safe_float(test_at_val_threshold.get("roc_auc")),
                "test_pr_auc": safe_float(test_at_val_threshold.get("pr_auc")),
                "val_oracle_f1": safe_float(val_oracle.get("f1")),
                "test_oracle_f1": safe_float(test_oracle.get("f1")),
                "test_oracle_precision": safe_float(test_oracle.get("precision")),
                "test_oracle_recall": safe_float(test_oracle.get("recall")),
                "threshold_transfer_f1_gap": safe_float(test_oracle.get("f1")) - safe_float(test_at_val_threshold.get("f1")),
                **{f"val_{key}": value for key, value in val_dist.items()},
                **{f"test_{key}": value for key, value in test_dist.items()},
            }
        )
    return rows


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
        "component",
        "test_precision_at_val_threshold",
        "test_recall_at_val_threshold",
        "test_f1_at_val_threshold",
        "test_oracle_f1",
        "threshold_transfer_f1_gap",
        "test_roc_auc",
        "test_pr_auc",
        "test_separation_gap_p05_p95",
        "test_anomaly_below_normal_p95_rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in headers) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze score separability and threshold transfer for Count-Granger pairs")
    parser.add_argument("--datasets", nargs="*", help="Dataset mapping items: NAME=PATH")
    parser.add_argument("--config", default="configs/count_granger_config.yaml")
    parser.add_argument("--output_root", default="results/score_separability")
    parser.add_argument("--cache_root", default="results/cache/count_granger")
    args = parser.parse_args()

    datasets = parse_dataset_items(args.datasets)
    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, stamp)
    ensure_dir(output_dir)
    all_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    run_context: Dict[str, Any] = {"datasets": datasets, "config": args.config}

    for source_name, source_path in datasets.items():
        for target_name, target_path in datasets.items():
            if source_name == target_name:
                continue
            pair = f"{safe_name(source_path)}_to_{safe_name(target_path)}"
            cache_dir = os.path.join(args.cache_root, pair)
            print(f"[separability] processing {source_name} -> {target_name}; cache={cache_dir}", flush=True)
            paths = preprocess_count_series_pair(source_path, target_path, cache_dir, **config["preprocess"])
            source = load_count_series(paths["source"])
            target = load_count_series(paths["target"])
            detector, fit_info, indices = fit_detector_for_pair(source, target, config)
            val_idx = indices["target_val_idx"]
            test_idx = indices["target_test_idx"]
            labels_all = indices["eval_labels_all"]
            val_labels = labels_all[val_idx]
            test_labels = labels_all[test_idx]
            val_scores_all = detector.score(target["counts"][val_idx])
            test_scores_all = detector.score(target["counts"][test_idx])
            selected_component, threshold, val_metrics, _, test_selected_scores = select_score_component(
                val_scores_all,
                test_scores_all,
                val_labels,
                config["detect"],
                config.get("postprocess", {}),
            )
            test_selected_metrics = metrics_at_with_postprocess(test_selected_scores, test_labels, threshold, config.get("postprocess", {}))
            pair_info = {
                "source": source_name,
                "target": target_name,
                "pair": pair,
                "selected_component": selected_component,
                "selected_threshold": float(threshold),
                "selected_test_precision": safe_float(test_selected_metrics.get("precision")),
                "selected_test_recall": safe_float(test_selected_metrics.get("recall")),
                "selected_test_f1": safe_float(test_selected_metrics.get("f1")),
                "selected_test_roc_auc": safe_float(test_selected_metrics.get("roc_auc")),
                "selected_test_pr_auc": safe_float(test_selected_metrics.get("pr_auc")),
                "target_val_bins": int(len(val_idx)),
                "target_test_bins": int(len(test_idx)),
                "target_val_anomaly_ratio": float(val_labels.mean()) if len(val_labels) else 0.0,
                "target_test_anomaly_ratio": float(test_labels.mean()) if len(test_labels) else 0.0,
                "num_features": fit_info.get("num_features"),
                "num_edges": fit_info.get("num_edges"),
                "calibration_strategy": fit_info.get("target_normal_calibration", {}).get("strategy"),
                "calibration_reason": fit_info.get("target_normal_calibration", {}).get("reason"),
            }
            rows = component_rows(pair_info, val_scores_all, test_scores_all, val_labels, test_labels, config)
            all_rows.extend(rows)
            selected_rows.append(pair_info)
            write_csv(os.path.join(output_dir, f"{source_name}_to_{target_name}_components.csv"), rows)
            run_context[pair] = {"fit_info": fit_info, "selected": pair_info}

    write_csv(os.path.join(output_dir, "score_separability_components.csv"), all_rows)
    write_csv(os.path.join(output_dir, "score_separability_selected.csv"), selected_rows)
    write_markdown(os.path.join(output_dir, "score_separability_components.md"), all_rows)
    with open(os.path.join(output_dir, "run_context.json"), "w", encoding="utf-8") as file:
        json.dump(run_context, file, ensure_ascii=False, indent=2)
    print(f"[separability] components: {os.path.join(output_dir, 'score_separability_components.csv')}", flush=True)
    print(f"[separability] selected: {os.path.join(output_dir, 'score_separability_selected.csv')}", flush=True)


if __name__ == "__main__":
    main()
