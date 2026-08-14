
import argparse
import copy
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from sklearn import metrics

from anomaly_detector import AnomalyDetector
from count_granger_model import CountGrangerConfig, CountGrangerDetector
from count_series_preprocessing import load_count_series, preprocess_count_series_pair


DEFAULT_CONFIG = {
    "preprocess": {
        "time_bin_seconds": 60.0,
        "min_freq_ratio": 0.000005,
        "max_templates": 3000,
        "max_lines": None,
        "num_fields": None,
        "source_message_start": None,
        "target_message_start": None,
        "semantic_clustering": True,
        "semantic_cluster_count": None,
        "semantic_cluster_ratio": 0.08,
        "semantic_cluster_max_clusters": 256,
        "semantic_cluster_tfidf_max_features": 4096,
        "semantic_cluster_random_seed": 8,
    },
    "split": {
        "source_train_ratio": 0.5,
        "target_train_ratio": 0.4,
        "val_ratio": 0.3,
        "test_ratio": 0.3,
        "min_train_bins": 20,
        "eval_mode": "stratified_original",
        "eval_anomaly_ratio": 0.5,
        "eval_label_min_anomaly_lines": 1,
        "random_seed": 8,
        "val_size": None,
        "test_size": None,
    },
    "run": {
        "use_target_train": True,
        "source_sample_ratio": None,
        "min_target_calibration_normal_bins": 100,
        "target_calibration_fallback": "pooled",
        "transfer_edge_include_validation_normal": False,
    },
    "granger": {
        "max_lag": 5,
        "ridge_alpha": 1.0,
        "transform": "log1p",
        "normalize": "robust",
        "min_total_count": 20.0,
        "min_active_bins": 5,
        "min_variance": 0.0001,
        "max_features": 128,
        "redundancy_threshold": 0.95,
        "feature_selection_mode": "pooled",
        "source_feature_weight": 0.25,
        "source_feature_min_active_bins": 2,
        "edge_threshold": 0.01,
        "edge_selection": "top_k_per_target",
        "top_k_parents": 10,
        "remove_self_edges": True,
        "residual_score_weight": 1.0,
        "edge_score_weight": 0.2,
        "transfer_edge_enabled": False,
        "transfer_edge_min_node_total": 5.0,
        "transfer_edge_min_node_active_bins": 2,
        "transfer_edge_min_response_quantile": 0.5,
        "transfer_edge_min_confidence": 0.0,
        "transfer_edge_response_weight": 0.6,
        "transfer_edge_rank_weight": 0.3,
        "transfer_edge_weight_consistency_weight": 0.1,
        "target_score_enabled": False,
        "target_score_top_ratio": 0.05,
        "target_score_top_min": 3,
        "target_score_min_scale": 0.000001,
    },
    "detect": {
        "percentile": 95.0,
        "zscore": False,
        "threshold_strategy": "f1_at_precision",
        "min_recall": 0.9,
        "min_precision": 0.85,
        "min_threshold": None,
        "precision_threshold_min_recall": 0.2,
        "validation_best_min_recall": 0.2,
        "validation_best_min_f1": 0.05,
        "score_component": "validation_best",
        "candidate_components": ["score", "residual", "edge"],
        "weighted_search_alpha": 1.0,
        "weighted_search_alpha_values": None,
        "weighted_search_betas": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        "weighted_search_normalized": False,
    },
    "postprocess": {
        "enabled": False,
        "min_consecutive": 1,
        "rolling_mean_window": 1,
    },
    "output_root": "results/count_granger",
    "cache_root": "results/cache/count_granger",
    "output_dir": None,
    "cache_dir": None,
}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as file:
            deep_update(config, yaml.safe_load(file) or {})
    return config


def safe_name(path: Optional[str]) -> str:
    if not path:
        return "none"
    base = os.path.basename(path).replace(".log", "").replace(".txt", "")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base) or "dataset"


def make_output_dirs(args, config: Dict[str, Any]) -> None:
    if config.get("output_dir") is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair = f"{safe_name(args.source_data)}_to_{safe_name(args.target_data)}" if args.source_data else safe_name(args.target_data)
        config["output_dir"] = os.path.join(config.get("output_root") or "results/count_granger", f"{stamp}_{pair}")
    if config.get("cache_dir") is None:
        pair = f"{safe_name(args.source_data)}_to_{safe_name(args.target_data)}" if args.source_data else safe_name(args.target_data)
        config["cache_dir"] = os.path.join(config.get("cache_root") or "results/cache/count_granger", pair)
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)


def split_indices(num_bins: int, split_cfg: Dict[str, Any], *, is_target: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio = float(split_cfg.get("target_train_ratio" if is_target else "source_train_ratio", 0.4))
    val_ratio = float(split_cfg.get("val_ratio", 0.3))
    test_ratio = float(split_cfg.get("test_ratio", 0.3))
    total = max(train_ratio + val_ratio + test_ratio, 1e-8)
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    train_end = max(int(num_bins * train_ratio), int(split_cfg.get("min_train_bins", 20)))
    train_end = min(max(train_end, 1), max(num_bins - 2, 1))
    val_end = min(num_bins - 1, train_end + max(1, int(num_bins * val_ratio)))
    indices = np.arange(num_bins, dtype=np.int64)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _balanced_sample(rng: np.random.Generator, labels: np.ndarray, pool: np.ndarray, anomaly_ratio: float, size: Optional[int]) -> np.ndarray:
    normal = pool[labels[pool] == 0]
    anomaly = pool[labels[pool] == 1]
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    anomaly_ratio = min(max(float(anomaly_ratio), 1e-6), 1.0 - 1e-6)
    if size is None:
        total = int(min(len(anomaly) / anomaly_ratio, len(normal) / (1.0 - anomaly_ratio)))
        size = max(2, total)
    anomaly_count = min(len(anomaly), max(1, int(round(size * anomaly_ratio))))
    normal_count = min(len(normal), max(1, int(size - anomaly_count)))
    if anomaly_count == 0 or normal_count == 0:
        sampled = pool.copy()
        rng.shuffle(sampled)
        return np.sort(sampled)
    sampled = np.concatenate([anomaly[:anomaly_count], normal[:normal_count]])
    rng.shuffle(sampled)
    return np.sort(sampled.astype(np.int64))


def _sample_without_ratio(rng: np.random.Generator, pool: np.ndarray, size: Optional[int]) -> np.ndarray:
    sampled = pool.copy()
    rng.shuffle(sampled)
    if size is not None:
        sampled = sampled[: min(len(sampled), max(1, int(size)))]
    return np.sort(sampled.astype(np.int64))


def split_target_indices(num_bins: int, labels: np.ndarray, split_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, chronological_val, chronological_test = split_indices(num_bins, split_cfg, is_target=True)
    eval_mode = str(split_cfg.get("eval_mode", "stratified_random"))
    if eval_mode == "chronological":
        return train_idx, chronological_val, chronological_test
    rng = np.random.default_rng(int(split_cfg.get("random_seed", 8)))
    remaining = np.arange(train_idx[-1] + 1 if len(train_idx) else 0, num_bins, dtype=np.int64)
    normal = remaining[labels[remaining] == 0]
    anomaly = remaining[labels[remaining] == 1]
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    val_ratio = float(split_cfg.get("val_ratio", 0.3))
    test_ratio = float(split_cfg.get("test_ratio", 0.3))
    val_share = val_ratio / max(val_ratio + test_ratio, 1e-8)
    val_normal_end = int(round(len(normal) * val_share))
    val_anomaly_end = int(round(len(anomaly) * val_share))
    val_normal, test_normal = normal[:val_normal_end], normal[val_normal_end:]
    val_anomaly, test_anomaly = anomaly[:val_anomaly_end], anomaly[val_anomaly_end:]
    val_pool = np.concatenate([val_normal, val_anomaly])
    test_pool = np.concatenate([test_normal, test_anomaly])
    val_size = split_cfg.get("val_size")
    test_size = split_cfg.get("test_size")
    if eval_mode == "stratified_original":
        val_idx = _sample_without_ratio(rng, val_pool, None if val_size is None else int(val_size))
        test_idx = _sample_without_ratio(rng, test_pool, None if test_size is None else int(test_size))
        return train_idx, val_idx, test_idx
    if eval_mode != "stratified_random":
        raise ValueError("eval_mode must be one of {'chronological', 'stratified_random', 'stratified_original'}")
    anomaly_ratio = float(split_cfg.get("eval_anomaly_ratio", 0.5))
    val_idx = _balanced_sample(rng, labels, val_pool, anomaly_ratio, None if val_size is None else int(val_size))
    test_idx = _balanced_sample(rng, labels, test_pool, anomaly_ratio, None if test_size is None else int(test_size))
    return train_idx, val_idx, test_idx


def evaluation_labels(series: Dict[str, np.ndarray], split_cfg: Dict[str, Any]) -> np.ndarray:
    min_lines = int(split_cfg.get("eval_label_min_anomaly_lines", 1) or 1)
    if min_lines <= 1:
        return series["labels"].astype(np.int64)
    anomaly_counts = series.get("anomaly_counts")
    if anomaly_counts is None:
        anomaly_counts = series["labels"]
    return (anomaly_counts.astype(np.int64) >= min_lines).astype(np.int64)


def threshold_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    precision = precision[:-1]
    recall = recall[:-1]
    thresholds = np.asarray(thresholds)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return precision, recall, thresholds, f1


def choose_threshold_index(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[int, bool]:
    if mask is not None:
        valid = np.flatnonzero(mask)
        if len(valid) > 0:
            order = np.lexsort((-recall[valid], -precision[valid], -f1[valid]))
            return int(valid[int(order[0])]), True
    all_indices = np.arange(len(f1))
    order = np.lexsort((-recall[all_indices], -precision[all_indices], -f1[all_indices]))
    return int(all_indices[int(order[0])]), False


def threshold_item(
    threshold: float,
    precision: float,
    recall: float,
    f1: float,
    *,
    strategy: str,
    constraint_satisfied: bool,
) -> Dict[str, Any]:
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "strategy": strategy,
        "constraint_satisfied": bool(constraint_satisfied),
    }


def select_threshold(scores: np.ndarray, labels: np.ndarray, detect_cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    detector = AnomalyDetector(percentile=float(detect_cfg.get("percentile", 95.0)), zscore=bool(detect_cfg.get("zscore", False)))
    detector.fit_statistics(scores)
    normalized = detector.normalize(scores)
    strategy = str(detect_cfg.get("threshold_strategy", "f1"))
    if len(np.unique(labels)) < 2:
        threshold = detector.select_threshold_percentile(scores)
        metrics_item = AnomalyDetector._metrics_at_threshold(normalized, labels, threshold)
        metrics_item.update({"strategy": "percentile", "constraint_satisfied": True})
        return threshold, metrics_item
    if strategy == "percentile":
        threshold = detector.select_threshold_percentile(scores)
        metrics_item = AnomalyDetector._metrics_at_threshold(normalized, labels, threshold)
        metrics_item.update({"strategy": strategy, "constraint_satisfied": True})
    else:
        precision, recall, thresholds, f1 = threshold_curve(normalized, labels)
        if len(thresholds) == 0:
            threshold = float(normalized.max() + 1e-6)
            metrics_item = AnomalyDetector._metrics_at_threshold(normalized, labels, threshold)
            metrics_item.update({"strategy": strategy, "constraint_satisfied": False})
            return threshold, metrics_item
        if strategy == "precision_at_recall":
            min_recall = float(detect_cfg.get("min_recall", 0.9))
            mask = recall >= min_recall
        elif strategy == "f1_at_precision":
            min_precision = float(detect_cfg.get("min_precision", 0.9))
            min_recall = float(detect_cfg.get("precision_threshold_min_recall", 0.2))
            mask = (precision >= min_precision) & (recall >= min_recall)
        else:
            mask = None
        index, constrained = choose_threshold_index(precision, recall, f1, mask=mask)
        threshold = float(thresholds[index])
        metrics_item = threshold_item(
            threshold,
            float(precision[index]),
            float(recall[index]),
            float(f1[index]),
            strategy=strategy,
            constraint_satisfied=constrained if mask is not None else True,
        )
    min_threshold = detect_cfg.get("min_threshold")
    if min_threshold is not None and threshold < float(min_threshold):
        threshold = float(min_threshold)
        constrained = bool(metrics_item.get("constraint_satisfied", True))
        metrics_item = AnomalyDetector._metrics_at_threshold(normalized, labels, threshold)
        metrics_item.update({"strategy": strategy, "constraint_satisfied": constrained})
    return threshold, metrics_item


def metrics_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = scores > threshold
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    result = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if len(np.unique(labels)) >= 2:
        result["roc_auc"] = float(metrics.roc_auc_score(labels, scores))
        result["pr_auc"] = float(metrics.average_precision_score(labels, scores))
    return result


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1 or len(values) == 0:
        return values.astype(np.float64, copy=True)
    result = np.zeros(len(values), dtype=np.float64)
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = (cumsum[index + 1] - cumsum[start]) / (index - start + 1)
    return result


def apply_consecutive_filter(preds: np.ndarray, min_consecutive: int) -> np.ndarray:
    min_consecutive = int(max(1, min_consecutive))
    preds = np.asarray(preds, dtype=bool)
    if min_consecutive <= 1 or len(preds) == 0:
        return preds.copy()
    filtered = np.zeros(len(preds), dtype=bool)
    run_start = None
    for index, value in enumerate(preds):
        if value and run_start is None:
            run_start = index
        if (not value or index == len(preds) - 1) and run_start is not None:
            run_end = index if value and index == len(preds) - 1 else index - 1
            if run_end - run_start + 1 >= min_consecutive:
                filtered[run_start : run_end + 1] = True
            run_start = None
    return filtered


def apply_score_postprocess(scores: np.ndarray, postprocess_cfg: Dict[str, Any]) -> np.ndarray:
    if not bool(postprocess_cfg.get("enabled", False)):
        return scores.astype(np.float64, copy=True)
    return rolling_mean(scores, int(postprocess_cfg.get("rolling_mean_window", 1)))


def predictions_from_scores(scores: np.ndarray, threshold: float, postprocess_cfg: Dict[str, Any]) -> np.ndarray:
    processed_scores = apply_score_postprocess(scores, postprocess_cfg)
    preds = processed_scores > threshold
    if bool(postprocess_cfg.get("enabled", False)):
        preds = apply_consecutive_filter(preds, int(postprocess_cfg.get("min_consecutive", 1)))
    return preds


def metrics_at_with_postprocess(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    postprocess_cfg: Dict[str, Any],
) -> Dict[str, float]:
    processed_scores = apply_score_postprocess(scores, postprocess_cfg)
    preds = predictions_from_scores(scores, threshold, postprocess_cfg)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    result = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if len(np.unique(labels)) >= 2:
        result["roc_auc"] = float(metrics.roc_auc_score(labels, processed_scores))
        result["pr_auc"] = float(metrics.average_precision_score(labels, processed_scores))
    result["postprocess_enabled"] = bool(postprocess_cfg.get("enabled", False))
    result["min_consecutive"] = int(postprocess_cfg.get("min_consecutive", 1))
    result["rolling_mean_window"] = int(postprocess_cfg.get("rolling_mean_window", 1))
    return result


def select_score_component(
    val_components: Dict[str, np.ndarray],
    test_components: Dict[str, np.ndarray],
    val_labels: np.ndarray,
    detect_cfg: Dict[str, Any],
    postprocess_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, float], np.ndarray, np.ndarray]:
    requested = str(detect_cfg.get("score_component", "score"))
    candidates = list(detect_cfg.get("candidate_components", ["score", "residual", "edge"]))
    postprocess_cfg = postprocess_cfg or {}
    if requested != "validation_best":
        candidates = [requested]
    min_best_recall = float(detect_cfg.get("validation_best_min_recall", 0.2))
    min_best_f1 = float(detect_cfg.get("validation_best_min_f1", 0.05))

    def eligible(metrics_item: Dict[str, Any]) -> bool:
        return (
            bool(metrics_item.get("constraint_satisfied", True))
            and metrics_item.get("recall", 0.0) >= min_best_recall
            and metrics_item.get("f1", 0.0) >= min_best_f1
        )

    best = None

    def consider_candidate(name: str, val_scores: np.ndarray, test_scores: np.ndarray) -> None:
        nonlocal best
        val_for_threshold = apply_score_postprocess(val_scores, postprocess_cfg)
        threshold, val_metrics = select_threshold(val_for_threshold, val_labels, detect_cfg)
        if bool(postprocess_cfg.get("enabled", False)):
            val_metrics = metrics_at_with_postprocess(val_scores, val_labels, threshold, postprocess_cfg)
            val_metrics["strategy"] = str(detect_cfg.get("threshold_strategy", "f1"))
            val_metrics["constraint_satisfied"] = True
        item = (name, threshold, val_metrics, val_scores, test_scores)
        if best is None:
            best = item
            return
        _, _, best_metrics, _, _ = best
        current_eligible = eligible(val_metrics)
        best_eligible = eligible(best_metrics)
        if current_eligible != best_eligible:
            if current_eligible:
                best = item
            return
        if (
            val_metrics.get("f1", 0.0) > best_metrics.get("f1", 0.0)
            or (
                val_metrics.get("f1", 0.0) == best_metrics.get("f1", 0.0)
                and val_metrics.get("precision", 0.0) > best_metrics.get("precision", 0.0)
            )
            or (
                val_metrics.get("f1", 0.0) == best_metrics.get("f1", 0.0)
                and val_metrics.get("precision", 0.0) == best_metrics.get("precision", 0.0)
                and val_metrics.get("recall", 0.0) > best_metrics.get("recall", 0.0)
            )
        ):
            best = item

    for name in candidates:
        if name == "weighted_search":
            if "residual" not in val_components or "edge" not in val_components:
                continue
            normalized_search = bool(detect_cfg.get("weighted_search_normalized", False))
            alpha_values = detect_cfg.get("weighted_search_alpha_values")
            if normalized_search:
                alpha_values = alpha_values or [0.0, 0.25, 0.5, 0.75, 1.0]
                pairs = [(float(alpha), 1.0 - float(alpha)) for alpha in alpha_values]
            else:
                alpha = float(detect_cfg.get("weighted_search_alpha", 1.0))
                betas = detect_cfg.get("weighted_search_betas", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
                pairs = [(alpha, float(beta)) for beta in betas]
            for alpha, beta in pairs:
                candidate_name = f"weighted_search_a{alpha:g}_b{beta:g}".replace(".", "p")
                val_scores = alpha * val_components["residual"] + beta * val_components["edge"]
                test_scores = alpha * test_components["residual"] + beta * test_components["edge"]
                consider_candidate(candidate_name, val_scores, test_scores)
            continue
        if name not in val_components or name not in test_components:
            continue
        consider_candidate(name, val_components[name], test_components[name])
    if best is None:
        raise ValueError("No valid score component candidates")
    return best


def score_diagnostics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    result = {
        "mean": float(scores.mean()) if len(scores) else 0.0,
        "p95": float(np.percentile(scores, 95)) if len(scores) else 0.0,
    }
    if len(labels) and len(np.unique(labels)) >= 2:
        normal = scores[labels == 0]
        anomaly = scores[labels == 1]
        result.update(
            {
                "normal_mean": float(normal.mean()) if len(normal) else 0.0,
                "normal_p95": float(np.percentile(normal, 95)) if len(normal) else 0.0,
                "anomaly_mean": float(anomaly.mean()) if len(anomaly) else 0.0,
                "anomaly_p05": float(np.percentile(anomaly, 5)) if len(anomaly) else 0.0,
                "roc_auc": float(metrics.roc_auc_score(labels, scores)),
                "pr_auc": float(metrics.average_precision_score(labels, scores)),
            }
        )
    return result


def normal_bin_count(labels: Optional[np.ndarray]) -> int:
    if labels is None:
        return 0
    return int((labels.astype(np.int64) == 0).sum())


def sample_source_train_indices(source_train_idx: np.ndarray, target_train_size: int, run_cfg: Dict[str, Any], seed: int) -> np.ndarray:
    ratio = run_cfg.get("source_sample_ratio")
    if ratio is None:
        return source_train_idx
    ratio = float(ratio)
    if ratio < 0:
        raise ValueError("source_sample_ratio must be non-negative or null")
    sample_size = int(round(max(0, target_train_size) * ratio))
    sample_size = min(sample_size, len(source_train_idx))
    if sample_size <= 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(source_train_idx, size=sample_size, replace=False)
    return np.sort(sampled.astype(np.int64))


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


def save_scores(
    path: str,
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    threshold: float,
    selected_component: str,
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    selected_scores_override: Optional[np.ndarray] = None,
) -> None:
    postprocess_cfg = postprocess_cfg or {}
    if selected_scores_override is not None:
        selected_scores = selected_scores_override
    elif selected_component in scores:
        selected_scores = scores[selected_component]
    else:
        selected_scores = np.full(len(labels), np.nan, dtype=np.float64)
    processed_scores = apply_score_postprocess(selected_scores, postprocess_cfg)
    predictions = predictions_from_scores(selected_scores, threshold, postprocess_cfg)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "bin",
            "label",
            "selected_score",
            "processed_score",
            "combined",
            "residual",
            "edge",
            "transfer_edge",
            "level",
            "prediction",
            "selected_component",
        ])
        for index in range(len(labels)):
            writer.writerow([
                index,
                int(labels[index]),
                float(selected_scores[index]),
                float(processed_scores[index]),
                float(scores["score"][index]),
                float(scores["residual"][index]),
                float(scores["edge"][index]),
                float(scores.get("transfer_edge", np.full(len(labels), np.nan))[index]),
                float(scores.get("level", np.full(len(labels), np.nan))[index]),
                int(predictions[index]),
                selected_component,
            ])


def run_train(args, config: Dict[str, Any]) -> Dict[str, Any]:
    make_output_dirs(args, config)
    print(f"[count-main] output_dir={config['output_dir']}", flush=True)
    print(f"[count-main] cache_dir={config['cache_dir']}", flush=True)
    paths = preprocess_count_series_pair(args.source_data, args.target_data, config["cache_dir"], **config["preprocess"])
    target = load_count_series(paths["target"])
    source = load_count_series(paths["source"]) if paths.get("source") else None
    eval_labels_all = evaluation_labels(target, config["split"])
    target_train_idx, target_val_idx, target_test_idx = split_target_indices(len(target["counts"]), eval_labels_all, config["split"])
    run_cfg = config.get("run", {})
    train_series = []
    train_labels = []
    train_roles = []
    source_train_counts = None
    source_train_labels = None
    source_train_idx_original = None
    if source is not None:
        source_train_idx, _, _ = split_indices(len(source["counts"]), config["split"], is_target=False)
        source_train_idx_original = source_train_idx
        source_train_idx = sample_source_train_indices(
            source_train_idx,
            len(target_train_idx),
            run_cfg,
            int(config.get("split", {}).get("random_seed", 8)),
        )
        source_train_counts = source["counts"][source_train_idx]
        source_train_labels = source["labels"][source_train_idx]
        if len(source_train_idx):
            train_series.append(source_train_counts)
            train_labels.append(source_train_labels)
            train_roles.append("source")
        print(
            f"[count-main] source train bins={len(source_train_idx):,}"
            + (f" / original={len(source_train_idx_original):,}" if source_train_idx_original is not None else ""),
            flush=True,
        )
    use_target_train = bool(config.get("run", {}).get("use_target_train", True))
    target_train_counts = target["counts"][target_train_idx]
    target_train_labels = target["labels"][target_train_idx]
    if use_target_train:
        train_series.append(target_train_counts)
        train_labels.append(target_train_labels)
        train_roles.append("target")
    elif source is None:
        raise ValueError("source_only requires source_data and at least one source training series")
    if not train_series:
        raise ValueError("No training series available after source sampling and target train selection")
    print(
        f"[count-main] target bins: train={len(target_train_idx):,}, val={len(target_val_idx):,}, test={len(target_test_idx):,}; "
        f"target anomaly bins={int(target['labels'].sum()):,}; "
        f"eval anomaly bins={int(eval_labels_all.sum()):,}; "
        f"eval_label_min_anomaly_lines={int(config['split'].get('eval_label_min_anomaly_lines', 1) or 1)}; "
        f"target_train_for_fit={use_target_train}",
        flush=True,
    )
    detector = CountGrangerDetector(CountGrangerConfig(**config["granger"]))
    fit_info = detector.fit_many(train_series, train_labels, train_roles)
    fit_info["train_roles"] = train_roles
    fit_info["source_sample_ratio"] = run_cfg.get("source_sample_ratio")
    fit_info["source_train_bins_original"] = None if source_train_idx_original is None else int(len(source_train_idx_original))
    fit_info["source_train_bins_used"] = 0 if source_train_counts is None else int(len(source_train_counts))
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
        if calibration_fallback == "source" and source_train_counts is not None and source_normal_bins > 0:
            target_calibration = detector.calibrate_many_normal([source_train_counts], [source_train_labels])
            applied_calibration = "source"
        elif calibration_fallback == "pooled":
            calibration_series = []
            calibration_labels = []
            if source_train_counts is not None and source_normal_bins > 0:
                calibration_series.append(source_train_counts)
                calibration_labels.append(source_train_labels)
            if target_normal_bins > 0 and source_normal_bins > 0:
                calibration_series.append(target_train_counts)
                calibration_labels.append(target_train_labels)
            if calibration_series:
                target_calibration = detector.calibrate_many_normal(calibration_series, calibration_labels)
                applied_calibration = "pooled" if len(calibration_series) > 1 else "source"
        elif calibration_fallback not in {"fit_train", "fit", "none"}:
            print(f"[count-main] unknown target_calibration_fallback={calibration_fallback}; using fit_train", flush=True)
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
    transfer_target_counts = target_train_counts
    transfer_target_labels = target_train_labels
    if bool(run_cfg.get("transfer_edge_include_validation_normal", False)):
        val_normal_idx = target_val_idx[eval_labels_all[target_val_idx] == 0]
        if len(val_normal_idx):
            transfer_target_counts = np.concatenate([target_train_counts, target["counts"][val_normal_idx]], axis=0)
            transfer_target_labels = np.concatenate([target_train_labels, np.zeros(len(val_normal_idx), dtype=np.int64)], axis=0)
    transfer_edge_info = detector.fit_transfer_edge_confidence(
        transfer_target_counts,
        transfer_target_labels,
        source_train_counts,
        source_train_labels,
    )
    transfer_edge_info["include_validation_normal"] = float(bool(run_cfg.get("transfer_edge_include_validation_normal", False)))
    transfer_edge_info["target_confidence_bins"] = float(len(transfer_target_labels))
    transfer_edge_info["target_confidence_normal_bins"] = float(normal_bin_count(transfer_target_labels))
    fit_info["transfer_edge"] = transfer_edge_info
    target_score_info = detector.fit_target_scores(target_train_counts, target_train_labels)
    fit_info["target_scores"] = target_score_info
    print(
        "[count-main] calibration: "
        f"strategy={applied_calibration}, target_normal_bins={target_normal_bins}, "
        f"source_normal_bins={source_normal_bins}, normal_bins={target_calibration.get('normal_bins')}, "
        f"reason={calibration_reason}",
        flush=True,
    )
    if transfer_edge_info.get("enabled"):
        print(f"[count-main] transfer_edge={transfer_edge_info}", flush=True)
    if target_score_info.get("enabled"):
        print(f"[count-main] target_scores={target_score_info}", flush=True)
    print(f"[count-main] fit_info={fit_info}", flush=True)
    val_scores_all = detector.score(target["counts"][target_val_idx])
    test_scores_all = detector.score(target["counts"][target_test_idx])
    val_labels = eval_labels_all[target_val_idx]
    test_labels = eval_labels_all[target_test_idx]
    postprocess_cfg = config.get("postprocess", {})
    selected_component, threshold, val_metrics, val_selected_scores, test_selected_scores = select_score_component(
        val_scores_all,
        test_scores_all,
        val_labels,
        config["detect"],
        postprocess_cfg,
    )
    test_metrics = metrics_at_with_postprocess(test_selected_scores, test_labels, threshold, postprocess_cfg)
    print(f"[count-eval] selected score component={selected_component}", flush=True)
    print(f"[count-eval] validation diagnostics: {score_diagnostics(val_selected_scores, val_labels)}", flush=True)
    print(f"[count-eval] test diagnostics: {score_diagnostics(test_selected_scores, test_labels)}", flush=True)
    print(f"Best validation score={selected_component}, threshold={threshold:.6f}, metrics={val_metrics}", flush=True)
    print(
        f"Precision: {test_metrics['precision']:.3f}, Recall: {test_metrics['recall']:.3f}, F1-Score: {test_metrics['f1']:.3f}",
        flush=True,
    )
    detector.save(os.path.join(config["output_dir"], "count_granger_model.json"))
    save_scores(
        os.path.join(config["output_dir"], "target_val_scores.csv"),
        val_scores_all,
        val_labels,
        threshold,
        selected_component,
        postprocess_cfg,
        val_selected_scores,
    )
    save_scores(
        os.path.join(config["output_dir"], "target_test_scores.csv"),
        test_scores_all,
        test_labels,
        threshold,
        selected_component,
        postprocess_cfg,
        test_selected_scores,
    )
    summary = {
        "source_data": args.source_data,
        "target_data": args.target_data,
        "paths": paths,
        "config": config,
        "fit_info": fit_info,
        "selected_score_component": selected_component,
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_diagnostics": score_diagnostics(val_selected_scores, val_labels),
        "test_diagnostics": score_diagnostics(test_selected_scores, test_labels),
    }
    if bool(config.get("run", {}).get("save_summary", True)):
        with open(os.path.join(config["output_dir"], "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Count-series Granger log anomaly detection")
    parser.add_argument("--data", help="Single dataset log path. Alias for --target_data when --source_data is omitted.")
    parser.add_argument("--source_data", default=None, help="Optional source log path for cross-system fitting.")
    parser.add_argument("--target_data", default=None, help="Target log path for validation/test anomaly detection.")
    parser.add_argument("--config", default="configs/count_granger_config.yaml")
    parser.add_argument("--mode", default="train", choices=["train"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_data is None:
        args.target_data = args.data
    if args.target_data is None:
        raise ValueError("Provide --target_data or --data")
    config = load_config(args.config)
    run_train(args, config)


if __name__ == "__main__":
    main()
