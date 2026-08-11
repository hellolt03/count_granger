import csv
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics


class AnomalyDetector:
    def __init__(self, threshold: Optional[float] = None, percentile: float = 95.0, zscore: bool = True):
        self.threshold = threshold
        self.percentile = percentile
        self.zscore = zscore
        self.error_mean = 0.0
        self.error_std = 1.0

    def fit_statistics(self, errors: np.ndarray) -> None:
        self.error_mean = float(np.median(errors))
        q75, q25 = np.percentile(errors, [75, 25])
        self.error_std = float(max((q75 - q25) / 1.349, np.std(errors), 1e-8))

    def normalize(self, errors: np.ndarray) -> np.ndarray:
        if not self.zscore:
            return errors
        return (errors - self.error_mean) / self.error_std

    @staticmethod
    def _metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
        preds = scores > threshold
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        return {"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def select_threshold_percentile(self, errors: np.ndarray) -> float:
        scores = self.normalize(errors)
        self.threshold = float(np.percentile(scores, self.percentile))
        return self.threshold

    @staticmethod
    def _best_threshold_from_curve(scores: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        if len(thresholds) == 0:
            threshold = float(scores.max() + 1e-6)
            item = AnomalyDetector._metrics_at_threshold(scores, labels, threshold)
            return threshold, item
        precision = precision[:-1]
        recall = recall[:-1]
        denom = precision + recall
        f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        if mask is not None:
            valid = np.where(mask(precision, recall, f1))[0]
            if len(valid) > 0:
                local = valid[np.lexsort((-recall[valid], -precision[valid], -f1[valid]))[0]]
            else:
                local = int(np.lexsort((-precision, -f1))[0])
        else:
            local = int(np.lexsort((-precision, -f1))[0])
        threshold = float(thresholds[local])
        item = {"threshold": threshold, "precision": float(precision[local]), "recall": float(recall[local]), "f1": float(f1[local])}
        return threshold, item

    def select_threshold_by_f1(self, errors: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        scores = self.normalize(errors)
        self.threshold, best = self._best_threshold_from_curve(scores, labels)
        return self.threshold, best

    def select_threshold_by_precision_at_recall(self, errors: np.ndarray, labels: np.ndarray, min_recall: float = 0.9) -> Tuple[float, Dict[str, float]]:
        scores = self.normalize(errors)
        self.threshold, best = self._best_threshold_from_curve(scores, labels, mask=lambda precision, recall, f1: recall >= min_recall)
        return self.threshold, best

    def select_threshold_by_f1_at_precision(self, errors: np.ndarray, labels: np.ndarray, min_precision: float = 0.9) -> Tuple[float, Dict[str, float]]:
        scores = self.normalize(errors)
        self.threshold, best = self._best_threshold_from_curve(scores, labels, mask=lambda precision, recall, f1: precision >= min_precision)
        return self.threshold, best

    def detect(self, errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Threshold has not been selected")
        scores = self.normalize(errors)
        return scores > self.threshold, scores

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=2)


def forecast_error(model, forecast_logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    return model.forecast_loss(forecast_logits, history, reduction="none")


@torch.no_grad()
def estimate_reference_adjacency(model, loader, domain: str, device: torch.device, max_batches: int = 50) -> torch.Tensor:
    model.eval()
    refs = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        history = batch["history"].to(device)
        refs.append(model(history, domain=domain, hard=False)["soft_adjacency"].detach())
    if not refs:
        raise RuntimeError("Cannot estimate reference adjacency from empty loader")
    return torch.stack(refs).mean(dim=0)


@torch.no_grad()
def collect_score_components(
    model,
    loader,
    domain: str,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    model.eval()
    forecast, base_forecast, granger_forecast, exogenous, propagation, gate, labels, target_ids = [], [], [], [], [], [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index == 1 or batch_index % 100 == 0:
            print(f"[eval] collecting {domain} score components: batch={batch_index}", flush=True)
        history = batch["history"].to(device)
        output = model(history, domain=domain, hard=False)
        forecast_loss = forecast_error(model, output["forecast"], history)
        forecast.extend(forecast_loss.detach().cpu().numpy().tolist())
        if "base_forecast" in output:
            base_loss = forecast_error(model, output["base_forecast"], history)
            base_forecast.extend(base_loss.detach().cpu().numpy().tolist())
        if "granger_forecast" in output:
            granger_loss = forecast_error(model, output["granger_forecast"], history)
            granger_forecast.extend(granger_loss.detach().cpu().numpy().tolist())
        if "exogenous_score" in output:
            exogenous.extend(output["exogenous_score"].detach().cpu().numpy().tolist())
        if "propagation_score" in output:
            propagation.extend(output["propagation_score"].detach().cpu().numpy().tolist())
        if "granger_gate" in output:
            gate.extend(output["granger_gate"].mean(dim=1).detach().cpu().numpy().tolist())
        if "label" in batch:
            labels.extend(batch["label"].numpy().tolist())
        target_ids.extend(batch["target_id"].numpy().tolist())
    components = {
        "forecast": np.asarray(forecast, dtype=np.float32),
        "target_id": np.asarray(target_ids, dtype=np.int64),
    }
    if len(base_forecast) == len(forecast):
        components["base_forecast"] = np.asarray(base_forecast, dtype=np.float32)
    if len(granger_forecast) == len(forecast):
        components["granger_forecast"] = np.asarray(granger_forecast, dtype=np.float32)
    if len(exogenous) == len(forecast):
        components["exogenous"] = np.asarray(exogenous, dtype=np.float32)
    if len(propagation) == len(forecast):
        components["propagation"] = np.asarray(propagation, dtype=np.float32)
    if len(gate) == len(forecast):
        components["granger_gate"] = np.asarray(gate, dtype=np.float32)
    return components, np.asarray(labels, dtype=np.int64) if labels else None


def _robust_stats(values: np.ndarray) -> Dict[str, float]:
    center = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    scale = float(max((q75 - q25) / 1.349, np.std(values), 1e-8))
    return {"mean": center, "std": scale, "count": int(len(values))}


def fit_component_stats(components: Dict[str, np.ndarray], event_calibration: bool = False, min_event_count: int = 50) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    event_ids = components.get("target_id", np.asarray([], dtype=np.int64))
    for name, values in components.items():
        if name == "target_id":
            continue
        item = {"global": _robust_stats(values)}
        if event_calibration and len(event_ids) == len(values):
            event_stats = {}
            for event_id in np.unique(event_ids):
                mask = event_ids == event_id
                if int(mask.sum()) >= min_event_count:
                    event_stats[int(event_id)] = _robust_stats(values[mask])
            item["events"] = event_stats
        stats[name] = item
    return stats


def _normalize_component(values: np.ndarray, component_stats: Dict, event_ids: np.ndarray) -> np.ndarray:
    global_stats = component_stats["global"]
    events = component_stats.get("events", {})
    if len(events) == 0 or len(event_ids) != len(values):
        return ((values - global_stats["mean"]) / (global_stats["std"] + 1e-8)).astype(np.float32)
    normalized = np.empty_like(values, dtype=np.float32)
    for index, value in enumerate(values):
        item = events.get(int(event_ids[index]), global_stats)
        normalized[index] = (value - item["mean"]) / (item["std"] + 1e-8)
    return normalized


def combine_components(
    components: Dict[str, np.ndarray],
    stats: Dict[str, Dict],
    forecast_weight: float = 1.0,
    exogenous_weight: float = 0.0,
    propagation_weight: float = 0.0,
    base_forecast_weight: float = 0.0,
    granger_forecast_weight: float = 0.0,
    gate_weight: float = 0.0,
) -> np.ndarray:
    event_ids = components.get("target_id", np.asarray([], dtype=np.int64))
    score = np.zeros_like(components["forecast"], dtype=np.float32)
    weights = {
        "forecast": float(forecast_weight),
        "base_forecast": float(base_forecast_weight),
        "granger_forecast": float(granger_forecast_weight),
        "exogenous": float(exogenous_weight),
        "propagation": float(propagation_weight),
        "granger_gate": float(gate_weight),
    }
    for name, weight in weights.items():
        if weight == 0.0 or name not in components or name not in stats:
            continue
        score = score + weight * _normalize_component(components[name], stats[name], event_ids)
    return score.astype(np.float32)


def orient_scores_by_validation_labels(val_scores: np.ndarray, val_labels: np.ndarray, test_scores: np.ndarray, train_scores: np.ndarray):
    try:
        roc_auc = metrics.roc_auc_score(val_labels, val_scores)
    except ValueError:
        return val_scores, test_scores, train_scores
    if roc_auc < 0.5:
        print(f"[eval] validation roc_auc={roc_auc:.6f} < 0.5; reversing anomaly score direction", flush=True)
        return -val_scores, -test_scores, -train_scores
    return val_scores, test_scores, train_scores


def print_score_diagnostics(name: str, scores: np.ndarray, labels: Optional[np.ndarray]) -> None:
    if labels is None or len(np.unique(labels)) < 2:
        print(f"[eval] {name} scores: mean={scores.mean():.6f}, p95={np.percentile(scores, 95):.6f}", flush=True)
        return
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    print(
        f"[eval] {name} score diagnostics: normal_mean={normal.mean():.6f}, normal_p95={np.percentile(normal, 95):.6f}, "
        f"anomaly_mean={anomaly.mean():.6f}, anomaly_p05={np.percentile(anomaly, 5):.6f}, "
        f"roc_auc={metrics.roc_auc_score(labels, scores):.6f}, pr_auc={metrics.average_precision_score(labels, scores):.6f}",
        flush=True,
    )


def write_event_diagnostics(components: Dict[str, np.ndarray], labels: np.ndarray, preds: np.ndarray, output_path: str) -> None:
    target_ids = components.get("target_id", np.asarray([], dtype=np.int64))
    if len(target_ids) == 0:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rows = []
    for event_id in np.unique(target_ids):
        mask = target_ids == event_id
        total = int(mask.sum())
        anomalies = int(labels[mask].sum())
        fp = int(((preds == 1) & (labels == 0) & mask).sum())
        fn = int(((preds == 0) & (labels == 1) & mask).sum())
        rows.append([int(event_id), total, anomalies, anomalies / max(1, total), fp, fn])
    rows.sort(key=lambda row: row[4], reverse=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["event_id", "total", "anomalies", "anomaly_rate", "false_positive", "false_negative"])
        for event_id, total, anomalies, rate, fp, fn in rows:
            writer.writerow([event_id, total, anomalies, f"{rate:.8f}", fp, fn])

