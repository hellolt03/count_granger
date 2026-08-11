
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge

try:
    from scipy.stats import rankdata
except Exception:  # pragma: no cover
    rankdata = None


@dataclass
class CountGrangerConfig:
    max_lag: int = 5
    ridge_alpha: float = 1.0
    transform: str = "log1p"
    normalize: str = "robust"
    min_total_count: float = 20.0
    min_active_bins: int = 5
    min_variance: float = 1e-4
    max_features: int = 128
    redundancy_threshold: float = 0.95
    edge_threshold: float = 0.01
    edge_selection: str = "top_k_per_target"
    top_k_parents: int = 10
    remove_self_edges: bool = True
    residual_score_weight: float = 1.0
    edge_score_weight: float = 0.2


class CountGrangerDetector:
    def __init__(self, config: CountGrangerConfig):
        self.config = config
        self.selected_features: Optional[np.ndarray] = None
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.model: Optional[Ridge] = None
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None
        self.adjacency: Optional[np.ndarray] = None
        self.train_residual_center = 0.0
        self.train_residual_scale = 1.0
        self.train_edge_center = 0.0
        self.train_edge_scale = 1.0

    def _transform(self, counts: np.ndarray) -> np.ndarray:
        values = counts.astype(np.float64, copy=False)
        if self.config.transform == "log1p":
            values = np.log1p(values)
        elif self.config.transform == "sqrt":
            values = np.sqrt(np.maximum(values, 0.0))
        elif self.config.transform not in {"none", None}:
            raise ValueError("transform must be one of {'log1p', 'sqrt', 'none'}")
        return values

    def _fit_scaler(self, values: np.ndarray) -> None:
        if self.config.normalize == "none":
            self.center = np.zeros(values.shape[1], dtype=np.float64)
            self.scale = np.ones(values.shape[1], dtype=np.float64)
            return
        self.center = np.median(values, axis=0)
        q75, q25 = np.percentile(values, [75, 25], axis=0)
        robust = (q75 - q25) / 1.349
        std = values.std(axis=0)
        self.scale = np.maximum.reduce([robust, std, np.full(values.shape[1], 1e-6)])

    def _apply_scaler(self, values: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("Scaler is not fitted")
        return (values - self.center) / self.scale

    @staticmethod
    def _robust_stats(scores: np.ndarray) -> Tuple[float, float]:
        center = float(np.median(scores)) if len(scores) else 0.0
        if len(scores):
            q75, q25 = np.percentile(scores, [75, 25])
            scale = float(max((q75 - q25) / 1.349, np.std(scores), 1e-8))
        else:
            scale = 1.0
        return center, scale

    def _normal_mask(self, labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if labels is None:
            return None
        return labels.astype(np.int64) == 0

    def _select_features(self, series_list: Sequence[np.ndarray], labels_list: Sequence[Optional[np.ndarray]]) -> np.ndarray:
        normal_values = []
        for counts, labels in zip(series_list, labels_list):
            mask = self._normal_mask(labels)
            normal_values.append(counts if mask is None else counts[mask])
        merged_counts = np.concatenate(normal_values, axis=0) if normal_values else series_list[0]
        totals = merged_counts.sum(axis=0)
        active = (merged_counts > 0).sum(axis=0)
        transformed = self._transform(merged_counts)
        variance = transformed.var(axis=0)
        candidates = np.where(
            (totals >= self.config.min_total_count)
            & (active >= self.config.min_active_bins)
            & (variance >= self.config.min_variance)
        )[0]
        if len(candidates) == 0:
            candidates = np.argsort(-variance)[: min(self.config.max_features, transformed.shape[1])]
        order = candidates[np.argsort(-variance[candidates])]
        order = order[: max(1, self.config.max_features * 2)]
        selected = self._remove_redundant(transformed[:, order], order, variance)
        selected = selected[: max(1, self.config.max_features)]
        print(
            f"[count-granger] feature selection: candidates={len(candidates):,}, selected={len(selected):,}",
            flush=True,
        )
        return np.asarray(selected, dtype=np.int64)

    def _remove_redundant(self, values: np.ndarray, original_indices: np.ndarray, variance: np.ndarray) -> List[int]:
        if len(original_indices) <= 1 or self.config.redundancy_threshold >= 1.0:
            return list(map(int, original_indices))
        if rankdata is not None:
            ranked = np.apply_along_axis(rankdata, 0, values)
        else:
            ranked = np.argsort(np.argsort(values, axis=0), axis=0).astype(np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
            corr = np.corrcoef(ranked, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        keep: List[int] = []
        for local_idx, feature_idx in enumerate(original_indices):
            redundant = False
            for kept_feature in keep:
                kept_local = int(np.where(original_indices == kept_feature)[0][0])
                if abs(corr[local_idx, kept_local]) >= self.config.redundancy_threshold:
                    redundant = True
                    if variance[feature_idx] > variance[kept_feature]:
                        keep.remove(int(kept_feature))
                        keep.append(int(feature_idx))
                    break
            if not redundant:
                keep.append(int(feature_idx))
            if len(keep) >= self.config.max_features:
                break
        return keep

    def _lagged_xy(self, values: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lag = int(max(1, self.config.max_lag))
        if values.shape[0] <= lag:
            return np.empty((0, lag * values.shape[1])), np.empty((0, values.shape[1])), np.empty((0,), dtype=np.int64)
        x_rows = []
        y_rows = []
        time_indices = []
        for index in range(lag, values.shape[0]):
            if labels is not None and np.any(labels[index - lag : index + 1] != 0):
                continue
            history = [values[index - step] for step in range(1, lag + 1)]
            x_rows.append(np.concatenate(history, axis=0))
            y_rows.append(values[index])
            time_indices.append(index)
        if not x_rows:
            return np.empty((0, lag * values.shape[1])), np.empty((0, values.shape[1])), np.empty((0,), dtype=np.int64)
        return np.vstack(x_rows), np.vstack(y_rows), np.asarray(time_indices, dtype=np.int64)

    def fit_many(self, series_list: Sequence[np.ndarray], labels_list: Optional[Sequence[Optional[np.ndarray]]] = None) -> Dict[str, float]:
        labels_list = labels_list or [None] * len(series_list)
        self.selected_features = self._select_features(series_list, labels_list)
        transformed_train = []
        for counts, labels in zip(series_list, labels_list):
            mask = self._normal_mask(labels)
            subset = counts[:, self.selected_features]
            transformed = self._transform(subset if mask is None else subset[mask])
            transformed_train.append(transformed)
        scaler_values = np.concatenate(transformed_train, axis=0)
        self._fit_scaler(scaler_values)

        all_x = []
        all_y = []
        train_scores = []
        train_edge_scores = []
        for counts, labels in zip(series_list, labels_list):
            values = self._apply_scaler(self._transform(counts[:, self.selected_features]))
            x, y, _ = self._lagged_xy(values, labels)
            if len(x):
                all_x.append(x)
                all_y.append(y)
        if not all_x:
            raise ValueError("No normal lagged samples available for Count-Granger fitting")
        x_train = np.vstack(all_x)
        y_train = np.vstack(all_y)
        self.model = Ridge(alpha=float(self.config.ridge_alpha), fit_intercept=True)
        self.model.fit(x_train, y_train)
        coefficients = np.asarray(self.model.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        self.coefficients = coefficients
        self.intercept = np.asarray(self.model.intercept_, dtype=np.float64).reshape(-1)
        self.adjacency = self._coefficients_to_adjacency()
        pred = self.model.predict(x_train)
        residual_scores = self._residual_score(y_train, pred)
        edge_scores = self._edge_score_from_lagged(x_train)
        self.train_residual_center, self.train_residual_scale = self._robust_stats(residual_scores)
        self.train_edge_center, self.train_edge_scale = self._robust_stats(edge_scores)
        return {
            "num_features": int(len(self.selected_features)),
            "num_train_samples": int(len(x_train)),
            "mean_train_residual": float(residual_scores.mean()),
            "mean_train_edge_score": float(edge_scores.mean()),
            "num_edges": int((self.adjacency > self.config.edge_threshold).sum()) if self.adjacency is not None else 0,
            "edge_selection": self.config.edge_selection,
            "top_k_parents": int(self.config.top_k_parents),
            "remove_self_edges": bool(self.config.remove_self_edges),
        }

    def _coefficients_to_adjacency(self) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        num_features = self.coefficients.shape[0]
        lag = int(max(1, self.config.max_lag))
        adjacency = np.zeros((num_features, num_features), dtype=np.float64)
        for step in range(lag):
            block = np.abs(self.coefficients[:, step * num_features : (step + 1) * num_features])
            adjacency += block
        return self._sparsify_adjacency(adjacency)

    def _sparsify_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        adjacency = adjacency.copy()
        if bool(self.config.remove_self_edges):
            np.fill_diagonal(adjacency, 0.0)
        selection = str(self.config.edge_selection or "threshold")
        if selection == "none":
            return adjacency
        threshold = float(self.config.edge_threshold)
        if selection == "threshold":
            sparse = adjacency.copy()
            sparse[sparse <= threshold] = 0.0
            return sparse
        if selection != "top_k_per_target":
            raise ValueError("edge_selection must be one of {'none', 'threshold', 'top_k_per_target'}")
        top_k = int(max(1, self.config.top_k_parents))
        sparse = np.zeros_like(adjacency)
        for target_idx in range(adjacency.shape[0]):
            row = adjacency[target_idx]
            candidates = np.flatnonzero(row > threshold)
            if len(candidates) == 0:
                continue
            if len(candidates) > top_k:
                local = np.argpartition(row[candidates], -top_k)[-top_k:]
                candidates = candidates[local]
            sparse[target_idx, candidates] = row[candidates]
        return sparse

    @staticmethod
    def _residual_score(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(actual - predicted), axis=1)

    def _edge_score_from_lagged(self, x_lagged: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.adjacency is None:
            raise RuntimeError("Model is not fitted")
        edge_mask = self.adjacency > self.config.edge_threshold
        if not edge_mask.any():
            return np.zeros(x_lagged.shape[0], dtype=np.float64)
        num_features = self.adjacency.shape[0]
        lag = int(max(1, self.config.max_lag))
        scores = np.zeros(x_lagged.shape[0], dtype=np.float64)
        for step in range(lag):
            block = self.coefficients[:, step * num_features : (step + 1) * num_features]
            contribution = np.abs(x_lagged[:, step * num_features : (step + 1) * num_features] @ (block * edge_mask).T)
            scores += contribution.mean(axis=1)
        return scores / lag

    def _raw_score(self, counts: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None or self.selected_features is None:
            raise RuntimeError("Detector is not fitted")
        values = self._apply_scaler(self._transform(counts[:, self.selected_features]))
        x, y, time_indices = self._lagged_xy(values, labels=None)
        residual_full = np.zeros(counts.shape[0], dtype=np.float64)
        edge_full = np.zeros(counts.shape[0], dtype=np.float64)
        if len(x):
            pred = self.model.predict(x)
            residual = self._residual_score(y, pred)
            edge = self._edge_score_from_lagged(x)
            residual_full[time_indices] = residual
            edge_full[time_indices] = edge
            fill_residual = float(np.median(residual))
            fill_edge = float(np.median(edge))
        else:
            fill_residual = 0.0
            fill_edge = 0.0
        lag = int(max(1, self.config.max_lag))
        residual_full[: min(lag, len(residual_full))] = fill_residual
        edge_full[: min(lag, len(edge_full))] = fill_edge
        return {"residual": residual_full, "edge": edge_full}

    def calibrate_many_normal(
        self,
        series_list: Sequence[np.ndarray],
        labels_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Dict[str, float]:
        labels_list = labels_list or [None] * len(series_list)
        residual_parts = []
        edge_parts = []
        normal_bins = 0
        for counts, labels in zip(series_list, labels_list):
            if counts is None or len(counts) == 0:
                continue
            raw = self._raw_score(counts)
            mask = np.ones(len(counts), dtype=bool) if labels is None else labels.astype(np.int64) == 0
            if not np.any(mask):
                continue
            residual_parts.append(raw["residual"][mask])
            edge_parts.append(raw["edge"][mask])
            normal_bins += int(mask.sum())
        if normal_bins <= 0:
            raise ValueError("Normal calibration requires at least one normal bin")
        residual_values = np.concatenate(residual_parts, axis=0)
        edge_values = np.concatenate(edge_parts, axis=0)
        self.train_residual_center, self.train_residual_scale = self._robust_stats(residual_values)
        self.train_edge_center, self.train_edge_scale = self._robust_stats(edge_values)
        return {
            "normal_bins": int(normal_bins),
            "residual_center": float(self.train_residual_center),
            "residual_scale": float(self.train_residual_scale),
            "edge_center": float(self.train_edge_center),
            "edge_scale": float(self.train_edge_scale),
        }

    def calibrate_normal(self, counts: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        return self.calibrate_many_normal([counts], [labels])

    def score(self, counts: np.ndarray) -> Dict[str, np.ndarray]:
        raw = self._raw_score(counts)
        residual_norm = (raw["residual"] - self.train_residual_center) / self.train_residual_scale
        edge_norm = (raw["edge"] - self.train_edge_center) / self.train_edge_scale
        combined = self.config.residual_score_weight * residual_norm + self.config.edge_score_weight * edge_norm
        return {"score": combined, "residual": residual_norm, "edge": edge_norm}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "selected_features": None if self.selected_features is None else self.selected_features.tolist(),
            "center": None if self.center is None else self.center.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "coefficients": None if self.coefficients is None else self.coefficients.tolist(),
            "intercept": None if self.intercept is None else self.intercept.tolist(),
            "adjacency": None if self.adjacency is None else self.adjacency.tolist(),
            "train_residual_center": self.train_residual_center,
            "train_residual_scale": self.train_residual_scale,
            "train_edge_center": self.train_edge_center,
            "train_edge_scale": self.train_edge_scale,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
