
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
    feature_selection_mode: str = "pooled"
    source_feature_weight: float = 0.25
    source_feature_min_active_bins: int = 2
    edge_threshold: float = 0.01
    edge_selection: str = "top_k_per_target"
    top_k_parents: int = 10
    remove_self_edges: bool = True
    residual_score_weight: float = 1.0
    edge_score_weight: float = 0.2
    transfer_edge_enabled: bool = False
    transfer_edge_min_node_total: float = 5.0
    transfer_edge_min_node_active_bins: int = 2
    transfer_edge_min_response_quantile: float = 0.5
    transfer_edge_min_confidence: float = 0.0
    transfer_edge_response_weight: float = 0.6
    transfer_edge_rank_weight: float = 0.3
    transfer_edge_weight_consistency_weight: float = 0.1
    target_score_enabled: bool = False
    target_score_top_ratio: float = 0.05
    target_score_top_min: int = 3
    target_score_min_scale: float = 1e-6


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
        self.transfer_edge_confidence: Optional[np.ndarray] = None
        self.train_transfer_edge_center = 0.0
        self.train_transfer_edge_scale = 1.0
        self.transfer_edge_info: Dict[str, float] = {}
        self.level_center: Optional[np.ndarray] = None
        self.level_scale: Optional[np.ndarray] = None
        self.train_level_center = 0.0
        self.train_level_scale = 1.0
        self.target_score_info: Dict[str, float] = {}

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

    def _normal_counts(self, counts: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
        mask = self._normal_mask(labels)
        return counts if mask is None else counts[mask]

    def _select_features(
        self,
        series_list: Sequence[np.ndarray],
        labels_list: Sequence[Optional[np.ndarray]],
        series_roles: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if str(self.config.feature_selection_mode or "pooled") == "source_informed":
            return self._select_features_source_informed(series_list, labels_list, series_roles)
        normal_values = []
        for counts, labels in zip(series_list, labels_list):
            normal_values.append(self._normal_counts(counts, labels))
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

    def _select_features_source_informed(
        self,
        series_list: Sequence[np.ndarray],
        labels_list: Sequence[Optional[np.ndarray]],
        series_roles: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        roles = [str(role).lower() for role in (series_roles or ["target"] * len(series_list))]
        target_parts = [self._normal_counts(counts, labels) for counts, labels, role in zip(series_list, labels_list, roles) if role == "target"]
        source_parts = [self._normal_counts(counts, labels) for counts, labels, role in zip(series_list, labels_list, roles) if role == "source"]
        if not target_parts:
            target_parts = [self._normal_counts(counts, labels) for counts, labels in zip(series_list, labels_list)]
            source_parts = []
        target_counts = np.concatenate(target_parts, axis=0)
        source_counts = np.concatenate(source_parts, axis=0) if source_parts else np.empty((0, target_counts.shape[1]))
        target_totals = target_counts.sum(axis=0)
        target_active = (target_counts > 0).sum(axis=0)
        target_values = self._transform(target_counts)
        target_variance = target_values.var(axis=0)
        if len(source_counts):
            source_active = (source_counts > 0).sum(axis=0)
            source_variance = self._transform(source_counts).var(axis=0)
            source_gate = source_active >= int(self.config.source_feature_min_active_bins)
        else:
            source_active = np.zeros(target_counts.shape[1], dtype=np.float64)
            source_variance = np.zeros(target_counts.shape[1], dtype=np.float64)
            source_gate = np.zeros(target_counts.shape[1], dtype=bool)
        candidates = np.where(
            (target_totals >= self.config.min_total_count)
            & (target_active >= self.config.min_active_bins)
            & (target_variance >= self.config.min_variance)
        )[0]
        if len(candidates) == 0:
            candidates = np.argsort(-target_variance)[: min(self.config.max_features, target_values.shape[1])]
        target_score = target_variance / max(float(np.max(target_variance[candidates])) if len(candidates) else 0.0, 1e-12)
        source_score = source_variance / max(float(np.max(source_variance[candidates])) if len(candidates) else 0.0, 1e-12)
        source_score = source_score * source_gate.astype(np.float64)
        combined = target_score + float(self.config.source_feature_weight) * source_score
        order = candidates[np.argsort(-combined[candidates])]
        order = order[: max(1, self.config.max_features * 2)]
        selected = self._remove_redundant(target_values[:, order], order, target_variance)
        selected = selected[: max(1, self.config.max_features)]
        print(
            "[count-granger] source-informed feature selection: "
            f"candidates={len(candidates):,}, selected={len(selected):,}, "
            f"source_active_candidates={int(source_gate[candidates].sum()):,}",
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

    def fit_many(
        self,
        series_list: Sequence[np.ndarray],
        labels_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        series_roles: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        labels_list = labels_list or [None] * len(series_list)
        self.selected_features = self._select_features(series_list, labels_list, series_roles)
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
        self.adjacency = self._coefficients_to_adjacency_from(self.coefficients)
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
        return self._coefficients_to_adjacency_from(self.coefficients)

    def _coefficients_to_adjacency_from(self, coefficients: np.ndarray) -> np.ndarray:
        num_features = coefficients.shape[0]
        lag = int(max(1, self.config.max_lag))
        adjacency = np.zeros((num_features, num_features), dtype=np.float64)
        for step in range(lag):
            block = np.abs(coefficients[:, step * num_features : (step + 1) * num_features])
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

    def _top_feature_mean(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(values.shape[0], dtype=np.float64)
        top_count = max(
            int(self.config.target_score_top_min),
            int(np.ceil(float(self.config.target_score_top_ratio) * values.shape[1])),
        )
        top_count = min(max(1, top_count), values.shape[1])
        if top_count == values.shape[1]:
            return values.mean(axis=1)
        top_values = np.partition(values, -top_count, axis=1)[:, -top_count:]
        return top_values.mean(axis=1)

    def _robust_feature_stats(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(values) == 0:
            raise ValueError("Robust feature statistics require at least one sample")
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0) / 0.6745
        std = values.std(axis=0)
        scale = np.maximum.reduce(
            [mad, std, np.full(values.shape[1], float(self.config.target_score_min_scale))]
        )
        return center, scale

    def fit_target_scores(self, target_counts: np.ndarray, target_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        if not bool(self.config.target_score_enabled):
            self.target_score_info = {"enabled": 0.0, "reason": "disabled"}
            return self.target_score_info
        if self.selected_features is None:
            raise RuntimeError("Detector is not fitted")
        labels = np.zeros(len(target_counts), dtype=np.int64) if target_labels is None else target_labels.astype(np.int64)
        normal_mask = labels == 0
        if not np.any(normal_mask):
            self.target_score_info = {"enabled": 1.0, "reason": "no_target_normal_bins", "normal_bins": 0.0}
            return self.target_score_info
        values = self._transform(target_counts[:, self.selected_features])
        self.level_center, self.level_scale = self._robust_feature_stats(values[normal_mask])
        raw = self._target_raw_scores(target_counts)
        self.train_level_center, self.train_level_scale = self._robust_stats(raw["level"][normal_mask])
        self.target_score_info = {
            "enabled": 1.0,
            "normal_bins": float(normal_mask.sum()),
            "top_features": float(max(
                int(self.config.target_score_top_min),
                int(np.ceil(float(self.config.target_score_top_ratio) * len(self.selected_features))),
            )),
            "level_center": float(self.train_level_center),
            "level_scale": float(self.train_level_scale),
        }
        return self.target_score_info

    def _target_raw_scores(self, counts: np.ndarray) -> Dict[str, np.ndarray]:
        if self.selected_features is None:
            raise RuntimeError("Detector is not fitted")
        if self.level_center is None or self.level_scale is None:
            return {}
        values = self._transform(counts[:, self.selected_features])
        level_z = np.abs(values - self.level_center) / self.level_scale
        level = self._top_feature_mean(level_z)
        return {"level": level}

    def _edge_score_from_lagged(self, x_lagged: np.ndarray, edge_weights: Optional[np.ndarray] = None) -> np.ndarray:
        if self.coefficients is None or self.adjacency is None:
            raise RuntimeError("Model is not fitted")
        if edge_weights is None:
            edge_weights = (self.adjacency > self.config.edge_threshold).astype(np.float64)
        else:
            edge_weights = np.asarray(edge_weights, dtype=np.float64)
        if not np.any(edge_weights > 0):
            return np.zeros(x_lagged.shape[0], dtype=np.float64)
        num_features = self.adjacency.shape[0]
        lag = int(max(1, self.config.max_lag))
        scores = np.zeros(x_lagged.shape[0], dtype=np.float64)
        for step in range(lag):
            block = self.coefficients[:, step * num_features : (step + 1) * num_features]
            contribution = np.abs(x_lagged[:, step * num_features : (step + 1) * num_features] @ (block * edge_weights).T)
            scores += contribution.mean(axis=1)
        return scores / lag

    def _target_edge_response(self, x_lagged: np.ndarray, y_current: np.ndarray) -> np.ndarray:
        if self.adjacency is None:
            raise RuntimeError("Model is not fitted")
        num_features = self.adjacency.shape[0]
        lag = int(max(1, self.config.max_lag))
        if len(x_lagged) < 2:
            return np.zeros((num_features, num_features), dtype=np.float64)
        y = y_current.astype(np.float64, copy=False)
        y_centered = y - y.mean(axis=0, keepdims=True)
        y_scale = np.sqrt(np.sum(y_centered * y_centered, axis=0))
        response = np.zeros((num_features, num_features), dtype=np.float64)
        for step in range(lag):
            x = x_lagged[:, step * num_features : (step + 1) * num_features].astype(np.float64, copy=False)
            x_centered = x - x.mean(axis=0, keepdims=True)
            x_scale = np.sqrt(np.sum(x_centered * x_centered, axis=0))
            denom = np.outer(y_scale, x_scale)
            corr = y_centered.T @ x_centered
            corr = np.divide(corr, denom, out=np.zeros_like(corr), where=denom > 1e-12)
            response = np.maximum(response, np.abs(corr))
        return response

    @staticmethod
    def _row_minmax_scores(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        scores = np.zeros_like(values, dtype=np.float64)
        for target_idx in range(values.shape[0]):
            cols = np.flatnonzero(candidates[target_idx])
            if len(cols) == 0:
                continue
            row = values[target_idx, cols]
            min_value = float(row.min())
            max_value = float(row.max())
            if max_value <= min_value + 1e-12:
                scores[target_idx, cols] = 1.0
            else:
                scores[target_idx, cols] = (row - min_value) / (max_value - min_value)
        return scores

    @staticmethod
    def _row_rank_scores(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        scores = np.zeros_like(values, dtype=np.float64)
        for target_idx in range(values.shape[0]):
            cols = np.flatnonzero(candidates[target_idx])
            if len(cols) == 0:
                continue
            order = cols[np.argsort(-values[target_idx, cols])]
            if len(order) == 1:
                scores[target_idx, order[0]] = 1.0
                continue
            for rank, col in enumerate(order):
                scores[target_idx, col] = 1.0 - rank / (len(order) - 1)
        return scores

    def _source_prior_adjacency(
        self,
        source_counts: Optional[np.ndarray],
        source_labels: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if source_counts is None or self.selected_features is None:
            return None
        values = self._apply_scaler(self._transform(source_counts[:, self.selected_features]))
        x_source, y_source, _ = self._lagged_xy(values, source_labels)
        if not len(x_source):
            return None
        model = Ridge(alpha=float(self.config.ridge_alpha), fit_intercept=True)
        model.fit(x_source, y_source)
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        return self._coefficients_to_adjacency_from(coefficients)

    def fit_transfer_edge_confidence(
        self,
        target_counts: np.ndarray,
        target_labels: Optional[np.ndarray] = None,
        source_counts: Optional[np.ndarray] = None,
        source_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if not bool(self.config.transfer_edge_enabled):
            self.transfer_edge_confidence = None
            self.transfer_edge_info = {"enabled": 0.0, "reason": "disabled"}
            return self.transfer_edge_info
        if self.selected_features is None or self.adjacency is None:
            raise RuntimeError("Detector is not fitted")
        source_adjacency = self._source_prior_adjacency(source_counts, source_labels)
        prior = self.adjacency if source_adjacency is None else source_adjacency
        candidate_edges = prior > self.config.edge_threshold
        if bool(self.config.remove_self_edges):
            candidate_edges = candidate_edges.copy()
            np.fill_diagonal(candidate_edges, False)
        normal_mask = np.ones(len(target_counts), dtype=bool) if target_labels is None else target_labels.astype(np.int64) == 0
        normal_counts = target_counts[normal_mask][:, self.selected_features]
        if len(normal_counts) == 0 or not candidate_edges.any():
            self.transfer_edge_confidence = np.zeros_like(self.adjacency, dtype=np.float64)
            self.transfer_edge_info = {"enabled": 1.0, "num_transfer_edges": 0.0, "reason": "no_normal_or_candidate_edges"}
            return self.transfer_edge_info
        totals = normal_counts.sum(axis=0)
        active = (normal_counts > 0).sum(axis=0)
        node_exists = (totals >= float(self.config.transfer_edge_min_node_total)) & (
            active >= int(self.config.transfer_edge_min_node_active_bins)
        )
        node_gate = np.outer(node_exists, node_exists)
        values = self._apply_scaler(self._transform(target_counts[:, self.selected_features]))
        x_target, y_target, _ = self._lagged_xy(values, target_labels)
        if not len(x_target):
            self.transfer_edge_confidence = np.zeros_like(self.adjacency, dtype=np.float64)
            self.transfer_edge_info = {"enabled": 1.0, "num_transfer_edges": 0.0, "reason": "no_target_normal_lag_windows"}
            return self.transfer_edge_info
        response = self._target_edge_response(x_target, y_target)
        candidates = candidate_edges & node_gate
        if not candidates.any():
            self.transfer_edge_confidence = np.zeros_like(self.adjacency, dtype=np.float64)
            self.transfer_edge_info = {"enabled": 1.0, "num_transfer_edges": 0.0, "reason": "node_gate_removed_all_edges"}
            return self.transfer_edge_info
        response_values = response[candidates]
        quantile = min(1.0, max(0.0, float(self.config.transfer_edge_min_response_quantile)))
        cutoff = float(np.quantile(response_values, quantile)) if len(response_values) else 0.0
        candidates &= response >= cutoff
        response_score = self._row_minmax_scores(response, candidates)
        rank_score = self._row_rank_scores(response, candidates)
        prior_score = self._row_minmax_scores(prior, candidates)
        weight_consistency = 1.0 - np.abs(prior_score - response_score)
        response_weight = float(self.config.transfer_edge_response_weight)
        rank_weight = float(self.config.transfer_edge_rank_weight)
        consistency_weight = float(self.config.transfer_edge_weight_consistency_weight)
        total_weight = max(response_weight + rank_weight + consistency_weight, 1e-12)
        confidence = candidates.astype(np.float64) * (
            response_weight * response_score + rank_weight * rank_score + consistency_weight * weight_consistency
        ) / total_weight
        min_confidence = float(self.config.transfer_edge_min_confidence)
        confidence[confidence < min_confidence] = 0.0
        self.transfer_edge_confidence = confidence
        selected = confidence > 0
        self.transfer_edge_info = {
            "enabled": 1.0,
            "used_source_prior": float(source_adjacency is not None),
            "num_candidate_edges": float(candidate_edges.sum()),
            "num_node_alive_candidate_edges": float((candidate_edges & node_gate).sum()),
            "num_transfer_edges": float(selected.sum()),
            "mean_confidence": float(confidence[selected].mean()) if selected.any() else 0.0,
            "median_confidence": float(np.median(confidence[selected])) if selected.any() else 0.0,
            "response_cutoff": cutoff,
        }
        raw = self._raw_score(target_counts)
        transfer_values = raw.get("transfer_edge")
        if transfer_values is not None:
            mask = normal_mask
            self.train_transfer_edge_center, self.train_transfer_edge_scale = self._robust_stats(transfer_values[mask])
            self.transfer_edge_info.update(
                {
                    "transfer_edge_center": float(self.train_transfer_edge_center),
                    "transfer_edge_scale": float(self.train_transfer_edge_scale),
                }
            )
        return self.transfer_edge_info

    def _raw_score(self, counts: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None or self.selected_features is None:
            raise RuntimeError("Detector is not fitted")
        values = self._apply_scaler(self._transform(counts[:, self.selected_features]))
        x, y, time_indices = self._lagged_xy(values, labels=None)
        residual_full = np.zeros(counts.shape[0], dtype=np.float64)
        edge_full = np.zeros(counts.shape[0], dtype=np.float64)
        transfer_edge_full = np.zeros(counts.shape[0], dtype=np.float64)
        if len(x):
            pred = self.model.predict(x)
            residual = self._residual_score(y, pred)
            edge = self._edge_score_from_lagged(x)
            transfer_edge = None
            if self.transfer_edge_confidence is not None:
                transfer_edge = self._edge_score_from_lagged(x, self.transfer_edge_confidence)
            residual_full[time_indices] = residual
            edge_full[time_indices] = edge
            if transfer_edge is not None:
                transfer_edge_full[time_indices] = transfer_edge
            fill_residual = float(np.median(residual))
            fill_edge = float(np.median(edge))
            fill_transfer_edge = float(np.median(transfer_edge)) if transfer_edge is not None else 0.0
        else:
            fill_residual = 0.0
            fill_edge = 0.0
            fill_transfer_edge = 0.0
        lag = int(max(1, self.config.max_lag))
        residual_full[: min(lag, len(residual_full))] = fill_residual
        edge_full[: min(lag, len(edge_full))] = fill_edge
        transfer_edge_full[: min(lag, len(transfer_edge_full))] = fill_transfer_edge
        raw = {"residual": residual_full, "edge": edge_full}
        if self.transfer_edge_confidence is not None:
            raw["transfer_edge"] = transfer_edge_full
        raw.update(self._target_raw_scores(counts))
        return raw

    def calibrate_many_normal(
        self,
        series_list: Sequence[np.ndarray],
        labels_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Dict[str, float]:
        labels_list = labels_list or [None] * len(series_list)
        residual_parts = []
        edge_parts = []
        transfer_edge_parts = []
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
            if "transfer_edge" in raw:
                transfer_edge_parts.append(raw["transfer_edge"][mask])
            normal_bins += int(mask.sum())
        if normal_bins <= 0:
            raise ValueError("Normal calibration requires at least one normal bin")
        residual_values = np.concatenate(residual_parts, axis=0)
        edge_values = np.concatenate(edge_parts, axis=0)
        self.train_residual_center, self.train_residual_scale = self._robust_stats(residual_values)
        self.train_edge_center, self.train_edge_scale = self._robust_stats(edge_values)
        result = {
            "normal_bins": int(normal_bins),
            "residual_center": float(self.train_residual_center),
            "residual_scale": float(self.train_residual_scale),
            "edge_center": float(self.train_edge_center),
            "edge_scale": float(self.train_edge_scale),
        }
        if transfer_edge_parts:
            transfer_edge_values = np.concatenate(transfer_edge_parts, axis=0)
            self.train_transfer_edge_center, self.train_transfer_edge_scale = self._robust_stats(transfer_edge_values)
            result.update(
                {
                    "transfer_edge_center": float(self.train_transfer_edge_center),
                    "transfer_edge_scale": float(self.train_transfer_edge_scale),
                }
            )
        return result

    def calibrate_normal(self, counts: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        return self.calibrate_many_normal([counts], [labels])

    def score(self, counts: np.ndarray) -> Dict[str, np.ndarray]:
        raw = self._raw_score(counts)
        residual_norm = (raw["residual"] - self.train_residual_center) / self.train_residual_scale
        edge_norm = (raw["edge"] - self.train_edge_center) / self.train_edge_scale
        combined = self.config.residual_score_weight * residual_norm + self.config.edge_score_weight * edge_norm
        result = {"score": combined, "residual": residual_norm, "edge": edge_norm}
        if "transfer_edge" in raw:
            result["transfer_edge"] = (raw["transfer_edge"] - self.train_transfer_edge_center) / self.train_transfer_edge_scale
        if "level" in raw:
            result["level"] = (raw["level"] - self.train_level_center) / self.train_level_scale
        return result

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
            "transfer_edge_confidence": None if self.transfer_edge_confidence is None else self.transfer_edge_confidence.tolist(),
            "train_transfer_edge_center": self.train_transfer_edge_center,
            "train_transfer_edge_scale": self.train_transfer_edge_scale,
            "transfer_edge_info": self.transfer_edge_info,
            "level_center": None if self.level_center is None else self.level_center.tolist(),
            "level_scale": None if self.level_scale is None else self.level_scale.tolist(),
            "train_level_center": self.train_level_center,
            "train_level_scale": self.train_level_scale,
            "target_score_info": self.target_score_info,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
