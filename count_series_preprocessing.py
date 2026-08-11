
import json
import os
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from sequence_preprocessing import (
    build_semantic_template_clusters,
    build_shared_template_vocab,
    infer_message_start,
    normalize_template,
)

COUNT_SERIES_SCHEMA_VERSION = 1
LABEL_NORMAL = "-"


def _safe_name(path: str) -> str:
    base = os.path.basename(path).replace(".log", "").replace(".txt", "")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base) or "dataset"


def _line_items(
    path: str,
    *,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    message_start: Optional[int] = None,
) -> Iterator[Tuple[float, str, int]]:
    start_index = infer_message_start(path) if message_start is None else int(message_start)
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line_index, line in enumerate(file):
            if max_lines is not None and line_index >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=(num_fields - 1) if num_fields else -1)
            if len(parts) < 2:
                continue
            label = parts[0]
            try:
                timestamp = float(parts[1])
            except ValueError:
                timestamp = float(line_index)
            start = min(max(1, start_index), len(parts))
            message = " ".join(parts[start:])
            yield timestamp, normalize_template(message), 0 if label == LABEL_NORMAL else 1


def _scan_log(
    path: str,
    *,
    max_lines: Optional[int],
    num_fields: Optional[int],
    message_start: Optional[int],
) -> Dict[str, object]:
    counter: Counter[str] = Counter()
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None
    lines = 0
    anomalies = 0
    for timestamp, template, label in _line_items(path, max_lines=max_lines, num_fields=num_fields, message_start=message_start):
        counter[template] += 1
        min_ts = timestamp if min_ts is None else min(min_ts, timestamp)
        max_ts = timestamp if max_ts is None else max(max_ts, timestamp)
        lines += 1
        anomalies += int(label)
    if min_ts is None or max_ts is None:
        raise ValueError(f"No usable timestamped log lines found in {path}")
    return {"counter": counter, "min_ts": float(min_ts), "max_ts": float(max_ts), "lines": lines, "anomalies": anomalies}


def _build_counts_for_log(
    path: str,
    *,
    vocab: Dict[str, int],
    event_cluster_ids: List[int],
    cluster_vocab_size: int,
    min_ts: float,
    max_ts: float,
    time_bin_seconds: float,
    max_lines: Optional[int],
    num_fields: Optional[int],
    message_start: Optional[int],
) -> Dict[str, np.ndarray]:
    num_bins = int(np.floor((max_ts - min_ts) / time_bin_seconds)) + 1
    counts = np.zeros((num_bins, cluster_vocab_size), dtype=np.float32)
    labels = np.zeros(num_bins, dtype=np.int64)
    line_counts = np.zeros(num_bins, dtype=np.int64)
    for timestamp, template, label in _line_items(path, max_lines=max_lines, num_fields=num_fields, message_start=message_start):
        bin_index = int(np.floor((timestamp - min_ts) / time_bin_seconds))
        if bin_index < 0 or bin_index >= num_bins:
            continue
        event_id = int(vocab.get(template, 0))
        cluster_id = int(event_cluster_ids[event_id]) if 0 <= event_id < len(event_cluster_ids) else 0
        counts[bin_index, cluster_id] += 1.0
        labels[bin_index] = max(labels[bin_index], int(label))
        line_counts[bin_index] += 1
    return {"counts": counts, "labels": labels, "line_counts": line_counts}


def _meta_path(path: str) -> str:
    return path + ".count_series.meta.json"


def _has_cache(paths: Iterable[str], preprocess_config: dict) -> bool:
    for path in paths:
        if not path or not os.path.exists(path) or not os.path.exists(_meta_path(path)):
            return False
        try:
            with open(_meta_path(path), "r", encoding="utf-8") as file:
                meta = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False
        if meta.get("schema_version") != COUNT_SERIES_SCHEMA_VERSION or meta.get("preprocess") != preprocess_config:
            return False
    return True


def preprocess_count_series_pair(
    source_log: Optional[str],
    target_log: str,
    output_dir: str,
    *,
    time_bin_seconds: float = 60.0,
    min_freq_ratio: float = 0.000005,
    max_templates: int = 3000,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    source_message_start: Optional[int] = None,
    target_message_start: Optional[int] = None,
    semantic_clustering: bool = True,
    semantic_cluster_count: Optional[int] = None,
    semantic_cluster_ratio: float = 0.08,
    semantic_cluster_max_clusters: int = 256,
    semantic_cluster_tfidf_max_features: int = 4096,
    semantic_cluster_random_seed: int = 8,
) -> Dict[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    source_name = _safe_name(source_log) if source_log else None
    target_name = _safe_name(target_log)
    prefix = f"{source_name}_to_{target_name}" if source_name else target_name
    source_out = os.path.join(output_dir, f"{prefix}_source_count_series.npz") if source_log else None
    target_out = os.path.join(output_dir, f"{prefix}_target_count_series.npz")
    template_path = os.path.join(output_dir, f"{prefix}_count_templates.json")
    cluster_path = os.path.join(output_dir, f"{prefix}_count_clusters.json")
    preprocess_config = {
        "time_bin_seconds": time_bin_seconds,
        "min_freq_ratio": min_freq_ratio,
        "max_templates": max_templates,
        "max_lines": max_lines,
        "num_fields": num_fields,
        "source_message_start": source_message_start,
        "target_message_start": target_message_start,
        "semantic_clustering": semantic_clustering,
        "semantic_cluster_count": semantic_cluster_count,
        "semantic_cluster_ratio": semantic_cluster_ratio,
        "semantic_cluster_max_clusters": semantic_cluster_max_clusters,
        "semantic_cluster_tfidf_max_features": semantic_cluster_tfidf_max_features,
        "semantic_cluster_random_seed": semantic_cluster_random_seed,
    }
    cache_paths = [target_out] + ([source_out] if source_out else [])
    if _has_cache(cache_paths, preprocess_config):
        print(f"[count-series] found cached count series: {', '.join(path for path in cache_paths if path)}", flush=True)
        return {"source": source_out, "target": target_out, "template_path": template_path, "cluster_path": cluster_path}

    logs = []
    if source_log:
        logs.append(("source", source_log, source_message_start))
    logs.append(("target", target_log, target_message_start))
    scans = {}
    merged_counter: Counter[str] = Counter()
    for role, path, start in logs:
        print(f"[count-series] scanning {role}: {path}", flush=True)
        scan = _scan_log(path, max_lines=max_lines, num_fields=num_fields, message_start=start)
        scans[role] = scan
        merged_counter.update(scan["counter"])
        print(
            f"[count-series] {role}: lines={scan['lines']:,}, anomalies={scan['anomalies']:,}, "
            f"templates={len(scan['counter']):,}, time=({scan['min_ts']:.0f},{scan['max_ts']:.0f})",
            flush=True,
        )

    vocab = build_shared_template_vocab([], [], min_freq_ratio, max_templates, 0, template_counter=merged_counter)
    cluster_info = build_semantic_template_clusters(
        merged_counter,
        vocab,
        enabled=semantic_clustering,
        cluster_count=semantic_cluster_count,
        cluster_ratio=semantic_cluster_ratio,
        max_clusters=semantic_cluster_max_clusters,
        tfidf_max_features=semantic_cluster_tfidf_max_features,
        random_seed=semantic_cluster_random_seed,
    )
    if semantic_clustering:
        event_cluster_ids = [int(x) for x in cluster_info.get("event_cluster_ids", [0] * len(vocab))]
        cluster_vocab_size = int(cluster_info.get("cluster_vocab_size") or (max(event_cluster_ids) + 1 if event_cluster_ids else 1))
    else:
        event_cluster_ids = list(range(len(vocab)))
        cluster_vocab_size = len(vocab)
        cluster_info = {**cluster_info, "cluster_mode": "template_identity", "cluster_vocab_size": cluster_vocab_size, "event_cluster_ids": event_cluster_ids}
    with open(template_path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    with open(cluster_path, "w", encoding="utf-8") as file:
        json.dump(cluster_info, file, ensure_ascii=False, indent=2)

    outputs = {}
    for role, path, start in logs:
        out_path = source_out if role == "source" else target_out
        scan = scans[role]
        print(f"[count-series] vectorizing {role} into {time_bin_seconds:g}s bins", flush=True)
        item = _build_counts_for_log(
            path,
            vocab=vocab,
            event_cluster_ids=event_cluster_ids,
            cluster_vocab_size=cluster_vocab_size,
            min_ts=float(scan["min_ts"]),
            max_ts=float(scan["max_ts"]),
            time_bin_seconds=float(time_bin_seconds),
            max_lines=max_lines,
            num_fields=num_fields,
            message_start=start,
        )
        np.savez_compressed(
            out_path,
            counts=item["counts"],
            labels=item["labels"],
            line_counts=item["line_counts"],
            min_ts=np.asarray(float(scan["min_ts"]), dtype=np.float64),
            time_bin_seconds=np.asarray(float(time_bin_seconds), dtype=np.float64),
        )
        meta = {
            "schema_version": COUNT_SERIES_SCHEMA_VERSION,
            "role": role,
            "raw_path": path,
            "count_series_path": out_path,
            "template_path": template_path,
            "cluster_path": cluster_path,
            "num_bins": int(item["counts"].shape[0]),
            "num_features": int(item["counts"].shape[1]),
            "anomaly_bins": int(item["labels"].sum()),
            "line_count": int(scan["lines"]),
            "anomaly_lines": int(scan["anomalies"]),
            "preprocess": preprocess_config,
        }
        with open(_meta_path(out_path), "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
        outputs[role] = out_path
        print(
            f"[count-series] saved {role}: {out_path}, shape={item['counts'].shape}, "
            f"anomaly_bins={int(item['labels'].sum()):,}",
            flush=True,
        )
    return {"source": outputs.get("source"), "target": outputs["target"], "template_path": template_path, "cluster_path": cluster_path}


def load_count_series(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {"counts": data["counts"].astype(np.float32), "labels": data["labels"].astype(np.int64), "line_counts": data["line_counts"].astype(np.int64)}
