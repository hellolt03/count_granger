import hashlib
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - sklearn is expected but we keep a safe fallback.
    MiniBatchKMeans = None
    TfidfVectorizer = None

SEQUENCE_SCHEMA_VERSION = 10
LABEL_NORMAL = "-"

_NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
_HEX_RE = re.compile(r"0x[0-9A-Fa-f]+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s]+")
_WS_RE = re.compile(r"\s+")
_HDFS_BLOCK_RE = re.compile(r"blk_-?\d+")


def label_path(sequence_path: str) -> str:
    return sequence_path[:-4] + "_labels.npy" if sequence_path.endswith(".npy") else sequence_path + "_labels.npy"


def session_path(sequence_path: str) -> str:
    return sequence_path[:-4] + "_sessions.npy" if sequence_path.endswith(".npy") else sequence_path + "_sessions.npy"



def infer_message_start(path: str) -> int:
    """Infer the 0-based token index where the semantic log message begins."""
    lower_path = path.lower()
    if "hdfs" in lower_path:
        return 4  # Keep HDFS component plus message body; skip date/time/thread/level.
    if "bgl" in lower_path:
        return 6  # Keep RAS/KERNEL/LEVEL plus message body.
    if "thunderbird" in lower_path or "spirit" in lower_path:
        return 8  # Keep daemon/component plus message body.
    return 1  # Fallback: keep everything after the label.


def _is_hdfs_log(path: str) -> bool:
    return "hdfs" in os.path.normpath(path).lower()


def _hdfs_label_path(path: str) -> str:
    return os.path.join(os.path.dirname(path), "anomaly_label.csv")


def _hdfs_npz_path(path: str) -> str:
    return os.path.join(os.path.dirname(path), "HDFS.npz")


def _hdfs_templates_path(path: str) -> str:
    return os.path.join(os.path.dirname(path), "HDFS.log_templates.csv")


def _load_hdfs_anomaly_blocks(path: str) -> set[str]:
    label_file = _hdfs_label_path(path)
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"HDFS log requires anomaly labels at {label_file}")
    anomaly_blocks: set[str] = set()
    with open(label_file, "r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            block_id = (row.get("BlockId") or "").strip()
            label = (row.get("Label") or "").strip().lower()
            if block_id and label == "anomaly":
                anomaly_blocks.add(block_id)
    return anomaly_blocks


def _load_hdfs_templates(path: str) -> Dict[str, str]:
    template_file = _hdfs_templates_path(path)
    if not os.path.exists(template_file):
        return {}
    templates: Dict[str, str] = {}
    with open(template_file, "r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            event_id = (row.get("EventId") or "").strip()
            template = (row.get("EventTemplate") or "").strip()
            if event_id and template:
                templates[event_id] = template.replace("[*]", " ").strip()
    return templates


def read_hdfs_labeled_messages(
    path: str,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    message_start: int = 4,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    npz_file = _hdfs_npz_path(path)
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Session-aware HDFS preprocessing requires {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    traces = data["x_data"]
    trace_labels = data["y_data"].astype(np.int64)
    templates = _load_hdfs_templates(path)
    messages: List[str] = []
    labels: List[int] = []
    sessions: List[Tuple[int, int]] = []
    skipped = 0
    for trace, label in zip(traces, trace_labels):
        events = list(trace)
        if not events:
            skipped += 1
            continue
        if max_lines is not None and len(messages) + len(events) > max_lines:
            break
        start = len(messages)
        for event_id in events:
            event = str(event_id)
            messages.append(templates.get(event, event))
            labels.append(int(label))
        sessions.append((start, len(messages)))
    print(
        f"[sequence] HDFS session traces: sessions={len(sessions):,}; skipped={skipped:,}; "
        f"events={len(messages):,}; anomaly_events={int(np.sum(labels)):,}; "
        f"anomaly_sessions={int(np.sum(trace_labels[:len(sessions)])):,}",
        flush=True,
    )
    return messages, np.asarray(labels, dtype=np.int64), np.asarray(sessions, dtype=np.int64)


def read_labeled_messages_with_sessions(
    path: str,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    message_start: int = 1,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    if _is_hdfs_log(path):
        return read_hdfs_labeled_messages(path, max_lines=max_lines, num_fields=num_fields, message_start=message_start)
    messages: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line_index, line in enumerate(file):
            if max_lines is not None and line_index >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=(num_fields - 1) if num_fields else -1)
            label = parts[0] if parts else LABEL_NORMAL
            start = min(max(1, int(message_start)), len(parts)) if parts else 0
            message = " ".join(parts[start:]) if parts else ""
            messages.append(message)
            labels.append(0 if label == LABEL_NORMAL else 1)
    return messages, np.asarray(labels, dtype=np.int64), None


def read_labeled_messages(
    path: str,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    message_start: int = 1,
) -> Tuple[List[str], np.ndarray]:
    messages, labels, _ = read_labeled_messages_with_sessions(
        path, max_lines=max_lines, num_fields=num_fields, message_start=message_start
    )
    return messages, labels


def normalize_template(message: str) -> str:
    # Replace unstable values so equivalent log messages share one event template.
    text = message.lower()
    text = _PATH_RE.sub(" <path> ", text)
    text = _IP_RE.sub(" <ip> ", text)
    text = _HEX_RE.sub(" <hex> ", text)
    text = _NUMERIC_RE.sub(" <num> ", text)
    text = re.sub(r"[^a-z0-9_<>=:\-/\.]+", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text or "<empty>"


def collect_template_statistics(source_messages: Iterable[str], target_messages: Iterable[str]) -> Tuple[Counter[str], int]:
    counter: Counter[str] = Counter()
    total = 0
    for message in source_messages:
        counter[normalize_template(message)] += 1
        total += 1
    for message in target_messages:
        counter[normalize_template(message)] += 1
        total += 1
    return counter, total


def _stable_hash_bucket(template: str, num_buckets: int) -> int:
    digest = hashlib.md5(template.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % max(1, num_buckets)


def _is_special_vocab_entry(template: str) -> bool:
    return template == "<UNK>" or template.startswith("<HASH_")


def build_shared_template_vocab(
    source_messages: Iterable[str],
    target_messages: Iterable[str],
    min_freq_ratio: float = 0.001,
    max_events: int = 500,
    num_hash_buckets: int = 128,
    template_counter: Optional[Counter[str]] = None,
) -> Dict[str, int]:
    if template_counter is None:
        counter, total = collect_template_statistics(source_messages, target_messages)
    else:
        counter = template_counter
        total = int(sum(counter.values()))
    # Keep frequent templates as explicit event IDs; map rare templates to hash buckets.
    min_count = max(1, int(total * min_freq_ratio))
    top_capacity = max(1, max_events - 1 - max(0, num_hash_buckets))
    kept = [template for template, count in counter.most_common() if count >= min_count][:top_capacity]
    vocab = {"<UNK>": 0}
    vocab.update({template: index + 1 for index, template in enumerate(kept)})
    hash_start = len(vocab)
    for bucket in range(max(0, num_hash_buckets)):
        vocab[f"<HASH_{bucket}>"] = hash_start + bucket
    print(
        f"[sequence] extracted {len(counter):,} raw templates; top_templates={len(kept):,}; "
        f"hash_buckets={num_hash_buckets}; num_events={len(vocab):,}; min_count={min_count}",
        flush=True,
    )
    return vocab


def _auto_cluster_count(num_templates: int, max_clusters: int, cluster_ratio: float) -> int:
    if num_templates <= 1:
        return 1
    target = max(1, int(round(num_templates * cluster_ratio)))
    target = min(max_clusters, target)
    target = min(target, max(1, num_templates // 2))
    return max(1, target)


def build_semantic_template_clusters(
    template_counter: Counter[str],
    vocab: Dict[str, int],
    *,
    enabled: bool = True,
    cluster_count: Optional[int] = None,
    cluster_ratio: float = 0.08,
    max_clusters: int = 256,
    tfidf_max_features: int = 4096,
    random_seed: int = 8,
) -> Dict[str, object]:
    explicit_templates = [template for template in vocab.keys() if not _is_special_vocab_entry(template)]
    if not enabled or not explicit_templates:
        return {
            "enabled": False,
            "cluster_mode": "disabled",
            "cluster_count": 1,
            "cluster_vocab_size": 1,
            "event_cluster_ids": [0] * len(vocab),
            "template_cluster_ids": {template: 0 for template in explicit_templates},
            "cluster_representatives": {"0": "<disabled>"},
        }

    if TfidfVectorizer is None or MiniBatchKMeans is None:
        print("[sequence] semantic clustering fallback: sklearn unavailable, using a single cluster", flush=True)
        template_cluster_ids = {template: 1 for template in explicit_templates}
        event_cluster_ids = [0] * len(vocab)
        for template, event_id in vocab.items():
            if not _is_special_vocab_entry(template):
                event_cluster_ids[event_id] = 1
        return {
            "enabled": True,
            "cluster_mode": "fallback_single_cluster",
            "cluster_count": 1,
            "cluster_vocab_size": 2,
            "event_cluster_ids": event_cluster_ids,
            "template_cluster_ids": template_cluster_ids,
            "cluster_representatives": {"1": explicit_templates[0]},
        }

    ordered_templates = sorted(explicit_templates, key=lambda template: (-template_counter.get(template, 0), template))
    if cluster_count is None:
        cluster_count = _auto_cluster_count(len(ordered_templates), max_clusters=max_clusters, cluster_ratio=cluster_ratio)
    cluster_count = max(1, min(int(cluster_count), max(1, len(ordered_templates) // 2)))
    if cluster_count <= 1:
        template_cluster_ids = {template: 1 for template in explicit_templates}
        event_cluster_ids = [0] * len(vocab)
        for template, event_id in vocab.items():
            if not _is_special_vocab_entry(template):
                event_cluster_ids[event_id] = 1
        return {
            "enabled": True,
            "cluster_mode": "single_cluster",
            "cluster_count": 1,
            "cluster_vocab_size": 2,
            "event_cluster_ids": event_cluster_ids,
            "template_cluster_ids": template_cluster_ids,
            "cluster_representatives": {"1": ordered_templates[0]},
        }

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        min_df=1,
        max_features=tfidf_max_features,
    )
    feature_matrix = vectorizer.fit_transform(ordered_templates)
    weights = np.asarray([max(1, template_counter.get(template, 1)) for template in ordered_templates], dtype=np.float64)
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(random_seed),
        batch_size=min(1024, len(ordered_templates)),
        n_init=10,
    )
    try:
        clusterer.fit(feature_matrix, sample_weight=weights)
    except TypeError:
        clusterer.fit(feature_matrix)
    labels = clusterer.labels_.astype(np.int64)
    template_cluster_ids = {template: int(label) + 1 for template, label in zip(ordered_templates, labels)}
    cluster_representatives: Dict[str, str] = {}
    cluster_best: Dict[int, Tuple[int, str]] = {}
    for template, label in zip(ordered_templates, labels):
        count = int(template_counter.get(template, 1))
        current = cluster_best.get(int(label))
        if current is None or count > current[0] or (count == current[0] and template < current[1]):
            cluster_best[int(label)] = (count, template)
    for label, (_, template) in cluster_best.items():
        cluster_representatives[str(int(label) + 1)] = template

    event_cluster_ids = [0] * len(vocab)
    for template, event_id in vocab.items():
        if _is_special_vocab_entry(template):
            continue
        event_cluster_ids[event_id] = template_cluster_ids.get(template, 0)
    print(
        f"[sequence] semantic clustering enabled; templates={len(ordered_templates):,}; clusters={cluster_count:,}; "
        f"tfidf_features={feature_matrix.shape[1]:,}",
        flush=True,
    )
    return {
        "enabled": True,
        "cluster_mode": "char_tfidf_kmeans",
        "cluster_count": int(cluster_count),
        "cluster_vocab_size": int(cluster_count + 1),
        "event_cluster_ids": event_cluster_ids,
        "template_cluster_ids": template_cluster_ids,
        "cluster_representatives": cluster_representatives,
    }


def encode_messages(messages: Iterable[str], vocab: Dict[str, int], num_hash_buckets: int = 128) -> np.ndarray:
    hash_start = len(vocab) - max(0, num_hash_buckets)
    encoded = []
    for message in messages:
        template = normalize_template(message)
        if template in vocab:
            encoded.append(vocab[template])
        elif num_hash_buckets > 0:
            encoded.append(hash_start + _stable_hash_bucket(template, num_hash_buckets))
        else:
            encoded.append(0)
    return np.asarray(encoded, dtype=np.int64)


def _auto_cluster_count(num_templates: int, max_clusters: int, cluster_ratio: float) -> int:
    if num_templates <= 1:
        return 1
    target = max(1, int(round(num_templates * cluster_ratio)))
    target = min(max_clusters, target)
    target = min(target, max(1, num_templates // 2))
    return max(1, target)


def _meta_path(sequence_path: str) -> str:
    return sequence_path + ".sequence.meta.json"


def has_current_sequence_cache(source_path: str, target_path: str, config: dict) -> bool:
    required = [source_path, target_path, label_path(source_path), label_path(target_path), _meta_path(source_path), _meta_path(target_path)]
    if not all(os.path.exists(path) for path in required):
        return False
    try:
        with open(_meta_path(source_path), "r", encoding="utf-8") as file:
            source_meta = json.load(file)
        with open(_meta_path(target_path), "r", encoding="utf-8") as file:
            target_meta = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    for path, meta in ((source_path, source_meta), (target_path, target_meta)):
        if meta.get("session_path") and not os.path.exists(session_path(path)):
            return False
    return (
        source_meta.get("schema_version") == SEQUENCE_SCHEMA_VERSION
        and target_meta.get("schema_version") == SEQUENCE_SCHEMA_VERSION
        and source_meta.get("preprocess") == config
        and target_meta.get("preprocess") == config
        and source_meta.get("num_events") == target_meta.get("num_events")
    )


def preprocess_log_pair_sequence(
    source_log: str,
    target_log: str,
    output_dir: str,
    window_seconds: float = 10.0,
    min_freq_ratio: float = 0.001,
    max_events: int = 500,
    max_lines: Optional[int] = None,
    num_fields: Optional[int] = None,
    num_hash_buckets: int = 128,
    source_message_start: Optional[int] = None,
    target_message_start: Optional[int] = None,
    semantic_clustering: bool = True,
    semantic_cluster_count: Optional[int] = None,
    semantic_cluster_ratio: float = 0.08,
    semantic_cluster_max_clusters: int = 256,
    semantic_cluster_tfidf_max_features: int = 4096,
    semantic_cluster_random_seed: int = 8,
) -> Tuple[str, str]:
    # Cache shared event sequences by source-target pair so repeated experiments are fast.
    os.makedirs(output_dir, exist_ok=True)
    source_out = os.path.join(output_dir, "source_sequence.npy")
    target_out = os.path.join(output_dir, "target_sequence.npy")
    source_start = infer_message_start(source_log) if source_message_start is None else int(source_message_start)
    target_start = infer_message_start(target_log) if target_message_start is None else int(target_message_start)
    preprocess_config = {
        "min_freq_ratio": min_freq_ratio,
        "max_events": max_events,
        "max_lines": max_lines,
        "num_fields": num_fields,
        "num_hash_buckets": num_hash_buckets,
        "source_message_start": source_start,
        "target_message_start": target_start,
        "semantic_clustering": semantic_clustering,
        "semantic_cluster_count": semantic_cluster_count,
        "semantic_cluster_ratio": semantic_cluster_ratio,
        "semantic_cluster_max_clusters": semantic_cluster_max_clusters,
        "semantic_cluster_tfidf_max_features": semantic_cluster_tfidf_max_features,
        "source_reader": "hdfs_session" if _is_hdfs_log(source_log) else "line_prefix",
        "target_reader": "hdfs_session" if _is_hdfs_log(target_log) else "line_prefix",
    }
    if has_current_sequence_cache(source_out, target_out, preprocess_config):
        print(f"[main] found cached sequence files: {source_out}, {target_out}", flush=True)
        return source_out, target_out
    print("[sequence] loading raw source/target logs", flush=True)
    print(
        f"[sequence] message_start: source={source_start}, target={target_start}",
        flush=True,
    )
    source_messages, source_labels, source_sessions = read_labeled_messages_with_sessions(
        source_log, max_lines=max_lines, num_fields=num_fields, message_start=source_start
    )
    target_messages, target_labels, target_sessions = read_labeled_messages_with_sessions(
        target_log, max_lines=max_lines, num_fields=num_fields, message_start=target_start
    )
    template_counter, _ = collect_template_statistics(source_messages, target_messages)
    print(
        f"[sequence] loaded source={len(source_messages):,}, target={len(target_messages):,}; "
        "building shared templates",
        flush=True,
    )
    vocab = build_shared_template_vocab(
        source_messages,
        target_messages,
        min_freq_ratio,
        max_events,
        num_hash_buckets,
        template_counter=template_counter,
    )
    cluster_info = build_semantic_template_clusters(
        template_counter,
        vocab,
        enabled=semantic_clustering,
        cluster_count=semantic_cluster_count,
        cluster_ratio=semantic_cluster_ratio,
        max_clusters=semantic_cluster_max_clusters,
        tfidf_max_features=semantic_cluster_tfidf_max_features,
        random_seed=semantic_cluster_random_seed,
    )
    source_sequence = encode_messages(source_messages, vocab, num_hash_buckets)
    target_sequence = encode_messages(target_messages, vocab, num_hash_buckets)
    np.save(source_out, source_sequence)
    np.save(target_out, target_sequence)
    np.save(label_path(source_out), source_labels)
    np.save(label_path(target_out), target_labels)
    for path, sessions in ((source_out, source_sessions), (target_out, target_sessions)):
        session_file = session_path(path)
        if sessions is None:
            if os.path.exists(session_file):
                os.remove(session_file)
        else:
            np.save(session_file, sessions)
    template_path = os.path.join(output_dir, "shared_sequence_templates.json")
    with open(template_path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    cluster_path = os.path.join(output_dir, "shared_sequence_clusters.json")
    with open(cluster_path, "w", encoding="utf-8") as file:
        json.dump(cluster_info, file, ensure_ascii=False, indent=2)
    metadata = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "num_events": len(vocab),
        "num_hash_buckets": num_hash_buckets,
        "template_path": template_path,
        "cluster_path": cluster_path,
        "preprocess": preprocess_config,
    }
    for path, sequence, labels, sessions in [
        (source_out, source_sequence, source_labels, source_sessions),
        (target_out, target_sequence, target_labels, target_sessions),
    ]:
        item = {**metadata, "length": int(len(sequence)), "alert_lines": int(labels.sum())}
        if sessions is not None:
            item["session_path"] = session_path(path)
            item["num_sessions"] = int(len(sessions))
        with open(_meta_path(path), "w", encoding="utf-8") as file:
            json.dump(item, file, indent=2)
    print(f"[sequence] saved source={source_out}, shape={source_sequence.shape}, target={target_out}, shape={target_sequence.shape}, num_events={len(vocab)}", flush=True)
    print(f"[sequence] saved labels: source_anomaly_lines={int(source_labels.sum()):,}, target_anomaly_lines={int(target_labels.sum()):,}", flush=True)
    return source_out, target_out


def num_events_from_meta(sequence_path: str) -> int:
    with open(_meta_path(sequence_path), "r", encoding="utf-8") as file:
        return int(json.load(file)["num_events"])
