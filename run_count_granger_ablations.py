
import argparse
import copy
import csv
import json
import os
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

from count_granger_main import deep_update, load_config, run_train, safe_name


DEFAULT_ABLATIONS = [
    "full",
    "no_cluster",
    "target_only",
    "source_only",
    "edge_only",
    "residual_only",
    "robust_residual_only",
    "edge_consistency_only",
    "causal_score",
    "combined_score",
    "validation_best",
    "stratified_balanced",
    "no_score_selection",
    "no_precision_threshold",
    "chronological",
]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ablation_path(ablation_dir: str, name: str) -> str:
    return os.path.join(ablation_dir, name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")


def pair_name(source_data: Optional[str], target_data: str) -> str:
    return f"{safe_name(source_data)}_to_{safe_name(target_data)}" if source_data else safe_name(target_data)


def flatten_result(name: str, status: str, summary: Optional[Dict[str, Any]], error: Optional[str], config_path: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ablation": name,
        "status": status,
        "config_path": config_path,
        "error": error or "",
    }
    if not summary:
        return row
    fit = summary.get("fit_info", {})
    test = summary.get("test_metrics", {})
    val = summary.get("validation_metrics", {})
    diagnostics = summary.get("test_diagnostics", {})
    num_features = int(fit.get("num_features", 0) or 0)
    num_edges = int(fit.get("num_edges", 0) or 0)
    edge_density = float(num_edges / max(num_features * num_features, 1))
    row.update(
        {
            "output_dir": summary.get("config", {}).get("output_dir", ""),
            "cache_dir": summary.get("config", {}).get("cache_dir", ""),
            "selected_score_component": summary.get("selected_score_component", ""),
            "threshold": summary.get("threshold", ""),
            "test_precision": test.get("precision", ""),
            "test_recall": test.get("recall", ""),
            "test_f1": test.get("f1", ""),
            "test_roc_auc": test.get("roc_auc", ""),
            "test_pr_auc": test.get("pr_auc", ""),
            "val_precision": val.get("precision", ""),
            "val_recall": val.get("recall", ""),
            "val_f1": val.get("f1", ""),
            "num_features": num_features,
            "num_edges": num_edges,
            "edge_density": edge_density,
            "test_normal_p95": diagnostics.get("normal_p95", ""),
            "test_anomaly_p05": diagnostics.get("anomaly_p05", ""),
        }
    )
    return row


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = [
        "ablation",
        "status",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "selected_score_component",
        "num_features",
        "num_edges",
        "edge_density",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_one(
    *,
    name: str,
    override_path: str,
    base_config: Dict[str, Any],
    source_data: Optional[str],
    target_data: str,
    run_dir: str,
    cache_root: str,
    save_summary: bool = True,
) -> Dict[str, Any]:
    override = load_yaml(override_path)
    ablation_meta = override.get("ablation", {}) if isinstance(override.get("ablation"), dict) else {}
    run_options = override.get("run", {}) if isinstance(override.get("run"), dict) else {}
    use_source = bool(run_options.get("use_source", True))
    effective_source = source_data if use_source else None
    effective_config = copy.deepcopy(base_config)
    deep_update(effective_config, override)
    effective_config["output_dir"] = os.path.join(run_dir, name)
    effective_config["cache_root"] = cache_root
    effective_config["cache_dir"] = os.path.join(cache_root, pair_name(effective_source, target_data))
    effective_config.setdefault("run", {})["save_summary"] = bool(save_summary)
    os.makedirs(effective_config["output_dir"], exist_ok=True)
    os.makedirs(effective_config["cache_dir"], exist_ok=True)
    save_yaml(os.path.join(effective_config["output_dir"], "effective_config.yaml"), effective_config)
    args = SimpleNamespace(source_data=effective_source, target_data=target_data, data=None, mode="train")
    print("=" * 80, flush=True)
    print(f"[count-ablation] running {name}: {ablation_meta.get('description', '')}", flush=True)
    print(f"[count-ablation] source={effective_source}, target={target_data}", flush=True)
    summary = run_train(args, effective_config)
    summary["ablation_name"] = name
    summary["ablation_config_path"] = override_path
    summary["ablation_description"] = ablation_meta.get("description", "")
    if save_summary:
        save_json(os.path.join(effective_config["output_dir"], "ablation_summary.json"), summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Count-Granger ablation experiments")
    parser.add_argument("--source_data", default=None, help="Optional source log path")
    parser.add_argument("--target_data", default=None, help="Target log path")
    parser.add_argument("--data", default=None, help="Single dataset alias for --target_data")
    parser.add_argument("--base_config", default="configs/count_granger_config.yaml")
    parser.add_argument("--ablation_dir", default="configs/count_granger_ablations")
    parser.add_argument("--ablations", nargs="*", default=None, help="Ablation names without .yaml; defaults to the standard suite")
    parser.add_argument("--output_root", default="results/count_granger_ablations")
    parser.add_argument("--cache_root", default="results/cache/count_granger_ablations")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--no_summary", action="store_true", help="Do not write per-run summary files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_data = args.target_data or args.data
    if not target_data:
        raise ValueError("Provide --target_data or --data")
    ablations = args.ablations or DEFAULT_ABLATIONS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{pair_name(args.source_data, target_data)}"
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    base_config = load_config(args.base_config)
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    save_json(
        os.path.join(run_dir, "run_context.json"),
        {
            "source_data": args.source_data,
            "target_data": target_data,
            "base_config": args.base_config,
            "ablation_dir": args.ablation_dir,
            "ablations": ablations,
            "output_root": args.output_root,
            "cache_root": args.cache_root,
        },
    )
    for ablation in ablations:
        name = os.path.splitext(os.path.basename(ablation))[0]
        path = ablation_path(args.ablation_dir, ablation)
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ablation config not found: {path}")
            summary = run_one(
                name=name,
                override_path=path,
                base_config=base_config,
                source_data=args.source_data,
                target_data=target_data,
                run_dir=run_dir,
                cache_root=args.cache_root,
                save_summary=not args.no_summary,
            )
            summaries[name] = summary
            rows.append(flatten_result(name, "ok", summary, None, path))
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"[count-ablation] FAILED {name}: {error}", flush=True)
            rows.append(flatten_result(name, "failed", None, error, path))
            if args.stop_on_error:
                raise
    if not args.no_summary:
        write_csv(os.path.join(run_dir, "summary.csv"), rows)
        write_markdown(os.path.join(run_dir, "summary.md"), rows)
        save_json(os.path.join(run_dir, "summary.json"), {"rows": rows, "summaries": summaries})
        print("=" * 80, flush=True)
        print(f"[count-ablation] summary: {os.path.join(run_dir, 'summary.csv')}", flush=True)
        print(f"[count-ablation] markdown: {os.path.join(run_dir, 'summary.md')}", flush=True)
    else:
        print("[count-ablation] summary files disabled", flush=True)


if __name__ == "__main__":
    main()
