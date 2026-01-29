import os, glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['xtick.major.width'] = 1.0
matplotlib.rcParams['ytick.major.width'] = 1.0
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.minor.width'] = 0.5

base_dir = "results"
model_dirname = "GSAI-ML__LLaDA-8B-Base"
acc_datasets  = ["piqa", "winogrande", "arc_challenge"]
pass_datasets = ["humaneval", "mbpp"]
precisions    = ["bf16", "fp16", "fp32"]
run_suffix = "run1"

# =======================================
precisions_type3 = ["int8", "bf16", "fp16"]

precisions_type4 = [
    ("bf16", lambda ds, p: f"{ds}_bf16_{run_suffix}"),
    ("fp16", lambda ds, p: f"{ds}_fp16_{run_suffix}"),
    ("fp32", lambda ds, p: f"{ds}_fp32_{run_suffix}"),
    ("bf16_fp32", lambda ds, p: f"{ds}_bf16_fp32_{run_suffix}"),
    ("fp16_fp32", lambda ds, p: f"{ds}_fp16_fp32_{run_suffix}"),
    ("int8_fp32", lambda ds, p: f"{ds}_int8_fp32_{run_suffix}"),
]


def load_json_any(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin1") as f:
            return json.load(f)


def find_results_json(run_dir):
    matches = glob.glob(os.path.join(run_dir, "results*.json"))
    return sorted(matches)[0] if matches else None


def pick_metric_value(results_block: dict, ds: str):
    if ds in acc_datasets:
        for k in ["acc,none", "accuracy", "acc"]:
            if k in results_block and isinstance(results_block[k], (int, float)):
                return float(results_block[k])
    if ds == "humaneval":
        for k in ["pass@1,create_test", "pass@1,none", "pass@1"]:
            if k in results_block and isinstance(results_block[k], (int, float)):
                return float(results_block[k])
    if ds == "mbpp":
        for k in ["pass_at_1,none", "pass@1,none", "pass@1", "pass_at_1"]:
            if k in results_block and isinstance(results_block[k], (int, float)):
                return float(results_block[k])
    for v in results_block.values():
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)):
                    return float(vv)
    return None


def collect_precision_values_type1(ds: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    vals = {}
    for prec in precisions:
        run_name = f"{ds}_{prec}_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    vals[prec] = pick_metric_value(data["results"][ds], ds)
    return vals


def collect_precision_values_type2(ds: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    vals = {}
    for prec in precisions:
        if prec == "fp32":
            run_name = f"{ds}_fp32_{run_suffix}"
        else:
            run_name = f"{ds}_{prec}_fp32_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    vals[prec] = pick_metric_value(data["results"][ds], ds)
    return vals


def collect_precision_values_type3(ds: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    vals = {}
    for prec in precisions_type3:
        run_name = f"{ds}_{prec}_fp32_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    vals[prec] = pick_metric_value(data["results"][ds], ds)
    return vals


def collect_precision_values_type4(ds: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    vals = {}
    for prec_name, run_name_func in precisions_type4:
        run_name = run_name_func(ds, prec_name)
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    vals[prec_name] = pick_metric_value(data["results"][ds], ds)
    return vals


def print_precision_comparison(acc_data, pass_data, out_root: Path, comparison_type: str, 
                               sample_df_acc: Optional[pd.DataFrame] = None, 
                               sample_df_pass: Optional[pd.DataFrame] = None):
    """ """
    def _print(data_dict, datasets, name, sample_df: Optional[pd.DataFrame] = None):
        rows = []
        for ds in datasets:
            vals = data_dict.get(ds, {})
            if not vals:
                continue
            
            if comparison_type == "type3":
                int8 = vals.get("int8", None)
                bf16 = vals.get("bf16", None)
                fp16 = vals.get("fp16", None)
                prec_values = [v for v in [int8, bf16, fp16] if v is not None]
            elif comparison_type == "type4":
                bf16 = vals.get("bf16", None)
                fp16 = vals.get("fp16", None)
                fp32 = vals.get("fp32", None)
                bf16_fp32 = vals.get("bf16_fp32", None)
                fp16_fp32 = vals.get("fp16_fp32", None)
                int8_fp32 = vals.get("int8_fp32", None)
                prec_values = [v for v in [bf16, fp16, fp32, bf16_fp32, fp16_fp32, int8_fp32] if v is not None]
            else:
                bf16 = vals.get("bf16", None)
                fp16 = vals.get("fp16", None)
                fp32 = vals.get("fp32", None)
                prec_values = [v for v in [bf16, fp16, fp32] if v is not None]
            if len(prec_values) > 0:
                mean_val = float(np.mean(prec_values))
                std_val = float(np.std(prec_values, ddof=1)) if len(prec_values) > 1 else 0.0
                n_prec = len(prec_values)
                se_n = std_val / np.sqrt(n_prec) if n_prec > 0 else 0.0
                
                n_unique = 0
                if sample_df is not None:
                    ds_samples = sample_df[sample_df["dataset"] == ds]
                    if not ds_samples.empty:
                        n_unique = ds_samples["doc_id"].nunique()
                
                se_n_unique = std_val / np.sqrt(n_unique) if n_unique > 0 else 0.0
            else:
                mean_val = None
                std_val = None
                se_n = None
                se_n_unique = None
                n_prec = 0
                n_unique = 0
            
            if comparison_type == "type3":
                rows.append({
                    "dataset": ds,
                    "int8": int8,
                    "bf16": bf16,
                    "fp16": fp16,
                    "mean": mean_val,
                    "std": std_val,
                    "se_n": se_n,
                    "se_n_unique": se_n_unique,
                    "n": n_prec,
                    "n_unique": n_unique,
                })
            elif comparison_type == "type4":
                rows.append({
                    "dataset": ds,
                    "bf16": bf16,
                    "fp16": fp16,
                    "fp32": fp32,
                    "bf16_fp32": bf16_fp32,
                    "fp16_fp32": fp16_fp32,
                    "int8_fp32": int8_fp32,
                    "mean": mean_val,
                    "std": std_val,
                    "se_n": se_n,
                    "se_n_unique": se_n_unique,
                    "n": n_prec,
                    "n_unique": n_unique,
                })
            else:
                rows.append({
                    "dataset": ds,
                    "bf16": bf16,
                    "fp16": fp16,
                    "fp32": fp32,
                    "mean": mean_val,
                    "std": std_val,
                    "se_n": se_n,
                    "se_n_unique": se_n_unique,
                    "n": n_prec,
                    "n_unique": n_unique,
                })
        
        if rows:
            df = pd.DataFrame(rows)
            print(f"\n[{name.upper()} PRECISION DATASET-WISE COMPARISON ({comparison_type.upper()})]")
            formatters = {
                "mean": "{:.4f}".format,
                "std": "{:.4f}".format,
                "se_n": "{:.6f}".format,
                "se_n_unique": "{:.6f}".format,
            }
            if comparison_type == "type3":
                formatters.update({
                    "int8": "{:.4f}".format,
                    "bf16": "{:.4f}".format,
                    "fp16": "{:.4f}".format,
                })
            elif comparison_type == "type4":
                formatters.update({
                    "bf16": "{:.4f}".format,
                    "fp16": "{:.4f}".format,
                    "fp32": "{:.4f}".format,
                    "bf16_fp32": "{:.4f}".format,
                    "fp16_fp32": "{:.4f}".format,
                    "int8_fp32": "{:.4f}".format,
                })
            else:
                formatters.update({
                    "bf16": "{:.4f}".format,
                    "fp16": "{:.4f}".format,
                    "fp32": "{:.4f}".format,
                })
            print(df.to_string(index=False, formatters=formatters))
            df.to_csv(out_root / f"precision_datasetwise_{name}_{comparison_type}.csv", index=False)

    _print(acc_data, acc_datasets, "acc", sample_df_acc)
    _print(pass_data, pass_datasets, "pass", sample_df_pass)


def resolve_target_idx(obj: dict) -> Optional[int]:
    doc = obj.get("doc", {})
    label = doc.get("label")
    if isinstance(label, (int, float)) and label in (0, 1):
        return int(label)
    if isinstance(label, str) and label in ("0", "1"):
        return int(label)
    ans = doc.get("answer")
    if isinstance(ans, str) and ans in ("1", "2"):
        return int(ans) - 1
    t = obj.get("target")
    if isinstance(t, str) and t.isdigit():
        return int(t)
    answer_key = doc.get("answerKey")
    choices = doc.get("choices", {})
    labels = choices.get("label") if isinstance(choices, dict) else None
    if isinstance(answer_key, str) and isinstance(labels, list):
        try:
            return labels.index(answer_key)
        except ValueError:
            try:
                upper_labels = [str(x).upper() for x in labels]
                return upper_labels.index(answer_key.upper())
            except ValueError:
                return None
    return None


def extract_ll_list(obj: dict) -> Optional[List[float]]:
    filt = obj.get("filtered_resps")
    if not isinstance(filt, list) or len(filt) == 0:
        return None
    ll_list = []
    for item in filt:
        try:
            ll_list.append(float(item[0]))
        except Exception:
            return None
    return ll_list


def load_samplewise_rows_type1(ds: str, precision: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    run_name = f"{ds}_{precision}_{run_suffix}"
    if prefix:
        run_name = f"{prefix}{run_name}"
    run_dir = Path(base_dir) / run_name / model_dir
    files = sorted(run_dir.glob(f"samples_{ds}_*.jsonl"))
    if not files:
        return pd.DataFrame()
    path = files[-1]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is None:
                continue

            if ds in acc_datasets:
                target_idx = resolve_target_idx(obj)
                ll_list = extract_ll_list(obj)
                if target_idx is None or ll_list is None:
                    continue
                n = len(ll_list)
                if target_idx < 0 or target_idx >= n:
                    continue
                pred_idx = int(np.argmax(ll_list))
                correct = int(pred_idx == target_idx)
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "label": int(target_idx),
                    "pred": pred_idx,
                    "correct": correct,
                    "precision": precision,
                })
            else:
                p = obj.get("pass@1", obj.get("pass_at_1"))
                if p is None:
                    continue
                try:
                    pass_flag = int(round(float(p)))
                except Exception:
                    continue
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "pass_flag": pass_flag,
                    "precision": precision,
                })
    return pd.DataFrame(rows)


def load_samplewise_rows_type2(ds: str, precision: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    if precision == "fp32":
        run_name = f"{ds}_fp32_{run_suffix}"
    else:
        run_name = f"{ds}_{precision}_fp32_{run_suffix}"
    if prefix:
        run_name = f"{prefix}{run_name}"
    run_dir = Path(base_dir) / run_name / model_dir
    files = sorted(run_dir.glob(f"samples_{ds}_*.jsonl"))
    if not files:
        return pd.DataFrame()
    path = files[-1]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is None:
                continue

            if ds in acc_datasets:
                target_idx = resolve_target_idx(obj)
                ll_list = extract_ll_list(obj)
                if target_idx is None or ll_list is None:
                    continue
                n = len(ll_list)
                if target_idx < 0 or target_idx >= n:
                    continue
                pred_idx = int(np.argmax(ll_list))
                correct = int(pred_idx == target_idx)
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "label": int(target_idx),
                    "pred": pred_idx,
                    "correct": correct,
                    "precision": precision,
                })
            else:
                p = obj.get("pass@1", obj.get("pass_at_1"))
                if p is None:
                    continue
                try:
                    pass_flag = int(round(float(p)))
                except Exception:
                    continue
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "pass_flag": pass_flag,
                    "precision": precision,
                })
    return pd.DataFrame(rows)


def load_samplewise_rows_type3(ds: str, precision: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    run_name = f"{ds}_{precision}_fp32_{run_suffix}"
    if prefix:
        run_name = f"{prefix}{run_name}"
    run_dir = Path(base_dir) / run_name / model_dir
    files = sorted(run_dir.glob(f"samples_{ds}_*.jsonl"))
    if not files:
        return pd.DataFrame()
    path = files[-1]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is None:
                continue

            if ds in acc_datasets:
                target_idx = resolve_target_idx(obj)
                ll_list = extract_ll_list(obj)
                if target_idx is None or ll_list is None:
                    continue
                n = len(ll_list)
                if target_idx < 0 or target_idx >= n:
                    continue
                pred_idx = int(np.argmax(ll_list))
                correct = int(pred_idx == target_idx)
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "label": int(target_idx),
                    "pred": pred_idx,
                    "correct": correct,
                    "precision": precision,
                })
            else:
                p = obj.get("pass@1", obj.get("pass_at_1"))
                if p is None:
                    continue
                try:
                    pass_flag = int(round(float(p)))
                except Exception:
                    continue
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "pass_flag": pass_flag,
                    "precision": precision,
                })
    return pd.DataFrame(rows)


def load_samplewise_rows_type4(ds: str, precision_name: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    run_name_func = None
    for prec_name, func in precisions_type4:
        if prec_name == precision_name:
            run_name_func = func
            break
    if run_name_func is None:
        return pd.DataFrame()
    
    run_name = run_name_func(ds, precision_name)
    if prefix:
        run_name = f"{prefix}{run_name}"
    run_dir = Path(base_dir) / run_name / model_dir
    files = sorted(run_dir.glob(f"samples_{ds}_*.jsonl"))
    if not files:
        return pd.DataFrame()
    path = files[-1]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is None:
                continue

            if ds in acc_datasets:
                target_idx = resolve_target_idx(obj)
                ll_list = extract_ll_list(obj)
                if target_idx is None or ll_list is None:
                    continue
                n = len(ll_list)
                if target_idx < 0 or target_idx >= n:
                    continue
                pred_idx = int(np.argmax(ll_list))
                correct = int(pred_idx == target_idx)
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "label": int(target_idx),
                    "pred": pred_idx,
                    "correct": correct,
                    "precision": precision_name,
                })
            else:
                p = obj.get("pass@1", obj.get("pass_at_1"))
                if p is None:
                    continue
                try:
                    pass_flag = int(round(float(p)))
                except Exception:
                    continue
                rows.append({
                    "dataset": ds,
                    "doc_id": doc_id,
                    "pass_flag": pass_flag,
                    "precision": precision_name,
                })
    return pd.DataFrame(rows)


def summarize_samplewise_precision_type1(prefix: str = "", model_dir: str = None):
    """ """
    batch_results = []
    sample_results = []
    dataset_results = []

    # Accuracy datasets
    for ds in acc_datasets:
        prec_tables = {}
        for prec in precisions:
            prec_tables[prec] = load_samplewise_rows_type1(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        # Sample-level
        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "label", "pred", "correct"]].copy()
            sub = sub.rename(columns={
                "pred": f"pred_{prec}",
                "correct": f"correct_{prec}",
            })
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on=["doc_id", "label"], how="outer"
            )

        correct_cols = [c for c in merged_samples.columns if c.startswith("correct_")]
        merged_samples["sample_acc"] = merged_samples[correct_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples = merged_samples.rename(columns={"label": "truth"})
        sample_results.append(merged_samples)

        # Batch-level (per precision)
        for prec, df in prec_tables.items():
            mean_acc = df["correct"].mean()
            std_acc  = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
            n_samples = len(df)
            se_acc = std_acc / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_acc,
                "std": std_acc,
                "se": se_acc,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_acc = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_acc,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    # Pass datasets
    for ds in pass_datasets:
        prec_tables = {}
        for prec in precisions:
            prec_tables[prec] = load_samplewise_rows_type1(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "pass_flag"]].copy().rename(columns={"pass_flag": f"pass_{prec}"})
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on="doc_id", how="outer"
            )

        pass_cols = [c for c in merged_samples.columns if c.startswith("pass_")]
        merged_samples["sample_acc"] = merged_samples[pass_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples["truth"] = np.nan
        sample_results.append(merged_samples)

        for prec, df in prec_tables.items():
            mean_pass = df["pass_flag"].mean()
            std_pass  = df["pass_flag"].std(ddof=1) if len(df["pass_flag"]) > 1 else 0.0
            n_samples = len(df)
            se_pass = std_pass / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_pass,
                "std": std_pass,
                "se": se_pass,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_pass = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_pass,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    batch_df = pd.DataFrame(batch_results)
    sample_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_results)
    return batch_df, sample_df, dataset_df


def summarize_samplewise_precision_type2(prefix: str = "", model_dir: str = None):
    """ """
    batch_results = []
    sample_results = []
    dataset_results = []

    # Accuracy datasets
    for ds in acc_datasets:
        prec_tables = {}
        for prec in precisions:
            prec_tables[prec] = load_samplewise_rows_type2(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        # Sample-level
        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "label", "pred", "correct"]].copy()
            sub = sub.rename(columns={
                "pred": f"pred_{prec}",
                "correct": f"correct_{prec}",
            })
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on=["doc_id", "label"], how="outer"
            )

        correct_cols = [c for c in merged_samples.columns if c.startswith("correct_")]
        merged_samples["sample_acc"] = merged_samples[correct_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples = merged_samples.rename(columns={"label": "truth"})
        sample_results.append(merged_samples)

        # Batch-level (per precision)
        for prec, df in prec_tables.items():
            mean_acc = df["correct"].mean()
            std_acc  = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
            n_samples = len(df)
            se_acc = std_acc / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_acc,
                "std": std_acc,
                "se": se_acc,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_acc = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_acc,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    # Pass datasets
    for ds in pass_datasets:
        prec_tables = {}
        for prec in precisions:
            prec_tables[prec] = load_samplewise_rows_type2(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "pass_flag"]].copy().rename(columns={"pass_flag": f"pass_{prec}"})
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on="doc_id", how="outer"
            )

        pass_cols = [c for c in merged_samples.columns if c.startswith("pass_")]
        merged_samples["sample_acc"] = merged_samples[pass_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples["truth"] = np.nan
        sample_results.append(merged_samples)

        for prec, df in prec_tables.items():
            mean_pass = df["pass_flag"].mean()
            std_pass  = df["pass_flag"].std(ddof=1) if len(df["pass_flag"]) > 1 else 0.0
            n_samples = len(df)
            se_pass = std_pass / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_pass,
                "std": std_pass,
                "se": se_pass,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_pass = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_pass,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    batch_df = pd.DataFrame(batch_results)
    sample_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_results)
    return batch_df, sample_df, dataset_df


def summarize_samplewise_precision_type3(prefix: str = "", model_dir: str = None):
    """ """
    batch_results = []
    sample_results = []
    dataset_results = []

    # Accuracy datasets
    for ds in acc_datasets:
        prec_tables = {}
        for prec in precisions_type3:
            prec_tables[prec] = load_samplewise_rows_type3(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        # Sample-level
        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "label", "pred", "correct"]].copy()
            sub = sub.rename(columns={
                "pred": f"pred_{prec}",
                "correct": f"correct_{prec}",
            })
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on=["doc_id", "label"], how="outer"
            )

        correct_cols = [c for c in merged_samples.columns if c.startswith("correct_")]
        merged_samples["sample_acc"] = merged_samples[correct_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples = merged_samples.rename(columns={"label": "truth"})
        sample_results.append(merged_samples)

        # Batch-level (per precision)
        for prec, df in prec_tables.items():
            mean_acc = df["correct"].mean()
            std_acc  = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
            n_samples = len(df)
            se_acc = std_acc / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_acc,
                "std": std_acc,
                "se": se_acc,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_acc = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_acc,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    # Pass datasets
    for ds in pass_datasets:
        prec_tables = {}
        for prec in precisions_type3:
            prec_tables[prec] = load_samplewise_rows_type3(ds, prec, prefix, model_dir)
        if any(df.empty for df in prec_tables.values()):
            continue

        merged_samples = None
        for prec, df in prec_tables.items():
            sub = df[["doc_id", "pass_flag"]].copy().rename(columns={"pass_flag": f"pass_{prec}"})
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on="doc_id", how="outer"
            )

        pass_cols = [c for c in merged_samples.columns if c.startswith("pass_")]
        merged_samples["sample_acc"] = merged_samples[pass_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples["truth"] = np.nan
        sample_results.append(merged_samples)

        for prec, df in prec_tables.items():
            mean_pass = df["pass_flag"].mean()
            std_pass  = df["pass_flag"].std(ddof=1) if len(df["pass_flag"]) > 1 else 0.0
            n_samples = len(df)
            se_pass = std_pass / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec,
                "n": n_samples,
                "consistency_acc": mean_pass,
                "std": std_pass,
                "se": se_pass,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_pass = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_pass,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    batch_df = pd.DataFrame(batch_results)
    sample_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_results)
    return batch_df, sample_df, dataset_df


def summarize_samplewise_precision_type4(prefix: str = "", model_dir: str = None):
    """ """
    batch_results = []
    sample_results = []
    dataset_results = []

    # Accuracy datasets
    for ds in acc_datasets:
        prec_tables = {}
        for prec_name, _ in precisions_type4:
            prec_tables[prec_name] = load_samplewise_rows_type4(ds, prec_name, prefix, model_dir)
        prec_tables = {k: v for k, v in prec_tables.items() if not v.empty}
        if not prec_tables:
            continue

        # Sample-level
        merged_samples = None
        for prec_name, df in prec_tables.items():
            sub = df[["doc_id", "label", "pred", "correct"]].copy()
            sub = sub.rename(columns={
                "pred": f"pred_{prec_name}",
                "correct": f"correct_{prec_name}",
            })
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on=["doc_id", "label"], how="outer"
            )

        correct_cols = [c for c in merged_samples.columns if c.startswith("correct_")]
        merged_samples["sample_acc"] = merged_samples[correct_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples = merged_samples.rename(columns={"label": "truth"})
        sample_results.append(merged_samples)

        # Batch-level (per precision)
        for prec_name, df in prec_tables.items():
            mean_acc = df["correct"].mean()
            std_acc  = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
            n_samples = len(df)
            se_acc = std_acc / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec_name,
                "n": n_samples,
                "consistency_acc": mean_acc,
                "std": std_acc,
                "se": se_acc,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_acc = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_acc,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    # Pass datasets
    for ds in pass_datasets:
        prec_tables = {}
        for prec_name, _ in precisions_type4:
            prec_tables[prec_name] = load_samplewise_rows_type4(ds, prec_name, prefix, model_dir)
        prec_tables = {k: v for k, v in prec_tables.items() if not v.empty}
        if not prec_tables:
            continue

        merged_samples = None
        for prec_name, df in prec_tables.items():
            sub = df[["doc_id", "pass_flag"]].copy().rename(columns={"pass_flag": f"pass_{prec_name}"})
            merged_samples = sub if merged_samples is None else merged_samples.merge(
                sub, on="doc_id", how="outer"
            )

        pass_cols = [c for c in merged_samples.columns if c.startswith("pass_")]
        merged_samples["sample_acc"] = merged_samples[pass_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples["truth"] = np.nan
        sample_results.append(merged_samples)

        for prec_name, df in prec_tables.items():
            mean_pass = df["pass_flag"].mean()
            std_pass  = df["pass_flag"].std(ddof=1) if len(df["pass_flag"]) > 1 else 0.0
            n_samples = len(df)
            se_pass = std_pass / np.sqrt(n_samples) if n_samples > 0 else 0.0
            batch_results.append({
                "dataset": ds,
                "precision": prec_name,
                "n": n_samples,
                "consistency_acc": mean_pass,
                "std": std_pass,
                "se": se_pass,
            })

        # Dataset-level summary (sample-level: based on sample accuracies)
        sample_accs = merged_samples["sample_acc"].dropna().tolist()
        if sample_accs:
            mean_pass = float(np.mean(sample_accs))
            std = float(np.std(sample_accs, ddof=1)) if len(sample_accs) > 1 else 0.0
            n_unique = merged_samples["doc_id"].nunique()
            total_samples = sum(len(df) for df in prec_tables.values())
            se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
            se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
            dataset_results.append({
                "dataset": ds,
                "consistency_acc": mean_pass,
                "std": std,
                "se_n": se_n,
                "se_n_unique": se_n_unique,
                "n": len(prec_tables),
                "n_unique": n_unique,
                "total_samples": total_samples,
            })

    batch_df = pd.DataFrame(batch_results)
    sample_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_results)
    return batch_df, sample_df, dataset_df


# =============== process_model ===============
def process_model(prefix: str = "", model_dir: str = None, model_name: str = ""):
    """ """
    if model_name:
        out_root = Path("figs_precision_compare") / model_name
    else:
        out_root = Path("figs_precision_compare")
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f"TYPE 1: Model and Softmax Same Precision ({model_name if model_name else 'default'})")
    print("="*60)
    
    batch_df_type1, sample_df_type1, dataset_df_type1 = summarize_samplewise_precision_type1(prefix, model_dir)
    
    # dataset-wise
    acc_data_type1 = {ds: collect_precision_values_type1(ds, prefix, model_dir) for ds in acc_datasets}
    pass_data_type1 = {ds: collect_precision_values_type1(ds, prefix, model_dir) for ds in pass_datasets}
    sample_df_acc_type1 = sample_df_type1[sample_df_type1["dataset"].isin(acc_datasets)] if not sample_df_type1.empty else None
    sample_df_pass_type1 = sample_df_type1[sample_df_type1["dataset"].isin(pass_datasets)] if not sample_df_type1.empty else None
    print_precision_comparison(acc_data_type1, pass_data_type1, out_root, "type1", 
                               sample_df_acc_type1, sample_df_pass_type1)
    if not batch_df_type1.empty:
        print("\n[PRECISION TYPE1 SAMPLE-WISE BATCH RESULTS]")
        print(batch_df_type1.to_string(index=False,
                                     formatters={
                                         "consistency_acc": "{:.4f}".format,
                                         "std": "{:.4f}".format,
                                         "se": "{:.6f}".format,
                                     }))
        batch_df_type1.to_csv(out_root / "precision_samplewise_batch_type1.csv", index=False)
    if not sample_df_type1.empty:
        print("\n[PRECISION TYPE1 SAMPLE-WISE PER-DOC RESULTS]")
        sample_df_type1.to_csv(out_root / "precision_samplewise_per_doc_type1.csv", index=False)
    if not dataset_df_type1.empty:
        print("\n[PRECISION TYPE1 SAMPLE-WISE DATASET SUMMARY]")
        print("(Mean and std of precision-level accuracies)")
        print(dataset_df_type1.to_string(index=False,
                                       formatters={
                                           "consistency_acc": "{:.4f}".format,
                                           "std": "{:.4f}".format,
                                           "se_n": "{:.6f}".format,
                                           "se_n_unique": "{:.6f}".format,
                                       }))
        dataset_df_type1.to_csv(out_root / "precision_samplewise_dataset_summary_type1.csv", index=False)

    print("\n" + "="*60)
    print(f"TYPE 2: Model Various Precision, Softmax Fixed FP32 ({model_name if model_name else 'default'})")
    print("="*60)
    
    batch_df_type2, sample_df_type2, dataset_df_type2 = summarize_samplewise_precision_type2(prefix, model_dir)
    
    # dataset-wise
    acc_data_type2 = {ds: collect_precision_values_type2(ds, prefix, model_dir) for ds in acc_datasets}
    pass_data_type2 = {ds: collect_precision_values_type2(ds, prefix, model_dir) for ds in pass_datasets}
    sample_df_acc_type2 = sample_df_type2[sample_df_type2["dataset"].isin(acc_datasets)] if not sample_df_type2.empty else None
    sample_df_pass_type2 = sample_df_type2[sample_df_type2["dataset"].isin(pass_datasets)] if not sample_df_type2.empty else None
    print_precision_comparison(acc_data_type2, pass_data_type2, out_root, "type2",
                               sample_df_acc_type2, sample_df_pass_type2)
    if not batch_df_type2.empty:
        print("\n[PRECISION TYPE2 SAMPLE-WISE BATCH RESULTS]")
        print(batch_df_type2.to_string(index=False,
                                     formatters={
                                         "consistency_acc": "{:.4f}".format,
                                         "std": "{:.4f}".format,
                                         "se": "{:.6f}".format,
                                     }))
        batch_df_type2.to_csv(out_root / "precision_samplewise_batch_type2.csv", index=False)
    if not sample_df_type2.empty:
        print("\n[PRECISION TYPE2 SAMPLE-WISE PER-DOC RESULTS]")
        sample_df_type2.to_csv(out_root / "precision_samplewise_per_doc_type2.csv", index=False)
    if not dataset_df_type2.empty:
        print("\n[PRECISION TYPE2 SAMPLE-WISE DATASET SUMMARY]")
        print("(Mean and std of precision-level accuracies)")
        print(dataset_df_type2.to_string(index=False,
                                       formatters={
                                           "consistency_acc": "{:.4f}".format,
                                           "std": "{:.4f}".format,
                                           "se_n": "{:.6f}".format,
                                           "se_n_unique": "{:.6f}".format,
                                       }))
        dataset_df_type2.to_csv(out_root / "precision_samplewise_dataset_summary_type2.csv", index=False)

    print("\n" + "="*60)
    print(f"TYPE 3: Compare All Datasets with int8_fp32, bf16_fp32, fp16_fp32 ({model_name if model_name else 'default'})")
    print("="*60)
    
    batch_df_type3, sample_df_type3, dataset_df_type3 = summarize_samplewise_precision_type3(prefix, model_dir)
    
    # dataset-wise
    acc_data_type3 = {ds: collect_precision_values_type3(ds, prefix, model_dir) for ds in acc_datasets}
    pass_data_type3 = {ds: collect_precision_values_type3(ds, prefix, model_dir) for ds in pass_datasets}
    sample_df_acc_type3 = sample_df_type3[sample_df_type3["dataset"].isin(acc_datasets)] if not sample_df_type3.empty else None
    sample_df_pass_type3 = sample_df_type3[sample_df_type3["dataset"].isin(pass_datasets)] if not sample_df_type3.empty else None
    print_precision_comparison(acc_data_type3, pass_data_type3, out_root, "type3",
                               sample_df_acc_type3, sample_df_pass_type3)
    if not batch_df_type3.empty:
        print("\n[PRECISION TYPE3 SAMPLE-WISE BATCH RESULTS]")
        print(batch_df_type3.to_string(index=False,
                                     formatters={
                                         "consistency_acc": "{:.4f}".format,
                                         "std": "{:.4f}".format,
                                         "se": "{:.6f}".format,
                                     }))
        batch_df_type3.to_csv(out_root / "precision_samplewise_batch_type3.csv", index=False)
    if not sample_df_type3.empty:
        print("\n[PRECISION TYPE3 SAMPLE-WISE PER-DOC RESULTS]")
        sample_df_type3.to_csv(out_root / "precision_samplewise_per_doc_type3.csv", index=False)
    if not dataset_df_type3.empty:
        print("\n[PRECISION TYPE3 SAMPLE-WISE DATASET SUMMARY]")
        print("(Mean and std of precision-level accuracies)")
        print(dataset_df_type3.to_string(index=False,
                                       formatters={
                                           "consistency_acc": "{:.4f}".format,
                                           "std": "{:.4f}".format,
                                           "se_n": "{:.6f}".format,
                                           "se_n_unique": "{:.6f}".format,
                                       }))
        dataset_df_type3.to_csv(out_root / "precision_samplewise_dataset_summary_type3.csv", index=False)

    print("\n" + "="*60)
    print(f"TYPE 4: Combine All Settings from Type 1, 2, 3 ({model_name if model_name else 'default'})")
    print("="*60)
    
    batch_df_type4, sample_df_type4, dataset_df_type4 = summarize_samplewise_precision_type4(prefix, model_dir)
    
    # dataset-wise
    acc_data_type4 = {ds: collect_precision_values_type4(ds, prefix, model_dir) for ds in acc_datasets}
    pass_data_type4 = {ds: collect_precision_values_type4(ds, prefix, model_dir) for ds in pass_datasets}
    sample_df_acc_type4 = sample_df_type4[sample_df_type4["dataset"].isin(acc_datasets)] if not sample_df_type4.empty else None
    sample_df_pass_type4 = sample_df_type4[sample_df_type4["dataset"].isin(pass_datasets)] if not sample_df_type4.empty else None
    print_precision_comparison(acc_data_type4, pass_data_type4, out_root, "type4",
                               sample_df_acc_type4, sample_df_pass_type4)
    if not batch_df_type4.empty:
        print("\n[PRECISION TYPE4 SAMPLE-WISE BATCH RESULTS]")
        print(batch_df_type4.to_string(index=False,
                                     formatters={
                                         "consistency_acc": "{:.4f}".format,
                                         "std": "{:.4f}".format,
                                         "se": "{:.6f}".format,
                                     }))
        batch_df_type4.to_csv(out_root / "precision_samplewise_batch_type4.csv", index=False)
    if not sample_df_type4.empty:
        print("\n[PRECISION TYPE4 SAMPLE-WISE PER-DOC RESULTS]")
        sample_df_type4.to_csv(out_root / "precision_samplewise_per_doc_type4.csv", index=False)
    if not dataset_df_type4.empty:
        print("\n[PRECISION TYPE4 SAMPLE-WISE DATASET SUMMARY]")
        print("(Mean and std of precision-level accuracies)")
        print(dataset_df_type4.to_string(index=False,
                                       formatters={
                                           "consistency_acc": "{:.4f}".format,
                                           "std": "{:.4f}".format,
                                           "se_n": "{:.6f}".format,
                                           "se_n_unique": "{:.6f}".format,
                                       }))
        dataset_df_type4.to_csv(out_root / "precision_samplewise_dataset_summary_type4.csv", index=False)

    print(f"\n[OK] Output dir: {out_root.resolve()}")


# =============== Calculate Flip Rate ===============
def calculate_flip_rate_for_model(ds: str, prefix: str, model_dir: str, model_name: str, is_acc: bool):
    """ """
    precisions_to_use = precisions_type3  # ["int8", "bf16", "fp16"]
    
    all_precision_data = {}
    for prec in precisions_to_use:
        df = load_samplewise_rows_type3(ds, prec, prefix=prefix, model_dir=model_dir)
        if not df.empty:
            if is_acc:
                all_precision_data[prec] = df[["doc_id", "pred"]]
            else:
                all_precision_data[prec] = df[["doc_id", "pass_flag"]]
    
    if not all_precision_data:
        return None
    
    common_doc_ids = set(all_precision_data[list(all_precision_data.keys())[0]]["doc_id"])
    for prec_df in all_precision_data.values():
        common_doc_ids &= set(prec_df["doc_id"])
    
    if not common_doc_ids:
        return None
    
    flip_rates = []
    n_configs = len(all_precision_data)
    
    for doc_id in common_doc_ids:
        pred_list = []
        for prec, prec_df in all_precision_data.items():
            row = prec_df[prec_df["doc_id"] == doc_id]
            if not row.empty:
                if is_acc:
                    pred_list.append(int(row.iloc[0]["pred"]))
                else:
                    pred_list.append(int(row.iloc[0]["pass_flag"]))
        
        if len(pred_list) != n_configs:
            continue
        
        pred_counts = Counter(pred_list)
        
        majority_count = max(pred_counts.values()) if pred_counts else 0
        
        # Prediction Flip Rate
        flip_rate = 1 - (majority_count / float(n_configs))
        flip_rates.append(flip_rate)
    
    if not flip_rates:
        return None
    
    mean_flip_rate = np.mean(flip_rates)
    std_flip_rate = np.std(flip_rates, ddof=1) if len(flip_rates) > 1 else 0.0
    n_samples = len(flip_rates)
    se_flip_rate = std_flip_rate / np.sqrt(n_samples) if n_samples > 0 else 0.0
    
    return {
        "model": model_name,
        "dataset": ds,
        "type": "accuracy" if is_acc else "pass",
        "n_configs": n_configs,
        "mean_flip_rate": mean_flip_rate,
        "std_flip_rate": std_flip_rate,
        "se_flip_rate": se_flip_rate,
        "n_samples": n_samples,
    }


def calculate_flip_rate():
    """ """
    print("\n" + "="*80)
    print("Prediction Flip Rate (LLaDA and LLaDA-1.5, by precision settings)")
    print("="*80)
    
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    results = []
    
    for ds in acc_datasets:
        print(f"\nDataset: {ds} (accuracy)")
        
        result_llada = calculate_flip_rate_for_model(
            ds, prefix="", model_dir=None, model_name="llada", is_acc=True
        )
        if result_llada:
            results.append(result_llada)
            print(f"  llada: n_configs={result_llada['n_configs']}, "
                  f"mean_flip_rate={result_llada['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada['n_samples']}")
        
        result_llada15 = calculate_flip_rate_for_model(
            ds, prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15", is_acc=True
        )
        if result_llada15:
            results.append(result_llada15)
            print(f"  llada15: n_configs={result_llada15['n_configs']}, "
                  f"mean_flip_rate={result_llada15['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada15['n_samples']}")
    
    for ds in pass_datasets:
        print(f"\nDataset: {ds} (pass)")
        
        result_llada = calculate_flip_rate_for_model(
            ds, prefix="", model_dir=None, model_name="llada", is_acc=False
        )
        if result_llada:
            results.append(result_llada)
            print(f"  llada: n_configs={result_llada['n_configs']}, "
                  f"mean_flip_rate={result_llada['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada['n_samples']}")
        
        result_llada15 = calculate_flip_rate_for_model(
            ds, prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15", is_acc=False
        )
        if result_llada15:
            results.append(result_llada15)
            print(f"  llada15: n_configs={result_llada15['n_configs']}, "
                  f"mean_flip_rate={result_llada15['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada15['n_samples']}")
    
    if results:
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("Flip Rate summary")
        print("="*80)
        print("\n" + results_df.to_string(index=False,
                                         formatters={
                                             "mean_flip_rate": "{:.6f}".format,
                                             "std_flip_rate": "{:.6f}".format,
                                             "se_flip_rate": "{:.6f}".format,
                                         }))
        
        out_dir = Path("figs_precision_compare")
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_dir / "flip_rate_by_precision.csv", index=False)
        print(f"\nSaved: {out_dir / 'flip_rate_by_precision.csv'}")
        
        plot_flip_rate_barplot(results_df, out_dir)
    else:
        print("\n[WARNING] No results")


def plot_flip_rate_barplot(results_df: pd.DataFrame, out_dir: Path):
    """ """
    acc_df = results_df[results_df["type"] == "accuracy"].copy()
    if acc_df.empty:
        print("[WARNING] No accuracy data for plot")
        return
    
    datasets = sorted(acc_df["dataset"].unique())
    models = ["llada", "llada15"]
    model_labels = {"llada": "LLaDA", "llada15": "LLaDA-1.5"}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    colors = ['#4472C4', '#ED7D31']
    
    for i, model in enumerate(models):
        values = []
        errors = []
        for ds in datasets:
            row = acc_df[(acc_df["dataset"] == ds) & (acc_df["model"] == model)]
            if not row.empty:
                values.append(row.iloc[0]["mean_flip_rate"])
                errors.append(row.iloc[0]["se_flip_rate"])
            else:
                values.append(0)
                errors.append(0)
        
        offset = (i - 0.5) * width
        label = model_labels.get(model, model.upper())
        bars = ax.bar(x + offset, values, width, label=label, 
                     color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.2)
        
        ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                   color='black', capsize=4, capthick=1.2, linewidth=1.2, elinewidth=1.2)
    
    ax.set_xlabel('Dataset', fontsize=16, fontweight='normal')
    ax.set_ylabel('Flip Rate', fontsize=16, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], fontsize=14)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=13, width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=0.5, length=2)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
             edgecolor='gray', framealpha=0.9, fontsize=14, handlelength=1.5)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    plt.tight_layout()
    
    outfile = out_dir / "flip_rate_barplot.pdf"
    plt.savefig(outfile, format='pdf', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print(f"Saved: {outfile}")


def plot_precision_type3_barplot(prefix: str = "", model_dir: str = None, model_name: str = ""):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    
    all_datasets = acc_datasets + pass_datasets
    results = []
    
    for ds in all_datasets:
        vals = collect_precision_values_type3(ds, prefix=prefix, model_dir=model_dir)
        if vals and any(v is not None for v in vals.values()):
            is_acc = ds in acc_datasets
            results.append({
                "dataset": ds,
                "type": "accuracy" if is_acc else "pass",
                "int8_fp32": vals.get("int8"),
                "bf16_fp32": vals.get("bf16"),
                "fp16_fp32": vals.get("fp16"),
            })
    
    if not results:
        print(f"[WARNING] No type3 data for model {model_name or 'default'}")
        return
    
    results_df = pd.DataFrame(results)
    
    for data_type in ["accuracy", "pass"]:
        type_df = results_df[results_df["type"] == data_type].copy()
        if type_df.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        datasets = sorted(type_df["dataset"].unique())
        x = np.arange(len(datasets))
        width = 0.25
        
        precisions_to_plot = ["int8_fp32", "bf16_fp32", "fp16_fp32"]
        
        colors = ['#4472C4', '#ED7D31', '#55A868']
        
        for i, prec in enumerate(precisions_to_plot):
            values = []
            for ds in datasets:
                row = type_df[type_df["dataset"] == ds]
                if not row.empty:
                    val = row.iloc[0][prec]
                    values.append(val if val is not None else 0)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            label = prec
            bars = ax.bar(x + offset, values, width, label=label, 
                         color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.2)
        
        ax.set_xlabel('Dataset', fontsize=16, fontweight='normal')
        ylabel = 'Accuracy' if data_type == "accuracy" else 'Pass@1'
        ax.set_ylabel(ylabel, fontsize=16, fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], fontsize=14)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=13, width=0.8, length=4)
        ax.tick_params(axis='both', which='minor', labelsize=12, width=0.5, length=2)
        
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
                 edgecolor='gray', framealpha=0.9, fontsize=14, handlelength=1.5)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')
        
        plt.tight_layout()
        
        out_dir = Path("figs_precision_compare")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_suffix = f"_{model_name}" if model_name else ""
        outfile = out_dir / f"precision_type3_barplot_{data_type}{model_suffix}.pdf"
        plt.savefig(outfile, format='pdf', bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        
        print(f"Saved: {outfile}")


# =============== main ===============
def main():
    # Process default model (LLaDA-8B-Base)
    print("\n" + "="*80)
    print("PROCESSING DEFAULT MODEL: LLaDA-8B-Base")
    print("="*80)
    process_model(prefix="", model_dir=None, model_name="")
    
    print("\n" + "="*80)
    print("PLOTTING PRECISION TYPE3 BARPLOT (DEFAULT MODEL)")
    print("="*80)
    plot_precision_type3_barplot(prefix="", model_dir=None, model_name="")

    # Process LLADA15 model
    print("\n" + "="*80)
    print("PROCESSING LLADA15 MODEL: LLaDA-1.5")
    print("="*80)
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    process_model(prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15")
    
    print("\n" + "="*80)
    print("PLOTTING PRECISION TYPE3 BARPLOT (LLADA15 MODEL)")
    print("="*80)
    plot_precision_type3_barplot(prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15")
    
    calculate_flip_rate()


if __name__ == "__main__":
    main()
