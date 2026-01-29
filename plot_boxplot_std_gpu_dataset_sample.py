import os, glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

base_dir = "results"
model_dirname = "GSAI-ML__LLaDA-8B-Base"
acc_datasets  = ["piqa", "winogrande", "arc_challenge"]
pass_datasets = ["humaneval", "mbpp"]
gpus          = ["H100", "A100"]
run_suffix = "run1"
# =======================================

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

def collect_gpu_values(ds: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    vals = {}
    # H100
    if ds == "arc_challenge":
        run_name_h100 = f"{ds}_batch4_{run_suffix}"
    else:
        run_name_h100 = f"{ds}_base_{run_suffix}"
    if prefix:
        run_name_h100 = f"{prefix}{run_name_h100}"
    run_dir_h100 = os.path.join(base_dir, run_name_h100, model_dir)
    if os.path.isdir(run_dir_h100):
        json_path = find_results_json(run_dir_h100)
        if json_path:
            data = load_json_any(json_path)
            if "results" in data and ds in data["results"]:
                vals["H100"] = pick_metric_value(data["results"][ds], ds)

    # A100
    if prefix == "llada15_":
        if ds == "arc_challenge":
            run_name_a100 = f"{prefix}{ds}_batch4_a100_{run_suffix}"
        else:
            run_name_a100 = f"{prefix}{ds}_a100_{run_suffix}"
    else:
        if ds == "arc_challenge":
            run_name_a100 = f"{ds}_batch4_a100_{run_suffix}"
        else:
            run_name_a100 = f"{ds}_base_a100_{run_suffix}"
        if prefix:
            run_name_a100 = f"{prefix}{run_name_a100}"
    run_dir_a100 = os.path.join(base_dir, run_name_a100, model_dir)
    if os.path.isdir(run_dir_a100):
        json_path = find_results_json(run_dir_a100)
        if json_path:
            data = load_json_any(json_path)
            if "results" in data and ds in data["results"]:
                vals["A100"] = pick_metric_value(data["results"][ds], ds)
    return vals

def print_gpu_comparison(acc_data, pass_data, out_root):
    def _print(data_dict, datasets, name):
        rows = []
        for ds in datasets:
            vals = data_dict.get(ds, {})
            if not vals:
                continue
            h100 = vals.get("H100", None)
            a100 = vals.get("A100", None)
            delta = a100 - h100 if (h100 is not None and a100 is not None) else None
            rows.append((ds, h100, a100, delta))
        if rows:
            df = pd.DataFrame(rows, columns=["dataset", "H100", "A100", "delta(A100-H100)"])
            print(f"\n[{name.upper()} GPU DATASET-WISE COMPARISON]")
            print(df.to_string(index=False,
                               formatters={
                                   "H100": "{:.4f}".format,
                                   "A100": "{:.4f}".format,
                                   "delta(A100-H100)": "{:.4f}".format
                               }))
            df.to_csv(out_root / f"gpu_datasetwise_{name}.csv", index=False)

    _print(acc_data, acc_datasets, "acc")
    _print(pass_data, pass_datasets, "pass")

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


def load_samplewise_rows(ds: str, gpu: str, prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    if gpu == "H100":
        if ds == "arc_challenge":
            run_name = f"{ds}_batch4_{run_suffix}"
        else:
            run_name = f"{ds}_base_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
    else:
        # A100
        if prefix == "llada15_":
            if ds == "arc_challenge":
                run_name = f"{prefix}{ds}_batch4_a100_{run_suffix}"
            else:
                run_name = f"{prefix}{ds}_a100_{run_suffix}"
        else:
            run_name = f"{ds}_batch4_a100_{run_suffix}" if ds == "arc_challenge" else f"{ds}_base_a100_{run_suffix}"
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
                    "GPU": gpu,
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
                    "GPU": gpu,
                })
    return pd.DataFrame(rows)


def summarize_samplewise_gpu(prefix: str = "", model_dir: str = None):
    """ """
    batch_results = []
    sample_results = []
    dataset_results = []

    for ds in acc_datasets:
        gpu_tables = {}
        for gpu in gpus:
            gpu_tables[gpu] = load_samplewise_rows(ds, gpu, prefix=prefix, model_dir=model_dir)
        if any(df.empty for df in gpu_tables.values()):
            continue

        # Sample-level
        merged_samples = None
        for gpu, df in gpu_tables.items():
            sub = df[["doc_id", "label", "pred", "correct"]].copy()
            sub = sub.rename(columns={
                "pred": f"pred_{gpu}",
                "correct": f"correct_{gpu}",
            })
            merged_samples = sub if merged_samples is None else merged_samples.merge(sub, on=["doc_id", "label"], how="outer")

        correct_cols = [c for c in merged_samples.columns if c.startswith("correct_")]
        merged_samples["sample_acc"] = merged_samples[correct_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples = merged_samples.rename(columns={"label": "truth"})
        sample_results.append(merged_samples)

        # Batch-level (per GPU)
        for gpu, df in gpu_tables.items():
            mean_acc = df["correct"].mean()
            std_acc  = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
            batch_results.append({
                "dataset": ds,
                "gpu": gpu,
                "n": len(df),
                "consistency_acc": mean_acc,
                "std": std_acc,
            })

        # Dataset-level summary
        batch_accs = [df["correct"].mean() for df in gpu_tables.values()]
        mean_acc = float(np.mean(batch_accs))
        # gpu_std: variability across GPU-level accuracies
        gpu_std = float(np.std(batch_accs, ddof=1)) if len(batch_accs) > 1 else 0.0
        # sample_std: variability across per-sample accuracies (averaged over GPUs)
        sample_std = float(merged_samples["sample_acc"].std(ddof=1)) if len(merged_samples) > 1 else 0.0
        # sample_se: standard error of mean sample_acc across doc_ids
        sample_se = sample_std / np.sqrt(len(merged_samples)) if len(merged_samples) > 0 else 0.0
        n_batches = len(batch_accs)
        n_unique = merged_samples["doc_id"].nunique()
        total_samples = sum(len(df) for df in gpu_tables.values())
        se_n = gpu_std / np.sqrt(n_batches) if n_batches > 0 else 0.0
        se_n_unique = gpu_std / np.sqrt(n_unique) if n_unique > 0 else 0.0
        dataset_results.append({
            "dataset": ds,
            "consistency_acc": mean_acc,
            "gpu_std": gpu_std,
            "sample_std": sample_std,
            "sample_se": sample_se,
            "se_n": se_n,
            "se_n_unique": se_n_unique,
            "n": n_batches,
            "n_unique": n_unique,
            "total_samples": total_samples,
        })

    for ds in pass_datasets:
        gpu_tables = {}
        for gpu in gpus:
            gpu_tables[gpu] = load_samplewise_rows(ds, gpu, prefix=prefix, model_dir=model_dir)
        if any(df.empty for df in gpu_tables.values()):
            continue

        merged_samples = None
        for gpu, df in gpu_tables.items():
            sub = df[["doc_id", "pass_flag"]].copy().rename(columns={"pass_flag": f"pass_{gpu}"})
            merged_samples = sub if merged_samples is None else merged_samples.merge(sub, on="doc_id", how="outer")

        pass_cols = [c for c in merged_samples.columns if c.startswith("pass_")]
        merged_samples["sample_acc"] = merged_samples[pass_cols].mean(axis=1, skipna=True)
        merged_samples.insert(0, "dataset", ds)
        merged_samples["truth"] = np.nan
        sample_results.append(merged_samples)

        for gpu, df in gpu_tables.items():
            mean_pass = df["pass_flag"].mean()
            std_pass  = df["pass_flag"].std(ddof=1) if len(df["pass_flag"]) > 1 else 0.0
            batch_results.append({
                "dataset": ds,
                "gpu": gpu,
                "n": len(df),
                "consistency_acc": mean_pass,
                "std": std_pass,
            })

        batch_pass_rates = [df["pass_flag"].mean() for df in gpu_tables.values()]
        mean_pass = float(np.mean(batch_pass_rates))
        # gpu_std: variability across GPU-level pass rates
        gpu_std = float(np.std(batch_pass_rates, ddof=1)) if len(batch_pass_rates) > 1 else 0.0
        # sample_std: variability across per-sample pass rates (averaged over GPUs)
        sample_std = float(merged_samples["sample_acc"].std(ddof=1)) if len(merged_samples) > 1 else 0.0
        # sample_se: standard error of mean sample_acc across doc_ids
        sample_se = sample_std / np.sqrt(len(merged_samples)) if len(merged_samples) > 0 else 0.0
        n_batches = len(batch_pass_rates)
        n_unique = merged_samples["doc_id"].nunique()
        total_samples = sum(len(df) for df in gpu_tables.values())
        se_n = gpu_std / np.sqrt(n_batches) if n_batches > 0 else 0.0
        se_n_unique = gpu_std / np.sqrt(n_unique) if n_unique > 0 else 0.0
        dataset_results.append({
            "dataset": ds,
            "consistency_acc": mean_pass,
            "gpu_std": gpu_std,
            "sample_std": sample_std,
            "sample_se": sample_se,
            "se_n": se_n,
            "se_n_unique": se_n_unique,
            "n": n_batches,
            "n_unique": n_unique,
            "total_samples": total_samples,
        })

    batch_df = pd.DataFrame(batch_results)
    sample_df = pd.concat(sample_results, ignore_index=True) if sample_results else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_results)
    return batch_df, sample_df, dataset_df



# =============== Process single model ===============
def process_model(prefix: str = "", model_dir: str = None, out_root: Path = None, model_name: str = ""):
    """ """
    if out_root is None:
        out_root = Path("figs_gpu_compare")
    if model_name:
        out_root = out_root / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    # dataset-wise
    acc_data = {ds: collect_gpu_values(ds, prefix=prefix, model_dir=model_dir) for ds in acc_datasets}
    pass_data = {ds: collect_gpu_values(ds, prefix=prefix, model_dir=model_dir) for ds in pass_datasets}
    print_gpu_comparison(acc_data, pass_data, out_root)

    # sample-wise
    batch_df, sample_df, dataset_df = summarize_samplewise_gpu(prefix=prefix, model_dir=model_dir)
    if not batch_df.empty:
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} GPU SAMPLE-WISE BATCH RESULTS]")
        print(batch_df.to_string(index=False,
                                 formatters={
                                     "consistency_acc": "{:.4f}".format,
                                     "std": "{:.4f}".format,
                                 }))
        batch_df.to_csv(out_root / "gpu_samplewise_batch.csv", index=False)
    if not sample_df.empty:
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} GPU SAMPLE-WISE PER-DOC RESULTS]")
        sample_df.to_csv(out_root / "gpu_samplewise_per_doc.csv", index=False)
    if not dataset_df.empty:
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} GPU SAMPLE-WISE DATASET SUMMARY]")
        print("(Mean and std of GPU-level accuracies)")
        print(dataset_df.to_string(index=False,
                                   formatters={
                                       "consistency_acc": "{:.4f}".format,
                                       "gpu_std": "{:.4f}".format,
                                       "sample_std": "{:.4f}".format,
                                       "sample_se": "{:.6f}".format,
                                       "se_n": "{:.6f}".format,
                                       "se_n_unique": "{:.6f}".format,
                                   }))
        dataset_df.to_csv(out_root / "gpu_samplewise_dataset_summary.csv", index=False)

    print(f"\n[OK] Output dir for {model_name or 'default'}: {out_root.resolve()}")
    
    def recompute_dataset_pass(ds: str):
        """ """
        h100_df = load_samplewise_rows(ds, "H100", prefix=prefix, model_dir=model_dir)
        a100_df = load_samplewise_rows(ds, "A100", prefix=prefix, model_dir=model_dir)
        if h100_df.empty or a100_df.empty:
            print(f"[{ds}] Missing data")
            return
    
        if ds in acc_datasets:
            h100_pass = h100_df["correct"].mean()
            a100_pass = a100_df["correct"].mean()
        else:
            h100_pass = h100_df["pass_flag"].mean()
            a100_pass = a100_df["pass_flag"].mean()
    
        print(f"[{ds}] Recomputed pass@1 → H100: {h100_pass:.4f}, A100: {a100_pass:.4f}")
    
        vals = collect_gpu_values(ds, prefix=prefix, model_dir=model_dir)
        if vals:
            print(f"[{ds}] results.json pass@1 → H100: {vals.get('H100')}, A100: {vals.get('A100')}")
        print("-"*60)

    recompute_dataset_pass("humaneval")
    recompute_dataset_pass("mbpp")
    
    def coverage(ds, gpu):
        df = load_samplewise_rows(ds, gpu, prefix=prefix, model_dir=model_dir)
        if df.empty or "doc_id" not in df.columns:
            print(f"[{ds}] {gpu}: No data found")
            return
        print(ds, gpu, "n =", len(df), "unique doc_ids =", df["doc_id"].nunique())
        print("doc_ids head:", sorted(df["doc_id"].unique())[:10])

    coverage("humaneval", "H100")
    coverage("humaneval", "A100")

# =============== main ===============
def main():
    base_out_root = Path("figs_gpu_compare")
    
    print("="*60)
    print("[Default model: LLaDA-8B-Base]")
    print("="*60)
    process_model(prefix="", model_dir=None, out_root=base_out_root, model_name="")
    
    print("\n" + "="*60)
    print("[LLaDA-1.5 model]")
    print("="*60)
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    process_model(prefix="llada15_", model_dir=llada15_model_dirname, out_root=base_out_root, model_name="llada15")
    
    



if __name__ == "__main__":
    main()
