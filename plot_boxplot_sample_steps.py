import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- USER CONFIG -----------------------
base_dir      = "results"
model_dirname = "GSAI-ML__LLaDA-8B-Base"

acc_datasets  = ["piqa", "winogrande", "arc_challenge"]
pass_datasets = ["humaneval", "mbpp"]

batches   = ["steps64", "steps128", "steps256", "steps512", "steps1024", "base"]
run_suffix = "run1"
# -----------------------------------------------------------

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def find_sample_files(dataset: str, prefix: str = "", model_dir: str = None) -> Dict[str, Path]:
    """ """
    if model_dir is None:
        model_dir = model_dirname
    out = {}
    for b in batches:
        if prefix:
            dname = f"{prefix}{dataset}_{b}_{run_suffix}"
        else:
            dname = f"{dataset}_{b}_{run_suffix}"
        d = Path(base_dir) / dname / model_dir
        if not d.exists():
            continue
        files = sorted(d.glob(f"samples_{dataset}_*.jsonl"))
        if len(files) == 0:
            continue
        out[b] = files[-1]
    print(f"[DEBUG] dataset={dataset}, prefix={prefix}, found={list(map(str, out.values()))}")
    return out

# -------------------- ACC DATASET --------------------
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
            ll_list.append(_safe_float(item[0]))
        except Exception:
            ll_list.append(None)
    if any(v is None for v in ll_list):
        return None
    return ll_list

def load_multi_choice_rows(dataset: str, batch: str, path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            target_idx = resolve_target_idx(obj)
            ll_list = extract_ll_list(obj)
            if target_idx is None or ll_list is None:
                continue
            if target_idx < 0 or target_idx >= len(ll_list):
                continue

            lls = ll_list
            pred_idx = int(np.argmax(lls))
            ll_true = lls[target_idx]
            best_other = max([lls[j] for j in range(len(lls)) if j != target_idx]) if len(lls) > 1 else ll_true
            margin = ll_true - best_other
            correct = int(pred_idx == target_idx)

            rows.append({
                "dataset": dataset,
                "batch": batch,
                "doc_id": doc_id,
                "n_choices": len(lls),
                "label": int(target_idx),
                "ll_true": ll_true,
                "ll_best_other": best_other,
                "margin": margin,
                "pred": pred_idx,
                "correct": correct,
            })
    return pd.DataFrame(rows)

# -------------------- PASS DATASET --------------------
def extract_pass_flag(obj: dict) -> Optional[int]:
    for k in ("pass@1", "pass_at_1"):
        if k in obj:
            v = obj[k]
            try:
                f = float(v)
                return int(round(f))
            except Exception:
                return None
    return None

def load_pass_rows(dataset: str, batch: str, path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            p = extract_pass_flag(obj)
            if p is None:
                continue
            rows.append({
                "dataset": dataset,
                "batch": batch,
                "doc_id": doc_id,
                "pass_flag": int(p),
            })
    return pd.DataFrame(rows)

# -------------------- Build tables --------------------
def build_tables(prefix: str = "", model_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    acc_frames, pass_frames = [], []
    for ds in acc_datasets:
        files = find_sample_files(ds, prefix=prefix, model_dir=model_dir)
        for b, p in files.items():
            df = load_multi_choice_rows(ds, b, p)
            if not df.empty:
                acc_frames.append(df)
    acc_df = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()

    for ds in pass_datasets:
        files = find_sample_files(ds, prefix=prefix, model_dir=model_dir)
        for b, p in files.items():
            df = load_pass_rows(ds, b, p)
            if not df.empty:
                pass_frames.append(df)
    pass_df = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
    return acc_df, pass_df

# -------------------- Batch-level computation (vs ground truth) --------------------
def compute_consistency_acc(acc_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for ds, sub in acc_df.groupby("dataset"):
        # Get ground truth labels (same across all batches for same doc_id)
        gt_tbl = sub[["doc_id", "label"]].drop_duplicates(subset=["doc_id"])
        for b in sub["batch"].unique():
            cur = sub[sub["batch"] == b][["doc_id", "pred"]]
            merged = gt_tbl.merge(cur, on="doc_id", how="inner")
            if merged.empty:
                continue
            # Compare prediction with ground truth label
            correct = (merged["pred"] == merged["label"]).astype(int)
            mean_acc = correct.mean()
            std_acc  = correct.std(ddof=1) if len(correct) > 1 else 0.0
            results.append({
                "dataset": ds,
                "batch": b,
                "n": len(correct),
                "consistency_acc": mean_acc,  # Actually accuracy vs ground truth
                "std": std_acc,
            })
    return pd.DataFrame(results)

def compute_consistency_pass(pass_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for ds, sub in pass_df.groupby("dataset"):
        # For pass datasets, pass_flag is the result itself (pass rate)
        for b in sub["batch"].unique():
            cur = sub[sub["batch"] == b]["pass_flag"]
            if len(cur) == 0:
                continue
            # Pass rate is the mean of pass_flag
            mean_pass = cur.mean()
            std_pass  = cur.std(ddof=1) if len(cur) > 1 else 0.0
            results.append({
                "dataset": ds,
                "batch": b,
                "n": len(cur),
                "consistency_acc": mean_pass,  # Actually pass rate
                "std": std_pass,
            })
    return pd.DataFrame(results)

# -------------------- Sample-level summary --------------------
def summarize_by_sample_acc(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample-level: Compute mean and std of sample-level accuracies for each dataset.
    For each dataset, calculate accuracy for each sample across all batches, 
    then compute mean and std across samples.
    Returns a DataFrame with one row per dataset (same format as dataset-level).
    """
    results = []
    for ds, sub in acc_df.groupby("dataset"):
        # Get ground truth labels
        gt_tbl = sub[["doc_id", "label"]].drop_duplicates(subset=["doc_id"])
        n_unique = gt_tbl["doc_id"].nunique()
        
        # Calculate accuracy for each sample (doc_id)
        sample_accs = []
        for doc_id, label in zip(gt_tbl["doc_id"], gt_tbl["label"]):
            # Get all predictions for this doc_id across all batches
            doc_data = sub[sub["doc_id"] == doc_id][["batch", "pred"]]
            if doc_data.empty:
                continue
            
            # Compare each prediction with ground truth
            correct_flags = (doc_data["pred"] == label).astype(int)
            sample_acc = correct_flags.mean() if len(correct_flags) > 0 else 0.0
            sample_accs.append(sample_acc)
        
        if not sample_accs:
            continue
        
        # Compute mean and std of sample accuracies
        mean_acc = np.mean(sample_accs)
        std = np.std(sample_accs, ddof=1) if len(sample_accs) > 1 else 0.0
        n_samples = len(sample_accs)
        
        # Total number of samples across all batches
        total_samples = len(sub)
        
        # Standard errors
        se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
        se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
        
        results.append({
            "dataset": ds,
            "consistency_acc": mean_acc,  # Mean of sample accuracies
            "std": std,  # Std of sample accuracies
            "se_n_unique": se_n_unique,
            "se_n": se_n,
            "n_samples": n_samples,  # Number of unique samples
            "n_unique": n_unique,
            "total_samples": total_samples,
        })
    
    return pd.DataFrame(results)

def summarize_by_sample_pass(pass_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample-level: Compute mean and std of sample-level pass rates for each dataset.
    For each dataset, calculate pass rate for each sample across all batches,
    then compute mean and std across samples.
    Returns a DataFrame with one row per dataset (same format as dataset-level).
    """
    results = []
    for ds, sub in pass_df.groupby("dataset"):
        # Get unique doc_ids
        unique_docs = sub["doc_id"].unique()
        n_unique = len(unique_docs)
        
        # Calculate pass rate for each sample (doc_id)
        sample_pass_rates = []
        for doc_id in unique_docs:
            # Get all pass_flags for this doc_id across all batches
            doc_data = sub[sub["doc_id"] == doc_id]["pass_flag"]
            if len(doc_data) == 0:
                continue
            
            sample_pass_rate = doc_data.mean()
            sample_pass_rates.append(sample_pass_rate)
        
        if not sample_pass_rates:
            continue
        
        # Compute mean and std of sample pass rates
        mean_pass = np.mean(sample_pass_rates)
        std = np.std(sample_pass_rates, ddof=1) if len(sample_pass_rates) > 1 else 0.0
        n_samples = len(sample_pass_rates)
        
        # Total number of samples across all batches
        total_samples = len(sub)
        
        # Standard errors
        se_n_unique = std / np.sqrt(n_unique) if n_unique > 0 else 0.0
        se_n = std / np.sqrt(total_samples) if total_samples > 0 else 0.0
        
        results.append({
            "dataset": ds,
            "consistency_acc": mean_pass,  # Mean of sample pass rates
            "std": std,  # Std of sample pass rates
            "se_n_unique": se_n_unique,
            "se_n": se_n,
            "n_samples": n_samples,  # Number of unique samples
            "n_unique": n_unique,
            "total_samples": total_samples,
        })
    
    return pd.DataFrame(results)

# -------------------- Dataset-level summary --------------------
def summarize_consistency_by_dataset_acc(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-level: Compute mean and std of batch-level accuracies.
    For each dataset, calculate accuracy for each batch, then compute mean and std across batches.
    """
    results = []
    for ds, sub in acc_df.groupby("dataset"):
        # Get ground truth labels
        gt_tbl = sub[["doc_id", "label"]].drop_duplicates(subset=["doc_id"])
        n_unique = gt_tbl["doc_id"].nunique()

        # Calculate accuracy for each batch (ensure we check all expected batches)
        batch_accs = []
        available_batches = set(sub["batch"].unique())
        for b in batches:  # Check all expected batches including base
            if b not in available_batches:
                continue  # Skip if batch data doesn't exist
            cur = sub[sub["batch"] == b][["doc_id", "pred"]]
            merged = gt_tbl.merge(cur, on="doc_id", how="inner")
            if merged.empty:
                continue
            # Compare prediction with ground truth
            correct = (merged["pred"] == merged["label"]).astype(int)
            batch_acc = correct.mean()
            batch_accs.append(batch_acc)
        
        if not batch_accs:
            continue

        # Compute mean and std of batch accuracies
        mean_acc = np.mean(batch_accs)
        std = np.std(batch_accs, ddof=1) if len(batch_accs) > 1 else 0.0
        n_batches = len(batch_accs)
        
        # Total number of samples across all batches
        total_samples = len(sub)
        
        # Standard error (based on number of batches)
        se_n = std / np.sqrt(n_batches) if n_batches > 0 else 0.0

        results.append({
            "dataset": ds,
            "consistency_acc": mean_acc,  # Mean of batch accuracies
            "std": std,  # Std of batch accuracies
            "se_n": se_n,
            "n": n_batches,  # Number of batches (should be 6)
            "n_unique": n_unique,
            "total_samples": total_samples,
        })
    return pd.DataFrame(results)

def summarize_consistency_by_dataset_pass(pass_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-level: Compute mean and std of batch-level pass rates.
    For each dataset, calculate pass rate for each batch, then compute mean and std across batches.
    """
    results = []
    for ds, sub in pass_df.groupby("dataset"):
        # Get unique doc_ids
        n_unique = sub["doc_id"].nunique()

        # Calculate pass rate for each batch (ensure we check all expected batches)
        batch_pass_rates = []
        available_batches = set(sub["batch"].unique())
        for b in batches:  # Check all expected batches including base
            if b not in available_batches:
                continue  # Skip if batch data doesn't exist
            cur = sub[sub["batch"] == b]["pass_flag"]
            if len(cur) > 0:
                batch_pass_rate = cur.mean()
                batch_pass_rates.append(batch_pass_rate)
        
        if not batch_pass_rates:
            continue

        # Compute mean and std of batch pass rates
        mean_pass = np.mean(batch_pass_rates)
        std = np.std(batch_pass_rates, ddof=1) if len(batch_pass_rates) > 1 else 0.0
        n_batches = len(batch_pass_rates)
        
        # Total number of samples across all batches
        total_samples = len(sub)
        
        # Standard error (based on number of batches)
        se_n = std / np.sqrt(n_batches) if n_batches > 0 else 0.0

        results.append({
            "dataset": ds,
            "consistency_acc": mean_pass,  # Mean of batch pass rates
            "std": std,  # Std of batch pass rates
            "se_n": se_n,
            "n": n_batches,  # Number of batches (should be 6)
            "n_unique": n_unique,
            "total_samples": total_samples,
        })
    return pd.DataFrame(results)

# -------------------- Main --------------------
# ----------------------- Process single model -----------------------
def process_model(prefix: str = "", model_dir: str = None, out_root: Path = None, model_name: str = ""):
    """ """
    if out_root is None:
        out_root = Path("figs_samplewise_steps")
    if model_name:
        out_root = out_root / model_name
    out_root.mkdir(parents=True, exist_ok=True)
    
    acc_df, pass_df = build_tables(prefix=prefix, model_dir=model_dir)
    if acc_df.empty and pass_df.empty:
        print(f"[WARNING] No data found for model: {model_name or 'default'}")
        return

    if not acc_df.empty:
        acc_df.to_csv(out_root / "samplewise_table_acc.csv", index=False)
        
        # Batch-level results
        consistency_acc_df = compute_consistency_acc(acc_df)
        consistency_acc_df.to_csv(out_root / "consistency_table_acc.csv", index=False)
        
        # Sample-level results (per dataset, same format as dataset-level)
        sample_level_acc_df = summarize_by_sample_acc(acc_df)
        sample_level_acc_df.to_csv(out_root / "sample_level_acc.csv", index=False)
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} SAMPLE-LEVEL ACC SUMMARY]")
        print("(Mean and std of sample-level accuracies)")
        print(sample_level_acc_df.to_string(index=False,
                                            formatters={
                                                "consistency_acc": "{:.4f}".format,
                                                "std": "{:.4f}".format,
                                                "se_n_unique": "{:.6f}".format,
                                                "se_n": "{:.6f}".format,
                                            }))
        
        # Dataset-level results (mean and std of batch accuracies)
        summary_acc_df = summarize_consistency_by_dataset_acc(acc_df)
        summary_acc_df.to_csv(out_root / "consistency_summary_acc.csv", index=False)
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} DATASET-LEVEL ACC SUMMARY]")
        print("(Mean and std of batch-level accuracies)")
        print(summary_acc_df.to_string(index=False,
                                       formatters={
                                           "consistency_acc": "{:.4f}".format,
                                           "std": "{:.4f}".format,
                                           "se_n": "{:.6f}".format,
                                       }))

    if not pass_df.empty:
        pass_df.to_csv(out_root / "samplewise_table_pass.csv", index=False)
        
        # Batch-level results
        consistency_pass_df = compute_consistency_pass(pass_df)
        consistency_pass_df.to_csv(out_root / "consistency_table_pass.csv", index=False)
        
        # Sample-level results (per dataset, same format as dataset-level)
        sample_level_pass_df = summarize_by_sample_pass(pass_df)
        sample_level_pass_df.to_csv(out_root / "sample_level_pass.csv", index=False)
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} SAMPLE-LEVEL PASS SUMMARY]")
        print("(Mean and std of sample-level pass rates)")
        print(sample_level_pass_df.to_string(index=False,
                                            formatters={
                                                "consistency_acc": "{:.4f}".format,
                                                "std": "{:.4f}".format,
                                                "se_n_unique": "{:.6f}".format,
                                                "se_n": "{:.6f}".format,
                                            }))
        
        # Dataset-level results (mean and std of batch pass rates)
        summary_pass_df = summarize_consistency_by_dataset_pass(pass_df)
        summary_pass_df.to_csv(out_root / "consistency_summary_pass.csv", index=False)
        print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} DATASET-LEVEL PASS SUMMARY]")
        print("(Mean and std of batch-level pass rates)")
        print(summary_pass_df.to_string(index=False,
                                        formatters={
                                            "consistency_acc": "{:.4f}".format,
                                            "std": "{:.4f}".format,
                                            "se_n": "{:.6f}".format,
                                        }))

    print(f"\n[DONE] Output dir for {model_name or 'default'}: {out_root.resolve()}")

# ----------------------- Main -----------------------
def main():
    base_out_root = Path("figs_samplewise_steps")
    
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
