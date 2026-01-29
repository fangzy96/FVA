import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
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

# ----------------------- USER CONFIG -----------------------
base_dir      = "results"
model_dirname = "GSAI-ML__LLaDA-8B-Base"

acc_datasets  = ["piqa", "winogrande", "arc_challenge"]

batches   = ["mc8", "mc16", "mc32", "mc64", "mc256", "base"]
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
            line = line.strip()
            if not line:
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

def build_table(prefix: str = "", model_dir: str = None) -> pd.DataFrame:
    """ """
    frames = []
    for ds in acc_datasets:
        files = find_sample_files(ds, prefix=prefix, model_dir=model_dir)
        for b, p in files.items():
            df = load_multi_choice_rows(ds, b, p)
            if not df.empty:
                frames.append(df)
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def print_true_option_confidence_stats(df: pd.DataFrame):
    if df.empty:
        return
    print("[STATS] True-option confidence (ll_true) by dataset:")
    for ds, sub in df.groupby("dataset"):
        vals = sub["ll_true"].dropna().values
        if vals.size == 0:
            print(f"  - {ds}: no ll_true")
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        print(f"  - {ds}: mean={mean:.6f}, std={std:.6f}, n={vals.size}")

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

# -------------------- Plotting --------------------
def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def plot_delta_metric_vs_base(df: pd.DataFrame, outdir: Path, metric: str):
    _ensure_outdir(outdir)
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds].copy()
        if "base" not in sub["batch"].unique():
            continue
        base_tbl = sub[sub["batch"] == "base"][["doc_id", metric]].rename(columns={metric: f"{metric}_base"})
        deltas = []
        for b in [x for x in batches if x != "base"]:
            cur = sub[sub["batch"] == b][["doc_id", metric]].rename(columns={metric: f"{metric}_{b}"})
            merged = base_tbl.merge(cur, on="doc_id", how="inner")
            if merged.empty:
                continue
            merged[f"delta_{metric}_{b}"] = merged[f"{metric}_{b}"] - merged[f"{metric}_base"]
            deltas.append((b, merged[f"delta_{metric}_{b}"].values))
        if not deltas:
            continue
        fig = plt.figure(figsize=(10, 5))
        data = [vals for _, vals in deltas]
        labels = [name for name, _ in deltas]
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.axhline(0.0, linestyle="--")
        plt.ylabel(f"Delta {metric} (mc - base)")
        plt.title(f"{ds}: delta {metric} vs base")
        fig.tight_layout()
        fig.savefig(outdir / f"{ds}__delta_{metric}_vs_base_boxplot.png", dpi=200)
        plt.close(fig)

def plot_correct_vs_wrong_distributions(df: pd.DataFrame, outdir: Path, metric: str = "margin"):
    assert metric in {"ll_true", "margin"}
    _ensure_outdir(outdir)
    for ds in sorted(df["dataset"].unique()):
        sub_ds = df[df["dataset"] == ds].copy()
        if sub_ds.empty:
            continue
        fig = plt.figure(figsize=(12, 5))
        positions, data, labels = [], [], []
        pos = 1
        for b in [x for x in batches if x in sub_ds["batch"].unique()]:
            sub_b = sub_ds[sub_ds["batch"] == b]
            corr_vals = sub_b[sub_b["correct"] == 1][metric].dropna().values
            wrong_vals = sub_b[sub_b["correct"] == 0][metric].dropna().values
            if len(corr_vals) == 0 and len(wrong_vals) == 0:
                continue
            data.append(corr_vals); labels.append(f"{b}\ncorrect"); positions.append(pos); pos += 1
            data.append(wrong_vals); labels.append(f"{b}\nwrong");   positions.append(pos); pos += 2
        if not data:
            plt.close(fig); continue
        plt.boxplot(data, positions=positions, labels=labels, showfliers=False)
        plt.axhline(0.0, linestyle="--", color="red", alpha=0.6)
        plt.ylabel(metric)
        plt.title(f"{ds}: {metric} distribution for correct vs wrong (by mc)")
        fig.tight_layout()
        fig.savefig(outdir / f"{ds}__{metric}_correct_vs_wrong_boxplot.png", dpi=200)
        plt.close(fig)

# ----------------------- Process single model -----------------------
def process_model(prefix: str = "", model_dir: str = None, out_root: Path = None, model_name: str = ""):
    """ """
    if out_root is None:
        out_root = Path("figs_samplewise_mc")
    if model_name:
        out_root = out_root / model_name
    out_root.mkdir(parents=True, exist_ok=True)
    
    df = build_table(prefix=prefix, model_dir=model_dir)
    print_true_option_confidence_stats(df)
    if df.empty:
        print(f"[WARNING] No data found for model: {model_name or 'default'}")
        return

    for metric in ["ll_true", "margin"]:
        plot_delta_metric_vs_base(df, out_root / f"delta_vs_base_{metric}", metric=metric)
    for metric in ["ll_true", "margin"]:
        plot_correct_vs_wrong_distributions(df, out_root / f"correct_vs_wrong_{metric}", metric=metric)

    df.to_csv(out_root / "samplewise_table_acc.csv", index=False)

    # 3) Batch-level results
    consistency_acc_df = compute_consistency_acc(df)
    consistency_acc_df.to_csv(out_root / "consistency_table_acc.csv", index=False)
    
    # 4) Sample-level results (per dataset, same format as dataset-level)
    sample_level_acc_df = summarize_by_sample_acc(df)
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
    
    # 5) Dataset-level results (mean and std of batch accuracies)
    summary_acc_df = summarize_consistency_by_dataset_acc(df)
    summary_acc_df.to_csv(out_root / "consistency_summary_acc.csv", index=False)
    print(f"\n[{model_name.upper() if model_name else 'DEFAULT'} DATASET-LEVEL ACC SUMMARY]")
    print("(Mean and std of batch-level accuracies)")
    print(summary_acc_df.to_string(index=False,
                                   formatters={
                                       "consistency_acc": "{:.4f}".format,
                                       "std": "{:.4f}".format,
                                       "se_n": "{:.6f}".format,
                                   }))

    print(f"\n[DONE] Output dir for {model_name or 'default'}: {out_root.resolve()}")


def collect_mc_values_for_datasets(prefix: str = "", model_dir: str = None):
    """ """
    results = []
    
    for ds in acc_datasets:
        files = find_sample_files(ds, prefix=prefix, model_dir=model_dir)
        for mc_setting, file_path in files.items():
            df = load_multi_choice_rows(ds, mc_setting, file_path)
            if not df.empty:
                mean_acc = df["correct"].mean()
                std_acc = df["correct"].std(ddof=1) if len(df["correct"]) > 1 else 0.0
                n_samples = len(df)
                se_acc = std_acc / np.sqrt(n_samples) if n_samples > 0 else 0.0
                results.append({
                    "dataset": ds,
                    "mc_setting": mc_setting,
                    "accuracy": mean_acc,
                    "std": std_acc,
                    "se": se_acc,
                })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def plot_mc_barplot():
    """ """
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    results = []
    
    for model_name, prefix, model_dir in [("llada", "", None), ("llada15", "llada15_", llada15_model_dirname)]:
        df = collect_mc_values_for_datasets(prefix=prefix, model_dir=model_dir)
        if df.empty:
            continue
        
        for ds in acc_datasets:
            ds_data = df[df["dataset"] == ds]
            if not ds_data.empty:
                total_samples = ds_data.shape[0]
                mean_acc = ds_data["accuracy"].mean()
                se_acc = ds_data["se"].mean() if "se" in ds_data.columns else 0.0
                
                results.append({
                    "model": model_name,
                    "dataset": ds,
                    "accuracy": mean_acc,
                    "se": se_acc,
                })
    
    if not results:
        print("[WARNING] No MC data found for any model")
        return
    
    results_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = sorted(results_df["dataset"].unique())
    models = ["llada", "llada15"]
    model_labels = {"llada": "LLaDA", "llada15": "LLaDA-1.5"}
    
    x = np.arange(len(datasets))
    width = 0.35
    
    colors = ['#4472C4', '#ED7D31']
    
    for i, model in enumerate(models):
        values = []
        errors = []
        for ds in datasets:
            row = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)]
            if not row.empty:
                values.append(row.iloc[0]["accuracy"])
                errors.append(row.iloc[0]["se"])
            else:
                values.append(0)
                errors.append(0)
        
        offset = (i - 0.5) * width
        label = model_labels.get(model, model.upper())
        bars = ax.bar(x + offset, values, width, label=label, 
                     color=colors[i], alpha=0.7, edgecolor='white', linewidth=1.2)
        
        ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                   color='black', capsize=4, capthick=1.2, linewidth=1.2, elinewidth=1.2)
    
    ax.set_xlabel('Dataset', fontsize=16, fontweight='normal')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], fontsize=14)
    ax.set_ylim(bottom=0.08)
    ax.tick_params(axis='both', which='major', labelsize=13, width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=0.5, length=2)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    plt.tight_layout()
    
    out_dir = Path("figs_samplewise_mc")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "mc_barplot.pdf"
    plt.savefig(outfile, format='pdf', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print(f"Saved: {outfile}")


# ----------------------- Calculate Flip Rate -----------------------
def calculate_flip_rate_for_model_mc(ds: str, prefix: str, model_dir: str, model_name: str):
    """ """
    files = find_sample_files(ds, prefix=prefix, model_dir=model_dir)
    if not files:
        return None
    
    all_mc_data = {}
    for mc_setting, file_path in files.items():
        df = load_multi_choice_rows(ds, mc_setting, file_path)
        if not df.empty:
            all_mc_data[mc_setting] = df[["doc_id", "pred"]]
    
    if not all_mc_data:
        return None
    
    common_doc_ids = set(all_mc_data[list(all_mc_data.keys())[0]]["doc_id"])
    for mc_df in all_mc_data.values():
        common_doc_ids &= set(mc_df["doc_id"])
    
    if not common_doc_ids:
        return None
    
    flip_rates = []
    n_configs = len(all_mc_data)
    
    for doc_id in common_doc_ids:
        pred_list = []
        for mc_setting, mc_df in all_mc_data.items():
            row = mc_df[mc_df["doc_id"] == doc_id]
            if not row.empty:
                pred_list.append(int(row.iloc[0]["pred"]))
        
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
        "type": "accuracy",
        "n_configs": n_configs,
        "mean_flip_rate": mean_flip_rate,
        "std_flip_rate": std_flip_rate,
        "se_flip_rate": se_flip_rate,
        "n_samples": n_samples,
        "individual_flip_rates": flip_rates,
    }


def calculate_flip_rate_mc():
    """ """
    print("\n" + "="*80)
    print("Prediction Flip Rate (LLaDA and LLaDA-1.5, by MC settings)")
    print("="*80)
    
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    results = []
    individual_flip_rates_data = []
    
    for ds in acc_datasets:
        print(f"\nDataset: {ds} (accuracy)")
        
        result_llada = calculate_flip_rate_for_model_mc(
            ds, prefix="", model_dir=None, model_name="llada"
        )
        if result_llada:
            results.append(result_llada)
            print(f"  llada: n_configs={result_llada['n_configs']}, "
                  f"mean_flip_rate={result_llada['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada['n_samples']}")
            for fr in result_llada['individual_flip_rates']:
                individual_flip_rates_data.append({
                    "model": "llada",
                    "dataset": ds,
                    "type": "accuracy",
                    "flip_rate": fr
                })
        
        result_llada15 = calculate_flip_rate_for_model_mc(
            ds, prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15"
        )
        if result_llada15:
            results.append(result_llada15)
            print(f"  llada15: n_configs={result_llada15['n_configs']}, "
                  f"mean_flip_rate={result_llada15['mean_flip_rate']:.6f}, "
                  f"n_samples={result_llada15['n_samples']}")
            for fr in result_llada15['individual_flip_rates']:
                individual_flip_rates_data.append({
                    "model": "llada15",
                    "dataset": ds,
                    "type": "accuracy",
                    "flip_rate": fr
                })
    
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
        
        out_dir = Path("figs_samplewise_mc")
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df_clean = results_df.drop(columns=['individual_flip_rates'], errors='ignore')
        results_df_clean.to_csv(out_dir / "flip_rate_by_mc.csv", index=False)
        print(f"\nSaved: {out_dir / 'flip_rate_by_mc.csv'}")
        
        plot_flip_rate_barplot_mc(results_df_clean, out_dir)
    else:
        print("\n[WARNING] No results")


def plot_flip_rate_barplot_mc(results_df: pd.DataFrame, out_dir: Path):
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
                     color=colors[i], alpha=0.7, edgecolor='white', linewidth=1.2)
        
        ax.errorbar(x + offset, values, yerr=errors, fmt='none', 
                   color='black', capsize=4, capthick=1.2, linewidth=1.2, elinewidth=1.2)
    
    ax.set_xlabel('Dataset', fontsize=16, fontweight='normal')
    ax.set_ylabel('Flip Rate', fontsize=16, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], fontsize=14)
    ax.set_ylim(bottom=0.08)
    ax.tick_params(axis='both', which='major', labelsize=13, width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=0.5, length=2)
    
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

# -------------------- Main --------------------
def main():
    base_out_root = Path("figs_samplewise_mc")
    
    print("="*60)
    print("[Default model: LLaDA-8B-Base]")
    print("="*60)
    process_model(prefix="", model_dir=None, out_root=base_out_root, model_name="")
    
    print("\n" + "="*60)
    print("[LLaDA-1.5 model]")
    print("="*60)
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    process_model(prefix="llada15_", model_dir=llada15_model_dirname, out_root=base_out_root, model_name="llada15")
    
    print("\n" + "="*60)
    print("[MC barplot (two models)]")
    print("="*60)
    plot_mc_barplot()
    
    calculate_flip_rate_mc()

if __name__ == "__main__":
    main()
