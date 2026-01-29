import os, glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

base_dir = "results"
model_dirname = "GSAI-ML__LLaDA-8B-Base"
acc_datasets  = ["piqa", "winogrande", "arc_challenge"]
pass_datasets = ["humaneval", "mbpp"]
all_datasets = acc_datasets + pass_datasets
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

def collect_batch_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    batches = ["batch1", "batch4", "batch8", "batch16", "batch32", "base"]
    rows = []
    for ds in all_datasets:
        for batch in batches:
            run_name = f"{ds}_{batch}_{run_suffix}"
            if prefix:
                run_name = f"{prefix}{run_name}"
            run_dir = os.path.join(base_dir, run_name, model_dir)
            if not os.path.isdir(run_dir):
                continue
            json_path = find_results_json(run_dir)
            if not json_path:
                continue
            data = load_json_any(json_path)
            if "results" not in data or ds not in data["results"]:
                continue
            score = pick_metric_value(data["results"][ds], ds)
            if score is not None:
                rows.append({
                    "factor": "batch",
                    "setting": batch,
                    "dataset": ds,
                    "score": score
                })
    return pd.DataFrame(rows)

def collect_cfg_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    cfgs = ["cfg0", "cfg5", "cfg10", "cfg15", "cfg20", "base"]
    rows = []
    for ds in all_datasets:
        for cfg in cfgs:
            run_name = f"{ds}_{cfg}_{run_suffix}"
            if prefix:
                run_name = f"{prefix}{run_name}"
            run_dir = os.path.join(base_dir, run_name, model_dir)
            if not os.path.isdir(run_dir):
                continue
            json_path = find_results_json(run_dir)
            if not json_path:
                continue
            data = load_json_any(json_path)
            if "results" not in data or ds not in data["results"]:
                continue
            score = pick_metric_value(data["results"][ds], ds)
            if score is not None:
                rows.append({
                    "factor": "cfg",
                    "setting": cfg,
                    "dataset": ds,
                    "score": score
                })
    return pd.DataFrame(rows)

def collect_mc_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    mcs = ["mc8", "mc16", "mc32", "mc64", "mc256", "base"]
    rows = []
    for ds in all_datasets:
        if ds in pass_datasets:
            mcs_to_use = ["base"]
        else:
            mcs_to_use = mcs
        
        for mc in mcs_to_use:
            run_name = f"{ds}_{mc}_{run_suffix}"
            if prefix:
                run_name = f"{prefix}{run_name}"
            run_dir = os.path.join(base_dir, run_name, model_dir)
            if not os.path.isdir(run_dir):
                continue
            json_path = find_results_json(run_dir)
            if not json_path:
                continue
            data = load_json_any(json_path)
            if "results" not in data or ds not in data["results"]:
                continue
            score = pick_metric_value(data["results"][ds], ds)
            if score is not None:
                rows.append({
                    "factor": "mc",
                    "setting": mc,
                    "dataset": ds,
                    "score": score
                })
    return pd.DataFrame(rows)

def collect_steps_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    steps_list = ["steps64", "steps128", "steps256", "steps512", "steps1024", "base"]
    rows = []
    for ds in all_datasets:
        if ds in acc_datasets:
            steps_to_use = ["base"]
        else:
            steps_to_use = steps_list
        
        for steps in steps_to_use:
            run_name = f"{ds}_{steps}_{run_suffix}"
            if prefix:
                run_name = f"{prefix}{run_name}"
            run_dir = os.path.join(base_dir, run_name, model_dir)
            if not os.path.isdir(run_dir):
                continue
            json_path = find_results_json(run_dir)
            if not json_path:
                continue
            data = load_json_any(json_path)
            if "results" not in data or ds not in data["results"]:
                continue
            score = pick_metric_value(data["results"][ds], ds)
            if score is not None:
                rows.append({
                    "factor": "steps",
                    "setting": steps,
                    "dataset": ds,
                    "score": score
                })
    return pd.DataFrame(rows)

def collect_gpu_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    for ds in all_datasets:
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
                    score = pick_metric_value(data["results"][ds], ds)
                    if score is not None:
                        rows.append({
                            "factor": "gpu",
                            "setting": "H100",
                            "dataset": ds,
                            "score": score
                        })
        
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
                    score = pick_metric_value(data["results"][ds], ds)
                    if score is not None:
                        rows.append({
                            "factor": "gpu",
                            "setting": "A100",
                            "dataset": ds,
                            "score": score
                        })
    return pd.DataFrame(rows)

def collect_precision_data(prefix: str = "", model_dir: str = None):
    """ """
    if model_dir is None:
        model_dir = model_dirname
    rows = []
    for ds in all_datasets:
        # int8_fp32
        run_name = f"{ds}_int8_fp32_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    score = pick_metric_value(data["results"][ds], ds)
                    if score is not None:
                        rows.append({
                            "factor": "precision",
                            "setting": "int8_fp32",
                            "dataset": ds,
                            "score": score
                        })
        
        # fp16_fp32
        run_name = f"{ds}_fp16_fp32_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    score = pick_metric_value(data["results"][ds], ds)
                    if score is not None:
                        rows.append({
                            "factor": "precision",
                            "setting": "fp16_fp32",
                            "dataset": ds,
                            "score": score
                        })
        
        # bf16_fp32
        run_name = f"{ds}_bf16_fp32_{run_suffix}"
        if prefix:
            run_name = f"{prefix}{run_name}"
        run_dir = os.path.join(base_dir, run_name, model_dir)
        if os.path.isdir(run_dir):
            json_path = find_results_json(run_dir)
            if json_path:
                data = load_json_any(json_path)
                if "results" in data and ds in data["results"]:
                    score = pick_metric_value(data["results"][ds], ds)
                    if score is not None:
                        rows.append({
                            "factor": "precision",
                            "setting": "bf16_fp32",
                            "dataset": ds,
                            "score": score
                        })
    return pd.DataFrame(rows)

def calculate_pseudo_icc_factor_level(df: pd.DataFrame) -> Tuple[float, float, float]:
    """ """
    if df.empty:
        return 0.0, 0.0, 0.0
    
    factor_means = df.groupby("factor")["score"].mean()
    overall_mean = df["score"].mean()
    
    between_ss = 0.0
    for factor, factor_mean in factor_means.items():
        factor_count = len(df[df["factor"] == factor])
        between_ss += factor_count * (factor_mean - overall_mean) ** 2
    
    n_factors = len(factor_means)
    n_total = len(df)
    if n_factors > 1 and n_total > n_factors:
        between_var = between_ss / (n_factors - 1)
    else:
        between_var = 0.0
    
    within_ss = 0.0
    for factor in df["factor"].unique():
        factor_df = df[df["factor"] == factor]
        factor_mean = factor_df["score"].mean()
        
        for _, row in factor_df.iterrows():
            within_ss += (row["score"] - factor_mean) ** 2
    
    if n_total > n_factors:
        within_var = within_ss / (n_total - n_factors)
    else:
        within_var = 0.0
    
    # Pseudo-ICC = between / (between + within)
    total_var = between_var + within_var
    if total_var > 0:
        pseudo_icc = between_var / total_var
    else:
        pseudo_icc = 0.0
    
    return between_var, within_var, pseudo_icc

def calculate_pseudo_icc_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    results = []
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        if ds_df.empty:
            continue
        between_var, within_var, pseudo_icc = calculate_pseudo_icc_factor_level(ds_df)
        
        settings_info = []
        for factor in sorted(ds_df["factor"].unique()):
            factor_settings = sorted(ds_df[ds_df["factor"] == factor]["setting"].unique())
            settings_info.append(f"{factor}:[{','.join(factor_settings)}]")
        settings_str = "; ".join(settings_info)
        
        results.append({
            "dataset": ds,
            "between_factor_var": between_var,
            "within_factor_var": within_var,
            "pseudo_icc": pseudo_icc,
            "n_factors": ds_df["factor"].nunique(),
            "n_settings": len(ds_df),
            "settings": settings_str
        })
    return pd.DataFrame(results)

# =============== process_model_icc ===============
def process_model_icc(prefix: str = "", model_dir: str = None, model_name: str = ""):
    """ """
    model_label = model_name if model_name else "default"
    print("="*60)
    print(f"Scheme 1: pseudo-ICC (factor-level) - {model_label}")
    print("="*60)
    
    print("\n[Collecting data...]")
    all_data = []
    
    batch_df = collect_batch_data(prefix, model_dir)
    if not batch_df.empty:
        all_data.append(batch_df)
        print(f"  batch: {len(batch_df)} rows")
    
    cfg_df = collect_cfg_data(prefix, model_dir)
    if not cfg_df.empty:
        all_data.append(cfg_df)
        print(f"  cfg: {len(cfg_df)} rows")
    
    mc_df = collect_mc_data(prefix, model_dir)
    if not mc_df.empty:
        all_data.append(mc_df)
        print(f"  mc: {len(mc_df)} rows")
    
    steps_df = collect_steps_data(prefix, model_dir)
    if not steps_df.empty:
        all_data.append(steps_df)
        print(f"  steps: {len(steps_df)} rows")
    
    gpu_df = collect_gpu_data(prefix, model_dir)
    if not gpu_df.empty:
        all_data.append(gpu_df)
        print(f"  gpu: {len(gpu_df)} rows")
    
    precision_df = collect_precision_data(prefix, model_dir)
    if not precision_df.empty:
        all_data.append(precision_df)
        print(f"  precision: {len(precision_df)} rows")
    
    if not all_data:
        print("\n[ERROR] No data collected!")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n[Total] {len(combined_df)} rows")
    print(f"  Factors: {sorted(combined_df['factor'].unique())}")
    print(f"  Datasets: {sorted(combined_df['dataset'].unique())}")
    
    if model_name:
        out_dir = Path("figs_pseudo_icc") / model_name
    else:
        out_dir = Path("figs_pseudo_icc")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_dir / "all_factor_data.csv", index=False)
    print(f"\nSaved: {out_dir / 'all_factor_data.csv'}")
    
    print("\n" + "="*60)
    print("[Overall] All datasets combined")
    print("="*60)
    between_var, within_var, pseudo_icc = calculate_pseudo_icc_factor_level(combined_df)
    
    print(f"\nBetween-factor variance: {between_var:.6f}")
    print(f"Within-factor variance:  {within_var:.6f}")
    print(f"Pseudo-ICC (factor-level): {pseudo_icc:.6f}")
    
    if pseudo_icc > 0.7:
        interpretation = "Variance mainly from which factor"
    elif pseudo_icc < 0.3:
        interpretation = "Similar across factors, variation within settings"
    else:
        interpretation = "Both factor and setting contribute"
    print(f"\nInterpretation: {interpretation}")
    
    print("\n" + "="*60)
    print("[By dataset]")
    print("="*60)
    dataset_results = calculate_pseudo_icc_by_dataset(combined_df)
    if not dataset_results.empty:
        stats_cols = ["dataset", "between_factor_var", "within_factor_var", "pseudo_icc", "n_factors", "n_settings"]
        print("\n" + dataset_results[stats_cols].to_string(index=False,
                                             formatters={
                                                 "between_factor_var": "{:.6f}".format,
                                                 "within_factor_var": "{:.6f}".format,
                                                 "pseudo_icc": "{:.6f}".format,
                                             }))
        
        print("\n[Settings per dataset]")
        for _, row in dataset_results.iterrows():
            print(f"\n{row['dataset']}:")
            print(f"  {row['settings']}")
        
        dataset_results.to_csv(out_dir / "pseudo_icc_by_dataset.csv", index=False)
        print(f"\nSaved: {out_dir / 'pseudo_icc_by_dataset.csv'}")
    
    print("\n" + "="*60)
    print("[Factor statistics]")
    print("="*60)
    factor_stats = []
    for factor in sorted(combined_df["factor"].unique()):
        factor_df = combined_df[combined_df["factor"] == factor]
        factor_mean = factor_df["score"].mean()
        factor_std = factor_df["score"].std(ddof=1) if len(factor_df) > 1 else 0.0
        n_total = len(factor_df)
        factor_se = factor_std / np.sqrt(n_total) if n_total > 0 else 0.0
        factor_stats.append({
            "factor": factor,
            "n_settings": factor_df["setting"].nunique(),
            "n_datasets": factor_df["dataset"].nunique(),
            "n_total": n_total,
            "mean_score": factor_mean,
            "std_score": factor_std,
            "se": factor_se
        })
    
    factor_stats_df = pd.DataFrame(factor_stats)
    print("\n" + factor_stats_df.to_string(index=False,
                                          formatters={
                                              "mean_score": "{:.4f}".format,
                                              "std_score": "{:.4f}".format,
                                              "se": "{:.6f}".format,
                                          }))
    factor_stats_df.to_csv(out_dir / "factor_statistics.csv", index=False)
    print(f"\nSaved: {out_dir / 'factor_statistics.csv'}")
    
    print(f"\nDone. Output dir: {out_dir.resolve()}")


# =============== main ===============
def main():
    # Process default model (LLaDA-8B-Base)
    print("\n" + "="*80)
    print("PROCESSING DEFAULT MODEL: LLaDA-8B-Base")
    print("="*80)
    process_model_icc(prefix="", model_dir=None, model_name="")

    # Process LLADA15 model
    print("\n" + "="*80)
    print("PROCESSING LLADA15 MODEL: LLaDA-1.5")
    print("="*80)
    llada15_model_dirname = "GSAI-ML__LLaDA-1.5"
    process_model_icc(prefix="llada15_", model_dir=llada15_model_dirname, model_name="llada15")

if __name__ == "__main__":
    main()
