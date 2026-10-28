#!/usr/bin/env python3
"""
MediPredict - Data drift analysis with MLflow
=============================================
Use case : Diabetes risk screening for a small clinic (tabular ML, binary classification)
Dataset  : Pima Indians Diabetes Database (768 records, 8 features + Outcome)

What this script does
  1. Trains a baseline RandomForest model and logs it to MLflow.
  2. Treats the training split (50 %) as the REFERENCE data; the other 50 % is an unseen pool.
  3. Builds four SIMULATED "current" data windows (the dataset has no real time axis):
       - control_no_shift          : random sample of unseen data (should show NO drift)
       - population_shift_moderate : clinic starts seeing older / heavier / higher-glucose patients
       - population_shift_severe   : same, but much stronger
       - glucose_unit_change       : lab switches Glucose from mg/dL to mmol/L (data-quality drift)
  4. For every window, detects drift with the KS test + Population Stability Index (PSI),
     checks prediction drift and model performance, and logs everything to MLflow.

Run:  python drift_pipeline.py            (downloads the dataset if data/diabetes.csv is missing)
      python drift_pipeline.py --data path/to/diabetes.csv
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

# ----------------------------------------------------------------------------- configuration
EXPERIMENT_NAME = "MediPredict-Diabetes-Drift"
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNS[:-1]
TARGET = "Outcome"
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DATA_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]

# Drift decision rules (documented in the report)
ALPHA = 0.05                              # family-wise significance level for KS tests
KS_ALPHA = ALPHA / len(FEATURES)          # Bonferroni correction over 8 features
PSI_MIN_EFFECT = 0.10                     # a feature is flagged only if KS is significant AND PSI >= 0.1
PSI_SIGNIFICANT = 0.20                    # PSI band labels: <0.1 stable, 0.1-0.2 moderate, >=0.2 significant
DATASET_DRIFT_SHARE = 0.25                # dataset-level drift if >= 25 % of features are flagged

SCENARIOS = {
    "control_no_shift":            {"kind": "random"},
    "population_shift_moderate":   {"kind": "tilt", "k": 1.5},
    "population_shift_severe":     {"kind": "tilt", "k": 3.0},
    "glucose_unit_change":         {"kind": "unit_change"},
}


# ----------------------------------------------------------------------------- data loading
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for url in DATA_URLS:
            try:
                print(f"Downloading dataset from {url} ...")
                urllib.request.urlretrieve(url, path)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  download failed: {exc}")
        if not path.exists():
            sys.exit(
                "Could not download the dataset.\n"
                "Download the Pima Indians Diabetes CSV manually (Kaggle: 'Pima Indians Diabetes "
                f"Database' or UCI) and save it as {path}, or pass --data <file>."
            )
    df = pd.read_csv(path)
    if not set(COLUMNS).issubset(df.columns):          # file without header row
        df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df.shape != (768, 9):
        print(f"WARNING: expected 768 x 9, got {df.shape}")
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)   # zeros are missing values
    return df


# ----------------------------------------------------------------------------- drift metrics
def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using reference-quantile bins."""
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        ks = ks_2samp(ref[f].dropna(), cur[f].dropna())
        p = psi(ref[f], cur[f])
        rows.append({
            "feature": f,
            "ref_mean": ref[f].mean(), "cur_mean": cur[f].mean(),
            "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue, "psi": p,
            "missing_ref": ref[f].isna().mean(), "missing_cur": cur[f].isna().mean(),
            "psi_level": "stable" if p < 0.1 else ("moderate" if p < PSI_SIGNIFICANT else "significant"),
            "drift": bool(ks.pvalue < KS_ALPHA and p >= PSI_MIN_EFFECT),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- simulated windows
def make_current_window(pool: pd.DataFrame, ref: pd.DataFrame, spec: dict, n: int, seed: int):
    rng = np.random.default_rng(seed)
    kind = spec["kind"]
    if kind == "random":
        return pool.sample(n=min(n, len(pool)), replace=False, random_state=seed)
    if kind == "tilt":
        # Select patients with probability proportional to exp(k * z), where z is the mean
        # standardised Age/BMI/Glucose. Labels stay genuine because real rows are re-sampled.
        z = sum((pool[c].fillna(ref[c].median()) - ref[c].mean()) / ref[c].std()
                for c in ["Age", "BMI", "Glucose"]) / 3
        w = np.exp(spec["k"] * z)
        idx = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False,
                         p=(w / w.sum()).to_numpy())
        return pool.loc[idx]
    if kind == "unit_change":
        cur = pool.sample(n=min(n, len(pool)), replace=False, random_state=seed).copy()
        cur["Glucose"] = cur["Glucose"] / 18.0       # mg/dL -> mmol/L
        return cur
    raise ValueError(kind)


# ----------------------------------------------------------------------------- plots
def plot_distributions(ref, cur, title, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.ravel(), FEATURES):
        ax.hist(ref[f].dropna(), bins=15, density=True, alpha=0.55, label="reference")
        ax.hist(cur[f].dropna(), bins=15, density=True, alpha=0.55, label="current")
        ax.set_title(f, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_psi(table, title, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c0392b" if d else "#2e86c1" for d in table["drift"]]
    ax.barh(table["feature"], table["psi"], color=colors)
    ax.axvline(PSI_MIN_EFFECT, ls=":", c="gray")
    ax.axvline(PSI_SIGNIFICANT, ls="--", c="black")
    ax.set_xlabel("PSI (dotted = 0.1, dashed = 0.2; red = drift flagged)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetes.csv")
    ap.add_argument("--n-current", type=int, default=150, help="records in each current window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    out = Path("results")
    out.mkdir(exist_ok=True)

    df = load_data(Path(args.data))
    ref_df, pool_df = train_test_split(df, test_size=0.50, stratify=df[TARGET], random_state=args.seed)
    X_ref, y_ref = ref_df[FEATURES], ref_df[TARGET]

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3,
                                       class_weight="balanced", random_state=args.seed)),
    ])

    # reference predictions must be out-of-fold, otherwise a random forest looks over-confident
    ref_proba = cross_val_predict(model, X_ref, y_ref, method="predict_proba",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=args.seed))[:, 1]
    oof_auc = roc_auc_score(y_ref, ref_proba)
    model.fit(X_ref, y_ref)

    def perf(frame):
        proba = model.predict_proba(frame[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y = frame[TARGET]
        return proba, {"auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
                       "recall": recall_score(y, pred), "f1": f1_score(y, pred)}

    # ---- run 1: baseline model
    with mlflow.start_run(run_name="baseline_training"):
        mlflow.set_tag("stage", "baseline")
        mlflow.log_params({"model": "RandomForest", "n_estimators": 200, "max_depth": 5,
                           "min_samples_leaf": 3, "class_weight": "balanced",
                           "imputer": "median", "n_reference": len(ref_df),
                           "n_holdout": len(pool_df), "seed": args.seed})
        _, base = perf(pool_df)
        mlflow.log_metrics({"oof_auc_reference": oof_auc,
                            **{f"holdout_{k}": v for k, v in base.items()}})
        ref_df[FEATURES].describe().T.to_csv(out / "reference_summary.csv")
        mlflow.log_artifact(str(out / "reference_summary.csv"))
        # The model was created by this script, so its tree types are trusted for skops loading.
        trusted = ["numpy.dtype", "sklearn.tree._tree.Tree"]
        try:
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)   # MLflow >= 3
        except TypeError:
            mlflow.sklearn.log_model(model, artifact_path="model")                       # MLflow 2.x
    print(f"Baseline: OOF AUC={oof_auc:.3f}  hold-out AUC={base['auc']:.3f}  recall={base['recall']:.3f}")

    # ---- runs 2..5: drift scenarios
    summary = []
    for i, (name, spec) in enumerate(SCENARIOS.items()):
        cur = make_current_window(pool_df, ref_df, spec, args.n_current, args.seed + i)
        table = feature_drift_table(ref_df, cur)
        proba, p = perf(cur)
        pred_ks = ks_2samp(ref_proba, proba)
        n_drift = int(table["drift"].sum())
        share = n_drift / len(FEATURES)
        dataset_drift = share >= DATASET_DRIFT_SHARE
        alert = "DATASET_DRIFT" if dataset_drift else ("FEATURE_ALERT" if n_drift else "NONE")

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("stage", "drift_analysis")
            mlflow.set_tag("drift_type", "data_drift (simulated)")
            mlflow.set_tag("alert_level", alert)
            mlflow.log_params({"scenario": name, **{f"spec_{k}": v for k, v in spec.items()},
                               "n_current": len(cur), "ks_alpha_bonferroni": round(KS_ALPHA, 5),
                               "psi_min_effect": PSI_MIN_EFFECT, "dataset_drift_share": DATASET_DRIFT_SHARE})
            metrics = {"dataset_drift_detected": int(dataset_drift),
                       "alert_code": {"NONE": 0, "FEATURE_ALERT": 1, "DATASET_DRIFT": 2}[alert], "n_drifted_features": n_drift,
                       "share_drifted_features": share,
                       "pred_mean_reference": float(ref_proba.mean()), "pred_mean_current": float(proba.mean()),
                       "pred_ks_statistic": pred_ks.statistic, "pred_ks_pvalue": pred_ks.pvalue,
                       **{f"current_{k}": v for k, v in p.items()}}
            for _, r in table.iterrows():
                metrics[f"psi_{r.feature}"] = r.psi
                metrics[f"ks_pvalue_{r.feature}"] = r.ks_pvalue
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            table.to_csv(out / f"drift_table_{name}.csv", index=False)
            plot_distributions(ref_df, cur, f"{name}: feature distributions", out / f"distributions_{name}.png")
            plot_psi(table, f"{name}: PSI per feature", out / f"psi_{name}.png")
            for fn in (f"drift_table_{name}.csv", f"distributions_{name}.png", f"psi_{name}.png"):
                mlflow.log_artifact(str(out / fn))

        summary.append({"scenario": name, "n_current": len(cur), "drifted_features": n_drift,
                        "share": share, "alert_level": alert,
                        "drifted_list": ", ".join(table.loc[table.drift, "feature"]),
                        "max_psi": table["psi"].max(), "pred_mean_ref": ref_proba.mean(),
                        "pred_mean_cur": proba.mean(), "pred_ks_p": pred_ks.pvalue, **p})

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "drift_summary.csv", index=False)
    with mlflow.start_run(run_name="summary_all_scenarios"):
        mlflow.set_tag("stage", "summary")
        mlflow.log_artifact(str(out / "drift_summary.csv"))
    with open(out / "summary.md", "w") as fh:
        fh.write(summ.round(3).to_markdown(index=False) if hasattr(summ, "to_markdown") and _has_tabulate()
                 else summ.round(3).to_string(index=False))
    print("\n" + summ.round(3).to_string(index=False))
    print(f"\nAll results saved in ./{out}/ . Start the MLflow UI with:\n"
          f"  mlflow ui --backend-store-uri {args.tracking_uri} --port 5000")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()