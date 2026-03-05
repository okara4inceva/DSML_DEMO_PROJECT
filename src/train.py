# src/train.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


CONFIG = {
    "data_path": Path("data/heart_failure_clinical_records_dataset.csv"),
    "target": "DEATH_EVENT",
    "test_size": 0.2,
    "random_state": 42,
    "rf_n_estimators": 500,
    "cv_folds": 5,
}


def ensure_dirs() -> dict[str, Path]:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    models_dir = Path("models")

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {"reports": reports_dir, "figures": figures_dir, "models": models_dir}


def eval_proba(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def main() -> None:
    paths = ensure_dirs()

    # 1) Load data
    df = pd.read_csv(CONFIG["data_path"])
    X = df.drop(columns=[CONFIG["target"]])
    y = df[CONFIG["target"]]

    # 2) Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )

    # 3) Baseline: Logistic Regression (scaled)
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=500, random_state=CONFIG["random_state"])),
        ]
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_metrics = eval_proba(y_test, lr_proba)

    # 4) Random Forest
    rf = RandomForestClassifier(
        n_estimators=CONFIG["rf_n_estimators"],
        class_weight="balanced",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = eval_proba(y_test, rf_proba)

    # 5) Cross-validation (RF) — ROC-AUC + PR-AUC
    cv = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"])
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    cv_res = cross_validate(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    cv_summary = {
        "roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_res["test_roc_auc"], ddof=1)),
        "pr_auc_mean": float(np.mean(cv_res["test_pr_auc"])),
        "pr_auc_std": float(np.std(cv_res["test_pr_auc"], ddof=1)),
    }

    # 6) Feature importance plot
    import matplotlib.pyplot as plt  # локальный импорт, чтобы train.py был быстрее при импорте

    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure()
    importance.plot(kind="bar")
    plt.title("Feature Importance (Random Forest)")
    plt.ylabel("Importance")
    plt.xlabel("Clinical Features")
    plt.tight_layout()
    fi_path = paths["figures"] / "feature_importance.png"
    plt.savefig(fi_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 7) Save model artifact
    model_path = paths["models"] / "rf.joblib"
    joblib.dump(
        {
            "model": rf,
            "feature_names": list(X.columns),
            "config": CONFIG,
        },
        model_path,
    )

    # 8) Save metrics.json
    out = {
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_features": int(X.shape[1]),
            "death_rate": float(y.mean()),
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "train_death_rate": float(y_train.mean()),
            "test_death_rate": float(y_test.mean()),
        },
        "logistic_regression_test": lr_metrics,
        "random_forest_test": rf_metrics,
        "random_forest_cv_5fold": cv_summary,
        "artifacts": {
            "model_path": str(model_path),
            "feature_importance_png": str(fi_path),
        },
    }

    metrics_path = paths["reports"] / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("✅ Saved:")
    print(" -", metrics_path)
    print(" -", fi_path)
    print(" -", model_path)


if __name__ == "__main__":
    main()