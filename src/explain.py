# src/explain.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def ensure_dirs() -> dict[str, Path]:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {"reports": reports_dir, "figures": figures_dir}


def main() -> None:
    paths = ensure_dirs()

    # Load trained model artifact
    artifact = joblib.load("models/rf.joblib")
    rf = artifact["model"]
    feature_names = artifact["feature_names"]
    config = artifact["config"]

    # Load data
    df = pd.read_csv(config["data_path"])
    X = df.drop(columns=[config["target"]])

    # Align columns to training order
    X = X[feature_names]

    # SHAP (imports here to keep startup fast)
    import shap
    import matplotlib.pyplot as plt

    # Background for baseline (speed)
    background = X.sample(min(100, len(X)), random_state=config["random_state"])
    explainer = shap.TreeExplainer(rf, data=background)

    # Explain a subset (enough for plots)
    X_explain = X.sample(min(150, len(X)), random_state=config["random_state"])

    shap_values = explainer.shap_values(X_explain)

    # Get SHAP values for class 1 (DEATH_EVENT=1)
    if isinstance(shap_values, list):
        # [class0, class1]
        shap_class1 = shap_values[1]
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        # ndarray
        if getattr(shap_values, "ndim", 0) == 3:
            # (n_samples, n_features, n_classes)
            shap_class1 = shap_values[:, :, 1]
            base_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            # (n_samples, n_features)
            shap_class1 = shap_values
            base_value = explainer.expected_value

    # 1) SHAP summary (beeswarm)
    shap.summary_plot(shap_class1, X_explain, show=False)
    summary_path = paths["figures"] / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) SHAP bar (global importance)
    shap.summary_plot(shap_class1, X_explain, plot_type="bar", show=False)
    bar_path = paths["figures"] / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Waterfall for one patient
    i = 0
    x_i = X_explain.iloc[i]

    exp = shap.Explanation(
        values=shap_class1[i],          # must be (n_features,)
        base_values=base_value,
        data=x_i,                       # keep as Series for clarity
        feature_names=X_explain.columns,
    )

    shap.plots.waterfall(exp, show=False)
    wf_path = paths["figures"] / "shap_waterfall_patient0.png"
    plt.tight_layout()
    plt.savefig(wf_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Probability (for context)
    proba = rf.predict_proba(X_explain)[:, 1]
    print("Patient 0 predicted death probability:", float(proba[i]))
    print("✅ Saved:")
    print(" -", summary_path)
    print(" -", bar_path)
    print(" -", wf_path)


if __name__ == "__main__":
    main()