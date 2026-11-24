"""
plot_final_dual_curves.py — 使用 final_dual_model 绘制双模型曲线对比图

用法：
    python plot_final_dual_curves.py \
        --final_dir final_dual_model \
        --x_csv curve_dataset_clean.csv \
        --y_csv curve_clean.csv \
        --select middle \
        --n_samples 6

说明：
    真实曲线：curve_clean.csv 中的真实 time + C1/C2
    预测曲线：时间模型预测 Tmax 生成 np.linspace(0, Tmax_pred, 100) + 浓度模型预测 C1/C2
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_protocol import VALIDATION_SPLIT, load_split, read_dataset, split_id


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(X_raw, feature_dim, final_dir):
    if feature_dim == 24:
        return X_raw
    sys.path.insert(0, str(final_dir))
    if feature_dim == 38:
        import feature_build
        return np.asarray(feature_build.build_features(X_raw))
    if feature_dim == 50:
        import new_feature_build
        return np.asarray(new_feature_build.build_features(X_raw))
    raise ValueError(f"Unsupported feature_dim: {feature_dim}")




def model_kwargs(row, task):
    common = {
        "hidden": int(row.get("hidden", 256)) if not pd.isna(row.get("hidden", 256)) else 256,
        "dropout": float(row.get("dropout", 0.05)) if not pd.isna(row.get("dropout", 0.05)) else 0.05,
    }
    if task == "time":
        common.update({
            "num_blocks": int(row.get("num_blocks", 5)) if not pd.isna(row.get("num_blocks", 5)) else 5,
            "num_layers": int(row.get("num_layers", 6)) if not pd.isna(row.get("num_layers", 6)) else 6,
            "lstm_layers": int(row.get("lstm_layers", 3)) if not pd.isna(row.get("lstm_layers", 3)) else 3,
            "channels": int(row.get("channels", 128)) if not pd.isna(row.get("channels", 128)) else 128,
            "kernel_size": int(row.get("kernel_size", 3)) if not pd.isna(row.get("kernel_size", 3)) else 3,
            "num_levels": int(row.get("num_levels", 5)) if not pd.isna(row.get("num_levels", 5)) else 5,
            "nhead": int(row.get("nhead", 8)) if not pd.isna(row.get("nhead", 8)) else 8,
            "dim_feedforward": None if pd.isna(row.get("dim_feedforward", np.nan)) else int(row.get("dim_feedforward")),
        })
    else:
        common.update({
            "out_dim": 200,
            "num_blocks": int(row.get("num_blocks", 5)) if not pd.isna(row.get("num_blocks", 5)) else 5,
            "num_layers": int(row.get("num_layers", 6)) if not pd.isna(row.get("num_layers", 6)) else 6,
            "lstm_layers": int(row.get("lstm_layers", 3)) if not pd.isna(row.get("lstm_layers", 3)) else 3,
            "channels": int(row.get("channels", 128)) if not pd.isna(row.get("channels", 128)) else 128,
            "kernel_size": int(row.get("kernel_size", 3)) if not pd.isna(row.get("kernel_size", 3)) else 3,
            "num_levels": int(row.get("num_levels", 5)) if not pd.isna(row.get("num_levels", 5)) else 5,
            "nhead": int(row.get("nhead", 8)) if not pd.isna(row.get("nhead", 8)) else 8,
            "dim_feedforward": None if pd.isna(row.get("dim_feedforward", np.nan)) else int(row.get("dim_feedforward")),
        })
    return common


def time_inverse(y_scaled, scaler_y):
    y_log = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return np.clip(np.expm1(y_log), 0.0, None)


def predict_dual(final_dir, X_raw, selected, device):
    time_row = selected["time"]
    conc_row = selected["conc"]

    sys.path.insert(0, str(final_dir))
    from models import build_time_model, build_conc_model, default_time_groups, default_conc_groups

    time_feat = int(time_row["features"])
    conc_feat = int(conc_row["features"])
    X_time = build_features(X_raw, time_feat, final_dir)
    X_conc = build_features(X_raw, conc_feat, final_dir)

    time_scaler_X = joblib.load(final_dir / "time_scaler_X.pkl")
    time_scaler_Y = joblib.load(final_dir / "time_scaler_Y.pkl")
    conc_scaler_X = joblib.load(final_dir / "conc_scaler_X.pkl")
    conc_scaler_Y = joblib.load(final_dir / "conc_scaler_Y.pkl")

    X_time_s = time_scaler_X.transform(X_time)
    X_conc_s = conc_scaler_X.transform(X_conc)

    time_pt = final_dir / "time_model_best.pt"
    if time_row["model"] in {"rf", "xgb", "lgb"}:
        time_model = joblib.load(final_dir / "time_model_best.pkl")
        time_pred_scaled = time_model.predict(X_time_s)
    else:
        time_model = build_time_model(
            time_row["model"],
            in_dim=time_feat,
            groups=default_time_groups(time_feat),
            **model_kwargs(time_row, "time"),
        ).to(device)
        time_model.load_state_dict(torch.load(time_pt, map_location=device))
        time_model.eval()
        with torch.no_grad():
            time_pred_scaled = time_model(torch.FloatTensor(X_time_s).to(device)).cpu().numpy()

    conc_model = build_conc_model(
        conc_row["model"],
        in_dim=conc_feat,
        groups=default_conc_groups(conc_feat),
        **model_kwargs(conc_row, "conc"),
    ).to(device)
    conc_model.load_state_dict(torch.load(final_dir / "conc_model_best.pt", map_location=device))
    conc_model.eval()

    with torch.no_grad():
        conc_pred_scaled = conc_model(torch.FloatTensor(X_conc_s).to(device)).cpu().numpy()

    Tmax_pred = time_inverse(time_pred_scaled, time_scaler_Y)
    conc_pred = conc_scaler_Y.inverse_transform(conc_pred_scaled)
    conc_pred = np.clip(conc_pred, 0.0, None)

    return Tmax_pred, conc_pred


def select_samples(indices, sample_scores, mode, n_samples, seed):
    indices = list(indices)
    if len(indices) == 0:
        raise RuntimeError("No candidate indices.")
    n_samples = min(n_samples, len(indices))
    rng = random.Random(seed)

    if mode == "random":
        return rng.sample(indices, n_samples)

    sorted_idx = sorted(indices, key=lambda i: sample_scores.get(i, -999))
    if mode == "best":
        return sorted_idx[-n_samples:][::-1]
    if mode == "worst":
        return sorted_idx[:n_samples]
    if mode == "middle":
        mid = len(sorted_idx) // 2
        half = n_samples // 2
        return sorted_idx[max(0, mid - half): max(0, mid - half) + n_samples]
    raise ValueError(f"Unknown mode: {mode}")


def plot_samples(sample_ids, test_rows, Y_all, Tmax_pred, conc_pred, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_ids)
    fig, axes = plt.subplots(n, 2, figsize=(10, max(3.0 * n, 4.0)), squeeze=False)

    for row, idx in enumerate(sample_ids):
        t_true = Y_all[idx, 0:100]
        c1_true = Y_all[idx, 100:200]
        c2_true = Y_all[idx, 200:300]
        t_pred = np.linspace(0, Tmax_pred[idx], 100)
        c1_pred = conc_pred[idx, :100]
        c2_pred = conc_pred[idx, 100:200]

        ax = axes[row, 0]
        ax.plot(t_true, c1_true, lw=1.8, label="True")
        ax.plot(t_pred, c1_pred, lw=1.8, ls="--", label="Pred")
        ax.set_title(f"Sample {test_rows[idx]} — C1", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time", fontweight="bold")
        ax.set_ylabel("Concentration", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(prop={"weight": "bold", "size": 8})

        ax = axes[row, 1]
        ax.plot(t_true, c2_true, lw=1.8, label="True")
        ax.plot(t_pred, c2_pred, lw=1.8, ls="--", label="Pred")
        ax.set_title(f"Sample {test_rows[idx]} — C2", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time", fontweight="bold")
        ax.set_ylabel("Concentration", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(prop={"weight": "bold", "size": 8})

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{prefix}.png")
    out_pdf = os.path.join(out_dir, f"{prefix}.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_dir", type=str, default="final_dual_model")
    parser.add_argument("--x_csv", type=str, default="curve_dataset_clean.csv")
    parser.add_argument("--y_csv", type=str, default="curve_clean.csv")
    parser.add_argument("--out_dir", type=str, default="final_curve_figures")
    parser.add_argument("--select", type=str, default="middle", choices=["best", "middle", "worst", "random"])
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_only", action="store_true")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    X_raw, Y_all = read_dataset(args.x_csv, args.y_csv)
    manifest = load_split(final_dir / "split_manifest.json", X_raw, Y_all)
    expected_split_id = split_id(manifest)
    selected = load_json(final_dir / "selected_models.json")
    for role in ("time", "conc"):
        row = selected[role]
        if row.get("split_id") != expected_split_id or row.get("evaluation_split") != VALIDATION_SPLIT:
            raise ValueError(f"The selected {role} checkpoint must be chosen on the shared validation split.")
    if selected["time"].get("target") != "total":
        raise ValueError("Dual-curve reconstruction requires the total simulation duration model.")

    test_rows = np.asarray(manifest["indices"]["test"])
    y_test = Y_all[test_rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tmax_pred, conc_pred = predict_dual(final_dir, X_raw[test_rows], selected, device)
    y_true = y_test[:, 100:300]
    t_true = y_test[:, 99]

    report = {
        "evaluation_split": "test",
        "split_id": expected_split_id,
        "split_mode": manifest["mode"],
        "n_test": len(test_rows),
        "time": {
            "R2": float(r2_score(t_true, Tmax_pred)),
            "MAE": float(mean_absolute_error(t_true, Tmax_pred)),
            "RMSE": float(mean_squared_error(t_true, Tmax_pred) ** 0.5),
        },
        "concentration": {
            "R2": float(r2_score(y_true, conc_pred, multioutput="variance_weighted")),
            "R2_C1": float(r2_score(y_true[:, :100], conc_pred[:, :100], multioutput="variance_weighted")),
            "R2_C2": float(r2_score(y_true[:, 100:], conc_pred[:, 100:], multioutput="variance_weighted")),
            "MAE": float(mean_absolute_error(y_true, conc_pred)),
            "RMSE": float(mean_squared_error(y_true, conc_pred) ** 0.5),
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / "test_metrics.json", "w", encoding="utf-8") as result:
        json.dump(report, result, indent=2)
    print(f"Saved independent test metrics for {len(test_rows)} cases.")
    if args.metrics_only:
        return

    scores = {i: r2_score(y_true[i], conc_pred[i]) for i in range(len(y_true))}
    sample_ids = select_samples(range(len(test_rows)), scores, args.select, args.n_samples, args.seed)
    print("Selected test rows:", test_rows[sample_ids].tolist())
    pd.DataFrame({"sample_index": test_rows[sample_ids], "sample_R2": [scores[i] for i in sample_ids]}).to_csv(
        Path(args.out_dir) / f"selected_samples_{args.select}.csv", index=False,
    )
    plot_samples(sample_ids, test_rows, y_test, Tmax_pred, conc_pred, args.out_dir,
                 f"dual_curve_comparison_{args.select}_{len(sample_ids)}samples")


if __name__ == "__main__":
    main()
