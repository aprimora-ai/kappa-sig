from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from kappa_llm.detection import KappaDetector, DetectorConfig
from kappa_llm.regimes import RegimeClassifier, AttentionRegime


LOGGER = logging.getLogger("kappa_run_experiment")


PNG_NAMES = [
    "compare_oh_phi.png",
    "kappa_feature_importance.png",
    "kappa_observable_distributions.png",
    "kappa_regime_distribution.png",
    "kappa_roc_comparison.png",
]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _resolve_out_dir(model_name: str, out_dir: Optional[str]) -> Path:
    if out_dir:
        return Path(out_dir)
    label = model_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / label


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV vazio: {path}")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _infer_base_dir(fact_csv: Path, hallu_csv: Path) -> Optional[Path]:
    candidates = {fact_csv.parent, hallu_csv.parent}
    for c in list(candidates):
        if c.name in {"fact", "hallu"} and c.parent not in candidates:
            candidates.add(c.parent)
    for base in candidates:
        if (base / "fact" / "katashi_state.csv").exists() and (base / "hallu" / "katashi_state.csv").exists():
            return base
    return None


def _build_observables(df: pd.DataFrame) -> List[Dict[str, float]]:
    required = {"omega", "eta", "xi", "delta", "rscore"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(missing)}")
    obs = []
    for _, row in df.iterrows():
        obs.append(
            {
                "omega": float(row["omega"]),
                "eta": float(row["eta"]),
                "xi": float(row["xi"]),
                "delta": float(row["delta"]),
                "rscore": float(row["rscore"]),
            }
        )
    return obs


def _optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _single_feature_aucs(labels: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    aucs: Dict[str, float] = {}
    for col in df.columns:
        vals = df[col].astype(float).to_numpy()
        if np.std(vals) < 1e-12:
            aucs[col] = 0.5
            continue
        try:
            auc_pos = roc_auc_score(labels, vals)
            auc_neg = roc_auc_score(labels, -vals)
            aucs[col] = float(max(auc_pos, auc_neg))
        except ValueError:
            aucs[col] = 0.5
    return aucs


def _compute_composite_scores(df: pd.DataFrame, rscore: np.ndarray) -> np.ndarray:
    composite = rscore.astype(float).copy()
    if "rscore_std" in df.columns:
        std_vals = df["rscore_std"].astype(float).to_numpy()
        std_norm = std_vals / (std_vals.max() + 1e-12)
        composite = composite + 0.3 * std_norm
    if "eta_max" in df.columns:
        eta_vals = df["eta_max"].astype(float).to_numpy()
        eta_norm = eta_vals / (eta_vals.max() + 1e-12)
        composite = composite + 0.2 * eta_norm
    return composite


def _compute_metrics(
    fact_df: pd.DataFrame,
    hallu_df: pd.DataFrame,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    detector = KappaDetector(config=DetectorConfig())
    classifier = RegimeClassifier()

    fact_obs = _build_observables(fact_df)
    hallu_obs = _build_observables(hallu_df)

    all_obs = fact_obs + hallu_obs
    labels = np.array([0] * len(fact_obs) + [1] * len(hallu_obs))

    kappa_scores = np.array([detector.compute_kappa_score(o) for o in all_obs])
    rscore_values = np.array([o["rscore"] for o in all_obs])

    all_df = pd.concat([fact_df, hallu_df], ignore_index=True)
    composite_scores = _compute_composite_scores(all_df, rscore_values)

    kappa_thresh = _optimal_threshold(labels, kappa_scores)
    rscore_thresh = _optimal_threshold(labels, rscore_values)
    composite_thresh = _optimal_threshold(labels, composite_scores)

    kappa_preds = (kappa_scores > kappa_thresh).astype(int)
    rscore_preds = (rscore_values > rscore_thresh).astype(int)
    composite_preds = (composite_scores > composite_thresh).astype(int)

    regimes = [classifier.classify(o) for o in all_obs]
    regime_labels = [r.regime.value for r in regimes]
    katashi_preds = np.array([1 if r.regime == AttentionRegime.KATASHI else 0 for r in regimes])

    feature_importance = detector.get_feature_importance(all_obs, labels.tolist())
    single_aucs = _single_feature_aucs(labels, all_df)

    metrics = {
        "n_fact": int(len(fact_obs)),
        "n_hallu": int(len(hallu_obs)),
        "kappa_auc": float(roc_auc_score(labels, kappa_scores)),
        "kappa_accuracy": float(accuracy_score(labels, kappa_preds)),
        "kappa_f1": float(f1_score(labels, kappa_preds)),
        "kappa_precision": float(precision_score(labels, kappa_preds, zero_division=0)),
        "kappa_recall": float(recall_score(labels, kappa_preds, zero_division=0)),
        "kappa_threshold": float(kappa_thresh),
        "rscore_auc": float(roc_auc_score(labels, rscore_values)),
        "rscore_accuracy": float(accuracy_score(labels, rscore_preds)),
        "rscore_f1": float(f1_score(labels, rscore_preds)),
        "rscore_threshold": float(rscore_thresh),
        "composite_auc": float(roc_auc_score(labels, composite_scores)),
        "composite_accuracy": float(accuracy_score(labels, composite_preds)),
        "composite_f1": float(f1_score(labels, composite_preds)),
        "composite_threshold": float(composite_thresh),
        "regime_accuracy": float(accuracy_score(labels, katashi_preds)),
        "regime_f1": float(f1_score(labels, katashi_preds)),
        "regime_dist_fact": _count_regimes(regime_labels[: len(fact_obs)]),
        "regime_dist_hallu": _count_regimes(regime_labels[len(fact_obs) :]),
        "feature_importance": feature_importance,
        "single_aucs": single_aucs,
    }

    arrays = {
        "labels": labels,
        "kappa_scores": kappa_scores,
        "rscore_values": rscore_values,
        "composite_scores": composite_scores,
        "regime_labels": np.array(regime_labels),
    }

    return metrics, arrays


def _count_regimes(regimes: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in regimes:
        counts[r] = counts.get(r, 0) + 1
    return counts


def _plot_observable_distributions(fact_df: pd.DataFrame, hallu_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_cols = [c for c in fact_df.columns if c in hallu_df.columns]
    if not plot_cols:
        LOGGER.warning("Sem colunas em comum para kappa_observable_distributions.png")
        return

    n_plots = min(len(plot_cols), 12)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes_list = [axes] if n_plots == 1 else list(axes.flat)

    for ax, col in zip(axes_list, plot_cols[:n_plots]):
        ax.hist(fact_df[col], bins=20, alpha=0.6, label="Factual", color="#2ecc71", density=True)
        ax.hist(hallu_df[col], bins=20, alpha=0.6, label="Hallucination", color="#e74c3c", density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)

    for ax in axes_list[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("Kappa Observable Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_roc(labels: np.ndarray, arrays: Dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(8, 8))

    fpr_r, tpr_r, _ = roc_curve(labels, arrays["rscore_values"])
    ax.plot(fpr_r, tpr_r, label=f"R-Score (AUC={auc(fpr_r, tpr_r):.3f})", linewidth=2, color="orange")

    fpr_k, tpr_k, _ = roc_curve(labels, arrays["kappa_scores"])
    ax.plot(fpr_k, tpr_k, label=f"Kappa Score (AUC={auc(fpr_k, tpr_k):.3f})", linewidth=2, color="blue")

    fpr_c, tpr_c, _ = roc_curve(labels, arrays["composite_scores"])
    ax.plot(fpr_c, tpr_c, label=f"Composite (AUC={auc(fpr_c, tpr_c):.3f})", linewidth=2, color="purple")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regime_distribution(metrics: Dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    regime_fact = metrics.get("regime_dist_fact", {})
    regime_hallu = metrics.get("regime_dist_hallu", {})

    if not regime_fact and not regime_hallu:
        LOGGER.warning("Sem distribuições de regime para plotar")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"nagare": "#2ecc71", "utsuroi": "#f1c40f", "katashi": "#e74c3c"}

    for ax, dist, title in [(ax1, regime_fact, "Factual"), (ax2, regime_hallu, "Hallucination")]:
        if dist:
            labels = list(dist.keys())
            sizes = list(dist.values())
            pie_colors = [colors.get(l, "#95a5a6") for l in labels]
            ax.pie(
                sizes,
                labels=[f"{l.upper()}\n({v})" for l, v in zip(labels, sizes)],
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
            )
        ax.set_title(f"{title} Regime Distribution")

    fig.suptitle("Attention Regime Classification", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_feature_importance(metrics: Dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    single_aucs = metrics.get("single_aucs", {})
    if not single_aucs:
        LOGGER.warning("Sem single_aucs para plotar feature importance")
        return

    sorted_items = sorted(single_aucs.items(), key=lambda x: -x[1])
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#e74c3c" if v > 0.55 else "#3498db" if v > 0.52 else "#95a5a6" for v in values]
    ax.barh(names, values, color=bar_colors)
    ax.set_xlabel("AUC")
    ax.set_title("Single-Feature AUC (best direction)")
    ax.axvline(x=0.5, color="black", linewidth=0.5, linestyle="--", label="Random")
    ax.set_xlim(0.4, max(values) + 0.05)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_compare_oh_phi(
    fact_df: pd.DataFrame,
    hallu_df: pd.DataFrame,
    out_path: Path,
    base_dir: Optional[Path],
) -> None:
    import matplotlib.pyplot as plt

    if base_dir is not None:
        fact_state_path = base_dir / "fact" / "katashi_state.csv"
        hallu_state_path = base_dir / "hallu" / "katashi_state.csv"
        if fact_state_path.exists() and hallu_state_path.exists():
            state_fact = pd.read_csv(fact_state_path)
            state_hallu = pd.read_csv(hallu_state_path)

            def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            col_oh = pick_col(state_fact, ["Oh", "oh", "ohio", "Ohio"])
            col_phi = pick_col(state_fact, ["Phi", "phi", "Phi_greek"])

            if col_oh and col_phi:
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(211)
                ax1.plot(state_fact.index, state_fact[col_oh], label="fact")
                ax1.plot(state_hallu.index, state_hallu[col_oh], label="hallu")
                ax1.set_title("Oh (Kappa regime score)")
                ax1.legend()

                ax2 = fig.add_subplot(212)
                ax2.plot(state_fact.index, state_fact[col_phi], label="fact")
                ax2.plot(state_hallu.index, state_hallu[col_phi], label="hallu")
                ax2.set_title("Phi (damage / memory)")
                ax2.legend()

                fig.tight_layout()
                fig.savefig(out_path, dpi=160)
                plt.close(fig)
                return

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(fact_df.index, fact_df["omega"], label="fact")
    ax1.plot(hallu_df.index, hallu_df["omega"], label="hallu")
    ax1.set_title("Omega (proxy for Oh)")
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.plot(fact_df.index, fact_df["rscore"], label="fact")
    ax2.plot(hallu_df.index, hallu_df["rscore"], label="hallu")
    ax2.set_title("R-Score (proxy for Phi)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(
    out_dir: Path,
    metrics: Dict[str, object],
    params: Dict[str, object],
    generated_files: List[Path],
) -> None:
    lines = ["# Experiment Summary", "", "## Metrics", ""]

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    metric_rows = [
        ("kappa_auc", metrics.get("kappa_auc")),
        ("kappa_accuracy", metrics.get("kappa_accuracy")),
        ("kappa_f1", metrics.get("kappa_f1")),
        ("rscore_auc", metrics.get("rscore_auc")),
        ("composite_auc", metrics.get("composite_auc")),
    ]

    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for name, val in metric_rows:
        if val is not None:
            lines.append(f"| {name} | {fmt(val)} |")

    lines.extend(["", "## Parameters", "", "| Key | Value |", "| --- | --- |"])\

    for key, val in params.items():
        lines.append(f"| {key} | {val} |")

    lines.extend(["", "## Outputs", ""])
    for path in generated_files:
        lines.append(f"- {path.as_posix()}")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_load_halueval_metadata() -> Optional[Dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    try:
        ds = load_dataset("pminervini/HaluEval")
        return {"dataset": "pminervini/HaluEval", "splits": list(ds.keys())}
    except Exception:
        return {"dataset": "pminervini/HaluEval", "splits": []}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Kappa experiment end-to-end from CSVs or HaluEval metadata.")
    p.add_argument("--model_name", default="", help="Model label for outputs")
    p.add_argument("--input_mode", choices=["halueval", "csv"], default="csv")
    p.add_argument("--fact_csv", type=str, default="", help="Path to factual observables CSV")
    p.add_argument("--hallu_csv", type=str, default="", help="Path to hallucination observables CSV")
    p.add_argument("--selected_heads", type=str, default="", help="Optional selected_heads.json path")
    p.add_argument("--out_dir", type=str, default="", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    np.random.seed(args.seed)

    out_dir = _resolve_out_dir(args.model_name, args.out_dir or None)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "halueval":
        meta = _maybe_load_halueval_metadata()
        if meta is None:
            LOGGER.warning("datasets não instalado; seguindo sem metadados do HaluEval")
        else:
            LOGGER.info("HaluEval disponível: %s", meta)
    else:
        meta = None

    if not args.fact_csv or not args.hallu_csv:
        raise ValueError("--fact_csv e --hallu_csv são obrigatórios")

    fact_csv = Path(args.fact_csv)
    hallu_csv = Path(args.hallu_csv)
    fact_df = _coerce_numeric(_read_csv(fact_csv))
    hallu_df = _coerce_numeric(_read_csv(hallu_csv))

    if args.selected_heads:
        selected_path = Path(args.selected_heads)
        if not selected_path.exists():
            raise FileNotFoundError(f"selected_heads.json não encontrado: {selected_path}")
    else:
        selected_path = None

    metrics, arrays = _compute_metrics(fact_df, hallu_df)
    if meta:
        metrics["dataset"] = meta.get("dataset")
        metrics["splits"] = meta.get("splits")

    metrics_path = out_dir / "kappa_detection_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote %s", metrics_path.as_posix())

    base_dir = _infer_base_dir(fact_csv, hallu_csv)

    plot_paths = {
        "compare_oh_phi.png": out_dir / "compare_oh_phi.png",
        "kappa_feature_importance.png": out_dir / "kappa_feature_importance.png",
        "kappa_observable_distributions.png": out_dir / "kappa_observable_distributions.png",
        "kappa_regime_distribution.png": out_dir / "kappa_regime_distribution.png",
        "kappa_roc_comparison.png": out_dir / "kappa_roc_comparison.png",
    }

    _plot_compare_oh_phi(fact_df, hallu_df, plot_paths["compare_oh_phi.png"], base_dir)
    _plot_observable_distributions(fact_df, hallu_df, plot_paths["kappa_observable_distributions.png"])
    _plot_roc(arrays["labels"], arrays, plot_paths["kappa_roc_comparison.png"])
    _plot_regime_distribution(metrics, plot_paths["kappa_regime_distribution.png"])
    _plot_feature_importance(metrics, plot_paths["kappa_feature_importance.png"])

    generated_files = [metrics_path] + list(plot_paths.values())

    params = {
        "model_name": args.model_name or "",
        "input_mode": args.input_mode,
        "fact_csv": fact_csv.as_posix(),
        "hallu_csv": hallu_csv.as_posix(),
        "selected_heads": selected_path.as_posix() if selected_path else "",
        "out_dir": out_dir.as_posix(),
        "seed": args.seed,
    }

    _write_summary(out_dir, metrics, params, generated_files)

    LOGGER.info("Experiment complete. Outputs in %s", out_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
