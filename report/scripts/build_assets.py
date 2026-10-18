"""Render report assets from frozen summaries. no model fitting or predictions.

Run from report/. Inputs are read only. all writes are confined to generated/.
CSV parsing preserves round-trip float64 values. The manifest pins every input.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

REPORT = Path(__file__).resolve().parents[1]
A4 = REPORT.parent
OUT = REPORT / "generated"
INPUTS = {}
# Okabe-Ito colours. markers and line styles provide independent distinctions.
REPRESENTATION_COLOURS = {"standard": "#0072B2", "log_standard": "#D55E00"}
PROCEDURE_COLOURS = {
    "k40": "#0072B2",
    "k150": "#D55E00",
    "k400": "#CC79A7",
    "knn": "#009E73",
    "tree": "#333333",
}
COLUMN_WIDTH = 3.486
TEXT_WIDTH = 7.16
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def source(relative):
    path = A4 / relative
    INPUTS[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def csv(relative):
    return pd.read_csv(source(relative), float_precision="round_trip")


def saved(figure, name):
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"{name}.{suffix}")
    plt.close(figure)


def tex(name, rows):
    rows = [row + chr(92) if row.endswith(chr(92)) else row for row in rows]
    if name.endswith("_table"):
        rows.append(r"\bottomrule")
    (OUT / f"{name}.tex").write_text("\n".join(rows) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(source("results/audit/dataset_summary.json").read_text())
    partitions = json.loads(source("results/splits/partition_manifest.json").read_text())
    classes = csv("results/audit/class_distribution.csv")
    som = csv("results/som-v3/sweep_summary.csv")
    dev = csv("results/exploratory/chunk4_5b/combined_validation_summary.csv")
    original = csv("results/clustering/validation_summary.csv")
    cluster = csv("results/exploratory/chunk4_5b/cluster_diagnostics.csv")
    final = csv("results/exploratory/chunk4_6/nested_summary.csv")
    pc = csv("results/exploratory/chunk4_6/per_class_summary.csv")
    bootstrap = csv("results/exploratory/chunk4_6/bootstrap_intervals.csv")
    geometry = csv("results/assessment/som_diagnostics.csv")
    fragments = csv("results/exploratory/chunk4_6/fragmentation_summary.csv")
    structural = csv("results/exploratory/chunk4_6/structural_metrics.csv")
    transfer = csv("results/exploratory/chunk4_6/development_assessment_transfer.csv")
    runtime = csv("results/assessment/runtimes.csv")
    late = csv("results/assessment/late_convergence.csv")
    dev_classes = csv("results/exploratory/chunk4_5b/combined_per_class_summary.csv")
    exploratory_execution = json.loads(
        source("results/exploratory/chunk4_6/execution.json").read_text()
    )
    exploratory_analysis = json.loads(
        source("results/exploratory/chunk4_6/analysis.json").read_text()
    )
    baseline = json.loads(source("reference/selected_configurations.json").read_text())
    assert audit["rows"] == 257673 and audit["descriptive_features"] == 43
    assert len(som) == 8 and len(dev) == 50 and len(cluster) == 250
    assert set(dev.k) == {40, 50, 60, 80, 100, 150, 200, 250, 300, 400}
    assert set(dev.alpha) == {0, 0.25, 0.5, 0.75, 1}
    anchor = original.query(
        "side == 32 and k == 40 and weighting == 'hit_weighted' and mapping == 'class_balanced'"
    ).iloc[0]
    assert dev.query("k == 40 and alpha == 1").iloc[0].macro_f1_mean == anchor.macro_f1_mean
    assert dev.query("k == 150 and alpha == 0.5").iloc[0].macro_f1_mean == 0.41962832388006294
    assert dev.query("k == 400 and alpha == 0.5").iloc[0].macro_f1_mean == 0.44753629551298085

    rows = []
    for row in classes.itertuples():
        counts = [
            partitions["partitions"][p]["class_counts"][str(row[1])]
            for p in ("train", "validation", "assessment")
        ]
        rows.append(
            f"{row.attack_category} & {row.count:,} & {counts[0]:,} "
            f"& {counts[1]:,} & {counts[2]:,} \\"
        )
    rows += ["\\midrule", "Total & 257,673 & 41,229 & 10,306 & 206,138 \\"]
    tex("class_table", rows)
    rows = []
    for rep in ("standard", "log_standard"):
        for row in som[som.representation == rep].sort_values("side").itertuples():
            label = "Standard" if rep == "standard" else "Log-standard"
            rows.append(
                f"{label} & ${row.side}\\times{row.side}$ "
                f"& {row.validation_contextual_qe_mean:.4f} "
                f"$\\pm$ {row.validation_contextual_qe_std:.4f} "
                f"& {row.validation_te_mean:.3f} "
                f"& {100 * row.train_occupied_fraction_mean:.1f} \\"
            )
    tex("som_table", rows)

    methods = ["k40", "k150", "k400", "knn", "tree"]
    labels = ["SOM K40", "SOM K150", "SOM K400", "kNN", "Tree"]
    descriptions = [
        "Confirmatory",
        "Post-hoc",
        "Post-hoc",
        "Frozen baseline",
        "Frozen baseline",
    ]
    claims = []
    claim_names = ["Confirm", "ExploreA", "ExploreB", "Neighbour", "Tree"]
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    rows = []
    for method, label, status, cname in zip(
        methods, labels, descriptions, claim_names, strict=True
    ):
        sub = final[final.method == method].set_index("metric")
        values = [float(sub.loc[m, "mean"]) for m in metrics]
        for metric, suffix in zip(metrics, ["F", "B", "A", "W"], strict=True):
            claims.append(f"\\newcommand{{\\{cname}{suffix}}}{{{sub.loc[metric, 'mean']:.3f}}}")
        sd = sub.loc["macro_f1", "som_sd"]
        f1 = f"{values[0]:.4f}" + (
            f" $\\pm$ {sd:.4f}" if method.startswith("k") and method != "knn" else ""
        )
        rows.append(
            f"{label} & {status} & {f1} & {values[1]:.4f} "
            f"& {values[2]:.4f} & {values[3]:.4f} \\"
        )
    tex("overall_table", rows)
    tex("claims", claims)

    rows = []
    for c in [2, 5, 7, 8]:
        for method, label in zip(methods[:3], ["40 / 1", "150 / .5", "400 / .5"], strict=True):
            d = pc[(pc.method == method) & (pc.class_id == c)].set_index("metric")
            vals = [
                float(d.loc[m, "mean"]) for m in ("precision", "recall", "f1", "tp", "fp", "fn")
            ]
            name = d.iloc[0].class_name
            support = int(d.iloc[0].support)
            rows.append(
                f"{name} ({support:,}) & {label} & {vals[0]:.3f} "
                f"& {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.1f} "
                f"& {vals[4]:,.1f} & {vals[5]:.1f} \\"
            )
        rows.append("\\addlinespace[2pt]")
    tex("rare_table", rows)

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.3), layout="constrained")
    for rep, style, marker, colour, label in [
        ("standard", "--", "s", REPRESENTATION_COLOURS["standard"], "Standard"),
        ("log_standard", "-", "o", REPRESENTATION_COLOURS["log_standard"], "Log-standard"),
    ]:
        d = som[som.representation == rep].sort_values("units")
        for ax, key, ylab, factor in zip(
            axes,
            ["validation_contextual_qe", "validation_te", "train_occupied_fraction"],
            [
                "(a) Validation contextual error",
                "(b) Validation topographic error",
                r"(c) Training occupancy (\%)",
            ],
            [1, 1, 100],
            strict=True,
        ):
            ax.errorbar(
                d.units,
                d[key + "_mean"] * factor,
                yerr=d[key + "_std"] * factor,
                color=colour,
                ls=style,
                marker=marker,
                markersize=4,
                capsize=2,
                label=label,
            )
            ax.set(xlabel="Map units", ylabel=ylab, xticks=[64, 256, 576, 1024])
            ax.tick_params(axis="x", labelrotation=35)
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    saved(fig, "representation")

    model_path = "results/som-v3/runs/v3-log_standard-32x32-s47425/"
    with np.load(source(model_path + "codebook.npz"), allow_pickle=False) as z:
        w = z["codebook"].copy()
        coords = z["coordinates"].copy()
        assert int(z["completed_epochs"]) == 70 and int(z["seed"]) == 47425
    with np.load(source(model_path + "assignments.npz"), allow_pickle=False) as z:
        hits = z["train_hits"].copy()
    metadata = json.loads(source("results/preprocessing/log_standard.json").read_text())
    neighbours = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2) == 1
    # Distances of neighbouring codebook pairs only. this is not a new diagnostic run.
    neighbour_mean = np.zeros(len(w))
    for j in range(len(w)):
        neighbour_mean[j] = np.linalg.norm(w[neighbours[j]] - w[j], axis=1).mean()
    feature_ids = [metadata["feature_names"].index("numeric__" + f) for f in ("dur", "sbytes")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.7), layout="constrained")
    panels = [neighbour_mean, np.log1p(hits), w[:, feature_ids[0]], w[:, feature_ids[1]]]
    names = [
        "(a) Mean neighbour distance",
        "(b) log(1 + training hits)",
        "(c) Duration: transformed coordinate",
        "(d) Source bytes: transformed coordinate",
    ]
    for ax, data, label in zip(axes.flat, panels, names, strict=True):
        assert np.isfinite(data).all()
        if label.startswith(("(c)", "(d)")):
            limit = float(np.max(np.abs(data)))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            im = ax.imshow(data.reshape(32, 32), cmap="RdBu_r", norm=norm, origin="upper")
        else:
            cmap = "cividis" if label.startswith("(a)") else "viridis"
            im = ax.imshow(data.reshape(32, 32), cmap=cmap, origin="upper")
        ax.set(xticks=[], yticks=[], xlabel=label)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
    saved(fig, "geometry")

    heat = dev.pivot(index="k", columns="alpha", values="macro_f1_mean")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5), layout="constrained")
    im = ax.imshow(heat, cmap="cividis", vmin=0.23, vmax=0.46, aspect="auto")
    ax.set(
        xticks=range(5),
        xticklabels=["0", ".25", ".5", ".75", "1"],
        yticks=range(10),
        yticklabels=heat.index,
        xlabel=r"Prevalence correction, $\alpha$",
        ylabel=r"Codebook clusters, $K$",
    )
    for (i, j), value in np.ndenumerate(heat.to_numpy()):
        red, green, blue, _ = im.cmap(im.norm(value))
        # Choose text with the higher WCAG contrast against the cell colour.
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in (red, green, blue)
        ]
        luminance = sum(
            v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
        )
        annotation_colour = "black" if luminance > 0.179 else "white"
        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color=annotation_colour,
        )
    ax.axhline(5.5, color="white", lw=2.4, ls="--")
    ax.axhline(5.5, color="#222222", lw=1.2, ls="--")
    saved(fig, "sensitivity")

    grouped = cluster.groupby("k").mean(numeric_only=True)
    curve = dev[dev.alpha == 0.5].sort_values("k")
    fig, axes = plt.subplots(
        3, 1, figsize=(COLUMN_WIDTH, 4.35), sharex=True, layout="constrained"
    )
    axes[0].errorbar(
        curve.k,
        curve.macro_f1_mean,
        yerr=curve.macro_f1_som_sd,
        color=PROCEDURE_COLOURS["k400"],
        marker="o",
        markersize=3,
        capsize=2,
    )
    axes[0].set_ylabel(r"(a) Macro $F_1$, $\alpha$ = .5")
    axes[1].plot(
        grouped.index,
        100 * grouped.singleton_unit_clusters / grouped.index,
        "s--",
        color="#0072B2",
        markersize=3,
        label="Singleton units",
    )
    axes[1].plot(
        grouped.index,
        100 * grouped.tiny_clusters / grouped.index,
        "^:",
        color="#D55E00",
        markersize=3,
        label=r"$<10$ training rows",
    )
    axes[1].set_ylabel(r"(b) Clusters (\%)")
    axes[1].legend(frameon=False, loc="upper left")
    axes[2].plot(grouped.index, grouped.silhouette, "d-", color="#009E73", markersize=3)
    axes[2].set(xlabel=r"Codebook clusters, $K$", ylabel="(c) Silhouette")
    for ax in axes:
        ax.grid(alpha=0.2)
    saved(fig, "fragmentation")

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8), layout="constrained")
    for method, label, marker, style in zip(
        methods[:3],
        ["40: confirmatory", "150: post-hoc", "400: post-hoc"],
        ["o", "s", "^"],
        ["-", "--", ":"],
        strict=True,
    ):
        row = transfer[transfer.method == method].iloc[0]
        ax.plot(
            [0, 1],
            [row.development_macro_f1, row.assessment_macro_f1],
            marker=marker,
            ls=style,
            color=PROCEDURE_COLOURS[method],
            label=label,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Development", "Assessment"],
        ylabel=r"Macro $F_1$",
        ylim=(0.30, 0.48),
        xlim=(-0.12, 1.12),
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.42))
    ax.grid(axis="y", alpha=0.2)
    saved(fig, "transfer")

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.8), layout="constrained")
    # Nominal classes share centred ticks. offsets separate the five procedures.
    markers = ["o", "s", "^", "d", "x"]
    offsets = np.linspace(-0.2, 0.2, len(methods))
    for method, label, marker, offset in zip(
        methods,
        ["40: confirmatory", "150: post-hoc", "400: post-hoc", "kNN", "Tree"],
        markers,
        offsets,
        strict=True,
    ):
        d = pc[(pc.method == method) & (pc.metric == "f1")].sort_values("class_id")
        ax.plot(
            d.class_id + offset,
            d["mean"],
            marker=marker,
            ls="none",
            color=PROCEDURE_COLOURS[method],
            markersize=5,
            label=label,
        )
    ax.set(
        xticks=range(10),
        xticklabels=classes.attack_category,
        xlim=(-0.45, 9.45),
        ylim=(0, 1.05),
        ylabel=r"Assessment class $F_1$",
    )
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.2)
    saved(fig, "classes")

    # Exact values supporting the prose and source-to-display inspection.
    for name, frame in [
        ("representation_values", som),
        ("sensitivity_values", dev),
        ("assessment_values", final),
        ("class_values", pc),
        ("bootstrap_values", bootstrap),
        ("geometry_values", geometry),
        ("fragmentation_values", fragments),
        ("structural_values", structural),
        ("transfer_values", transfer),
    ]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    context = {
        "assessment_som_training_seconds": float(runtime.training_seconds.sum()),
        "exploratory_execution_and_analysis_seconds": exploratory_execution["wall_seconds"]
        + exploratory_analysis["wall_seconds"],
        "assessment_late_qe_change_percent_mean": float(
            late.training_qe_change_60_to_70_percent.mean()
        ),
        "original_development_anchor": anchor.to_dict(),
        "original_development_best_by_side": original.sort_values(
            "macro_f1_mean", ascending=False
        )
        .groupby("side")
        .first()
        .reset_index()
        .to_dict("records"),
        "development_rare_anchor": dev_classes.query(
            "k == 40 and alpha == 1 and class_id in [2,5,7,8]"
        ).to_dict("records"),
        "baseline_configurations": baseline,
        "transformed_development": metadata["transformed_partitions"],
    }
    (OUT / "supporting_values.json").write_text(json.dumps(context, indent=2) + "\n")
    snapshot = {
        "inputs": INPUTS,
        "label_order": audit["attack_category_mapping"],
        "display_som_seed": 47425,
        "display_clustering_seed_reference": 47432,
        "component_features": ["dur", "sbytes"],
        "writes": "report/generated only",
        "fits": 0,
        "prediction_computations": 0,
        "presentation": {
            "iteration": 2,
            "representation_colours": REPRESENTATION_COLOURS,
            "procedure_colours": PROCEDURE_COLOURS,
            "nominal_text_points": 10,
            "column_width_inches": COLUMN_WIDTH,
            "text_width_inches": TEXT_WIDTH,
            "geometry_maps": ["cividis", "viridis", "RdBu_r centred at zero"],
            "sensitivity_map": "cividis with luminance-based annotation contrast",
            "redundancy": "markers and line styles; numerical heatmap annotations",
        },
    }
    snapshot["assets"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir()
        if p.is_file()
        and p.name
        not in {
            "asset_manifest.json",
            "preservation.json",
            "verification.json",
            "visual_review.json",
        }
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    for relative, fingerprint in INPUTS.items():
        assert hashlib.sha256((A4 / relative).read_bytes()).hexdigest() == fingerprint
    print(
        f"Created six PDF/PNG figures and four tables from {len(INPUTS)} frozen inputs; "
        "zero fits."
    )


if __name__ == "__main__":
    main()
