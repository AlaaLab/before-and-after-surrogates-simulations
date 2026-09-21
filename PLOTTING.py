#######################################################################################
# Author: Franny Dean
# Function: code for results tables
#######################################################################################

import pandas as pd
import numpy as np
import argparse

# ============================================================
# SUMMARIZE RESULTS
def summarize_results(results):
    """
    Calculate bias, Monte Carlo SD, RMSE,
    average estimated effect, and true effect.
    """

    rows = []

    group_cols = [
        "SURROGATE",
        "RHO_X1",
        "RHO_X2",
        "LABELED_FRACTION",
    ]

    for keys, df in results.groupby(
        group_cols
    ):

        surrogate, rho_x1, rho_x2, labeled_fraction = keys

        true = df["ATT_TRUE"].mean()

        for estimator in [
            "ATT_PPI",
            "ATT_PPI_CS",
            "ATT_NAIVE",
            "ATT_DIM",
            "ATT_ORACLE",
        ]:

            estimates = df[estimator]

            bias = (
                estimates.mean()
                -
                true
            )

            sd = estimates.std(
                ddof=1
            )

            rmse = np.sqrt(
                np.mean(
                    (
                        estimates
                        -
                        df["ATT_TRUE"]
                    ) ** 2
                )
            )

            rows.append({
                "SURROGATE": surrogate,
                "RHO_X1": rho_x1,
                "RHO_X2": rho_x2,
                "LABELED_FRACTION": labeled_fraction,
                "ESTIMATOR": estimator,
                "TRUE_ATT": true,
                "MEAN_ESTIMATE": estimates.mean(),
                "BIAS": bias,
                "MC_SD": sd,
                "RMSE": rmse,
            })

    return pd.DataFrame(rows)

# ============================================================
# TABLE 1

def make_coverage_bias_table(
    df,
    mae_col="HOLDOUT_MAE",
    r2_col="PRED_R2",
):

    methods = {
        "PPI": {
            "estimate": "ATT_PPI",
            "ci_low": "PPI_CI_LOW",
            "ci_high": "PPI_CI_HIGH",
        },
        "Naive": {
            "estimate": "ATT_NAIVE",
            "ci_low": "NAIVE_CI_LOW",
            "ci_high": "NAIVE_CI_HIGH",
        },
        "DIM": {
            "estimate": "ATT_DIM",
            "ci_low": "DIM_CI_LOW",
            "ci_high": "DIM_CI_HIGH",
        },
    }

    df = df.copy()

    for method, cols in methods.items():

        # Bias for each simulation run
        df[f"{method}_BIAS"] = (
            df[cols["estimate"]] - df["ATT_TRUE"]
        )

        # CI coverage for each simulation run
        df[f"{method}_COVERED"] = (
            (df[cols["ci_low"]] <= df["ATT_TRUE"]) &
            (df[cols["ci_high"]] >= df["ATT_TRUE"])
        )

        # CI width for each simulation run
        df[f"{method}_WIDTH"] = (
            df[cols["ci_high"]] - df[cols["ci_low"]]
        )

    # SUMMARIZE MAIN SIMULATION RESULTS
    rows = []

    for (dgp, rho_x1, rho_x2), group in df.groupby(
        ["DGP", "RHO_X1", "RHO_X2"]
    ):

        row = {
            "DGP": dgp,
            "RHO_X1": rho_x1,
            "RHO_X2": rho_x2,
        }

        n = len(group)

        for method in methods:

            # Coverage
            row[f"{method}_COVERAGE"] = (
                group[f"{method}_COVERED"].mean()
            )

            # Bias + SE
            bias_values = group[f"{method}_BIAS"]

            row[f"{method}_BIAS"] = (
                bias_values.mean()
            )

            row[f"{method}_BIAS_SE"] = (
                bias_values.std(ddof=1) / np.sqrt(n)
            )

            # Interval width + SE
            width_values = group[f"{method}_WIDTH"]

            row[f"{method}_WIDTH"] = (
                width_values.mean()
            )

            row[f"{method}_WIDTH_SE"] = (
                width_values.std(ddof=1) / np.sqrt(n)
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    # SUMMARIZE MODEL PREDICTION DIAGNOSTICS
    diagnostic_rows = []

    for (dgp, rho_x1, rho_x2), group in df.groupby(
        ["DGP", "RHO_X1", "RHO_X2"]
    ):

        n = len(group)

        mae_values = group[mae_col]
        r2_values = group[r2_col]

        diagnostic_rows.append({
            "DGP": dgp,
            "RHO_X1": rho_x1,
            "RHO_X2": rho_x2,

            # MAE
            "MAE": mae_values.mean(),
            "MAE_SE": mae_values.std(ddof=1) / np.sqrt(n),

            # R2
            "R2": r2_values.mean(),
            "R2_SE": r2_values.std(ddof=1) / np.sqrt(n),
        })

    diagnostics_summary = pd.DataFrame(diagnostic_rows)

    # MERGE DIAGNOSTICS INTO MAIN TABLE
    summary = summary.merge(
        diagnostics_summary,
        on=["DGP", "RHO_X1", "RHO_X2"],
        how="left",
    )

    # SORT
    summary = summary.sort_values(
        ["DGP", "RHO_X1", "RHO_X2"]
    ).reset_index(drop=True)

    return summary

def format_table1(table):

    formatted = pd.DataFrame(index=table.index)

    # MODEL DIAGNOSTICS
    formatted[("Model", "MAE")] = [
        f"{mae:.3f} ({se:.3f})"
        for mae, se in zip(
            table["MAE"],
            table["MAE_SE"],
        )
    ]

    formatted[("Model", r"$R^2$")] = [
        f"{r2:.3f} ({se:.3f})"
        for r2, se in zip(
            table["R2"],
            table["R2_SE"],
        )
    ]

    # PPI
    formatted[("PPI", "Coverage")] = (
        100 * table["PPI_COVERAGE"]
    ).map(lambda x: f"{x:.1f}\\%")

    formatted[("PPI", "Bias")] = [
        f"{bias:.3f} ({se:.3f})"
        for bias, se in zip(
            table["PPI_BIAS"],
            table["PPI_BIAS_SE"],
        )
    ]

    formatted[("PPI", "Interval")] = [
        f"{width:.3f} ({se:.3f})"
        for width, se in zip(
            table["PPI_WIDTH"],
            table["PPI_WIDTH_SE"],
        )
    ]

    # NAIVE
    formatted[("Naive", "Bias")] = [
        f"{bias:.3f} ({se:.3f})"
        for bias, se in zip(
            table["Naive_BIAS"],
            table["Naive_BIAS_SE"],
        )
    ]

    # DIM
    formatted[("DIM", "Coverage")] = (
        100 * table["DIM_COVERAGE"]
    ).map(lambda x: f"{x:.1f}\\%")

    formatted[("DIM", "Bias")] = [
        f"{bias:.3f} ({se:.3f})"
        for bias, se in zip(
            table["DIM_BIAS"],
            table["DIM_BIAS_SE"],
        )
    ]

    formatted[("DIM", "Interval")] = [
        f"{width:.3f} ({se:.3f})"
        for width, se in zip(
            table["DIM_WIDTH"],
            table["DIM_WIDTH_SE"],
        )
    ]

    # INDEX: DGP + RHO X1 + RHO X2
    formatted.index = pd.MultiIndex.from_arrays(
        [
            table["DGP"],
            table["RHO_X1"],
            table["RHO_X2"],
        ],
        names=[
            "DGP",
            r"$\rho_{X1}$",
            r"$\rho_{X2}$",
        ],
    )

    # MULTI-LEVEL COLUMNS
    formatted.columns = pd.MultiIndex.from_tuples(
        formatted.columns
    )

    return formatted

def make_latex_table(formatted_table):

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{cc|cc|ccc|c|cc}")
    lines.append(r"\toprule")

    # Method names
    lines.append(
        r"& & \multicolumn{2}{c|}{Model Prediction} "
        r"& \multicolumn{3}{c|}{PPI} "
        r"& Naive "
        r"& \multicolumn{2}{c}{DIM} \\"
    )

    # Metric names
    lines.append(
        r"$\rho_{X1}$ & $\rho_{X2}$ "
        r"& MAE & $R^2$ "
        r"& Coverage & Bias & Interval "
        r"& Bias "
        r"& Bias & Interval \\"
    )

    lines.append(r"\midrule")

    # ACCESS MULTIINDEX ROWS
    for dgp in formatted_table.index.get_level_values(
        "DGP"
    ).unique():

        dgp_df = formatted_table.xs(
            dgp,
            level="DGP",
            drop_level=True,
        )

        # DGP PANEL HEADER
        dgp_label = str(dgp).replace("-", " ").title()

        lines.append(
            rf"\multicolumn{{10}}{{l}}{{\textbf{{{dgp_label} DGP}}}} \\"
        )

        lines.append(r"\addlinespace[2pt]")

        # ROWS WITHIN DGP
        for (rho_x1, rho_x2), row in dgp_df.iterrows():

            values = [
                f"{rho_x1:g}",
                f"{rho_x2:g}",

                # Model diagnostics
                row[("Model", "MAE")],
                row[("Model", r"$R^2$")],

                # PPI
                row[("PPI", "Coverage")],
                row[("PPI", "Bias")],
                row[("PPI", "Interval")],

                # Naive
                row[("Naive", "Bias")],

                # DIM
                row[("DIM", "Bias")],
                row[("DIM", "Interval")],
            ]

            lines.append(
                " & ".join(values) + r" \\"
            )

        lines.append(r"\addlinespace[3pt]")

    # TABLE FOOTER
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\label{tab:coverage_bias}")

    lines.append(r"\end{table}")

    return "\n".join(lines)

# ============================================================
# TABLE 2

def make_estimate_table2_by_labeled_fraction(
    df,
    rho_x1=None,
    rho_x2=None,
    rho_z=None,
):

    # FILTER TO ONE RHO CONFIGURATION
    plot_df = df.copy()

    if rho_x1 is not None:
        plot_df = plot_df[
            np.isclose(plot_df["RHO_X1"], rho_x1)
        ]

    if rho_x2 is not None:
        plot_df = plot_df[
            np.isclose(plot_df["RHO_X2"], rho_x2)
        ]

    if rho_z is not None:
        plot_df = plot_df[
            np.isclose(plot_df["RHO_Z"], rho_z)
        ]

    # ESTIMATORS
    estimators = {
        "DIM": {
            "att": "ATT_DIM",
            "ci_low": "DIM_CI_LOW",
            "ci_high": "DIM_CI_HIGH",
        },
        "PPI": {
            "att": "ATT_PPI",
            "ci_low": "PPI_CI_LOW",
            "ci_high": "PPI_CI_HIGH",
        },
        "Naive": {
            "att": "ATT_NAIVE",
            "ci_low": "NAIVE_CI_LOW",
            "ci_high": "NAIVE_CI_HIGH",
        },
    }

    tables = {}

    # MAKE ONE TABLE PER DGP
    for dgp in ["linear", "non-linear"]:

        dgp_df = plot_df[
            plot_df["DGP"].str.lower() == dgp.lower()
        ].copy()

        if dgp_df.empty:
            continue

        rows = []

        # Fractions are allowed to differ across DGPs.
        fractions = sorted(
            dgp_df["LABELED_FRACTION"].dropna().unique()
        )

        for fraction in fractions:

            row = {
                "Labeled data (%)": 100 * fraction
            }

            # DIM / PPI / NAIVE
            for estimator, cols in estimators.items():

                # PPI is not reported at 100% labeled data
                if estimator == "PPI" and np.isclose(fraction, 1.0):
                    continue

                estimator_df = dgp_df[
                    np.isclose(
                        dgp_df["LABELED_FRACTION"],
                        fraction,
                    )
                ].copy()

                if estimator_df.empty:
                    continue

                # RUN-LEVEL BIAS
                estimator_df["BIAS"] = (
                    estimator_df[cols["att"]]
                    - estimator_df["ATT_TRUE"]
                )

                # RUN-LEVEL CI WIDTH
                estimator_df["WIDTH"] = (
                    estimator_df[cols["ci_high"]]
                    - estimator_df[cols["ci_low"]]
                )

                # RUN-LEVEL COVERAGE
                estimator_df["COVERED"] = (
                    (estimator_df[cols["ci_low"]]
                     <= estimator_df["ATT_TRUE"])
                    &
                    (estimator_df[cols["ci_high"]]
                     >= estimator_df["ATT_TRUE"])
                )

                n = len(estimator_df)

                # COVERAGE
                coverage = estimator_df["COVERED"].mean()

                row[f"{estimator}_COVERAGE"] = coverage

                # BIAS + MONTE CARLO SE
                bias = estimator_df["BIAS"].mean()

                if n > 1:
                    bias_se = (
                        estimator_df["BIAS"].std(ddof=1)
                        / np.sqrt(n)
                    )
                else:
                    bias_se = np.nan

                row[f"{estimator}_BIAS"] = bias
                row[f"{estimator}_BIAS_SE"] = bias_se

                # INTERVAL WIDTH + MONTE CARLO SE
                width = estimator_df["WIDTH"].mean()

                if n > 1:
                    width_se = (
                        estimator_df["WIDTH"].std(ddof=1)
                        / np.sqrt(n)
                    )
                else:
                    width_se = np.nan

                row[f"{estimator}_WIDTH"] = width
                row[f"{estimator}_WIDTH_SE"] = width_se

            rows.append(row)

        table = pd.DataFrame(rows)

        table["Labeled data (%)"] = table[
            "Labeled data (%)"
        ].map(lambda x: f"{x:g}\\%")

        table = table.set_index("Labeled data (%)")

        formatted = pd.DataFrame(index=table.index)

        formatted["DIM Coverage"] = (
            100 * table["DIM_COVERAGE"]
        ).map(lambda x: f"{x:.1f}\\%")

        formatted["DIM Bias"] = [
            f"{bias:.3f} ({se:.3f})"
            for bias, se in zip(
                table["DIM_BIAS"],
                table["DIM_BIAS_SE"],
            )
        ]

        formatted["DIM Interval"] = [
            f"{width:.3f} ({se:.3f})"
            for width, se in zip(
                table["DIM_WIDTH"],
                table["DIM_WIDTH_SE"],
            )
        ]
        # PPI columns are only present where PPI exists.
        if "PPI_COVERAGE" in table.columns:

            formatted["PPI Coverage"] = (
                100 * table["PPI_COVERAGE"]
            ).map(lambda x: f"{x:.1f}\\%")

            formatted["PPI Bias"] = [
                f"{bias:.3f} ({se:.3f})"
                for bias, se in zip(
                    table["PPI_BIAS"],
                    table["PPI_BIAS_SE"],
                )
            ]

            formatted["PPI Interval"] = [
                f"{width:.3f} ({se:.3f})"
                for width, se in zip(
                    table["PPI_WIDTH"],
                    table["PPI_WIDTH_SE"],
                )
            ]

        formatted["Naive Bias"] = [
            f"{bias:.3f} ({se:.3f})"
            for bias, se in zip(
                table["Naive_BIAS"],
                table["Naive_BIAS_SE"],
            )
        ]

        tables[dgp] = formatted

    return tables

def combine_linear_and_nonlinear_to_latex(
    table1,
    table2,
):

    df_linear = table1["linear"].copy()
    df_nonlinear = table2["non-linear"].copy()

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    # DGP + labeled fraction + 3 PPI + 3 DIM + 1 Naive = 9
    lines.append(r"\begin{tabular}{lc|ccc|ccc|c}")
    lines.append(r"\toprule")

    lines.append(
        r"& & \multicolumn{3}{c|}{PPI} "
        r"& \multicolumn{3}{c|}{DIM} "
        r"& Naive \\"
    )

    lines.append(
        r"DGP & Labeled data "
        r"& Coverage & Bias & Interval "
        r"& Coverage & Bias & Interval "
        r"& Bias \\"
    )

    lines.append(r"\midrule")

    for i, (fraction, row) in enumerate(df_linear.iterrows()):

        values = [
            "Linear" if i == 0 else "",
            str(fraction),

            # PPI
            str(row["PPI Coverage"])
                if "PPI Coverage" in row.index else "--",
            str(row["PPI Bias"])
                if "PPI Bias" in row.index else "--",
            str(row["PPI Interval"])
                if "PPI Interval" in row.index else "--",

            # DIM
            str(row["DIM Coverage"]),
            str(row["DIM Bias"]),
            str(row["DIM Interval"]),

            # Naive
            str(row["Naive Bias"]),
        ]

        lines.append(
            " & ".join(values) + r" \\"
        )

    lines.append(r"\addlinespace[3pt]")
    lines.append(r"\midrule")
    lines.append(r"\addlinespace[3pt]")

    for i, (fraction, row) in enumerate(
        df_nonlinear.iterrows()
    ):

        values = [
            "Non-linear" if i == 0 else "",
            str(fraction),

            # PPI
            str(row["PPI Coverage"])
                if "PPI Coverage" in row.index else "--",
            str(row["PPI Bias"])
                if "PPI Bias" in row.index else "--",
            str(row["PPI Interval"])
                if "PPI Interval" in row.index else "--",

            # DIM
            str(row["DIM Coverage"]),
            str(row["DIM Bias"]),
            str(row["DIM Interval"]),

            # Naive
            str(row["Naive Bias"]),
        ]

        lines.append(
            " & ".join(values) + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\end{table}")

    return "\n".join(lines)
