#######################################################################################
# Author: Franny Dean
# Function: runs the before and after difference and the debias estimator simulations
#######################################################################################

import numpy as np
import pandas as pd
import argparse
import os

from sklearn.metrics import r2_score, mean_absolute_error

from utils.PLOTTING import *
from utils.SIMULATE import *

# ============================================================
# SETTINGS

parser = argparse.ArgumentParser()

# General
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--cov_shift_test", action="store_true")
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--save_path", type=str, default="")

# Population sizes
parser.add_argument("--n_train", type=int, default=20_000)
parser.add_argument("--n_obs", type=int, default=10_000)

# Monte Carlo
parser.add_argument("--n_trials", type=int, default=50)

# Treatment
parser.add_argument("--p_treat", type=float, default=0.50)
parser.add_argument("--confounded", action="store_true")

# Simulation
parser.add_argument(
    "--rho_z",
    type=float,
    default=1.01,
)
parser.add_argument(
    "--z_method",
    type=str,
    choices=["linear", "non-linear"],
    default="linear",
)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--treatment_bump", type=float, default=0.5)

# Surrogate quality
parser.add_argument(
    "--rho_x_combos",
    type=float,
    nargs=2,
    action="append",
    default=None,
)

# Outcome noise
parser.add_argument("--sigma_y", type=float, default=0.05)

# Time
parser.add_argument("--t_pre", type=int, default=0)
parser.add_argument("--t_post", type=int, default=2)
parser.add_argument("--t_end", type=int, default=7)

# PPI correction
parser.add_argument(
    "--labeled_fractions",
    nargs="+",
    type=float,
    default=[0.05],
)

args = parser.parse_args()

RANDOM_STATE = args.random_state
COV_SHIFT_TEST = args.cov_shift_test
EXP_NAME = args.exp_name
HEADPATH = args.save_path
rng = np.random.default_rng(RANDOM_STATE)

N_TRAIN = args.n_train
N_OBS = args.n_obs

N_TRIALS = args.n_trials

P_TREAT = args.p_treat
CONFOUNDED = args.confounded

RHO_Z = args.rho_z
Z_METHOD = args.z_method
BETA = args.beta
TREATMENT_BUMP = args.treatment_bump

RHO_X_COMBOS = [tuple(pair) for pair in args.rho_x_combos]

SIGMA_Y = args.sigma_y

TIME_POINTS = np.arange(0, 10)

TRAIN_K = [1, 2, 3, 4, 5, 6, 7]

T_PRE = args.t_pre
T_POST = args.t_post
T_END = args.t_end

K_PRE_END = T_END - T_PRE
K_POST_END = T_END - T_POST

LABELED_FRACTIONS = args.labeled_fractions

# SAVE DATA 
def save_simulation_data(training, obs, save_path):
    """
    Save the complete training and observational dictionaries.
    """

    # Save training dictionary
    np.savez(
        f"{save_path}_training.npz",
        **{
            key: value
            for key, value in training.items()
            if isinstance(value, np.ndarray)
        }
    )

    # Save observational dictionary
    np.savez(
        f"{save_path}_observational.npz",
        **{
            key: value
            for key, value in obs.items()
            if isinstance(value, np.ndarray)
        }
    )

    # Save any DataFrames contained in the dictionaries separately
    if "data" in training and isinstance(training["data"], pd.DataFrame):
        training["data"].to_csv(
            f"{save_path}_training_data.csv",
            index=False
        )

    print(f"Saved training data to:")
    print(f"  {save_path}_training.npz")

    print(f"Saved observational data to:")
    print(f"  {save_path}_observational.npz")

    if "data" in training:
        print(f"Saved training DataFrame to:")
        print(f"  {save_path}_training_data.csv")

# SINGLE SIMULATION
def run_single_trial(
    rho_x1,
    rho_x2,
    labeled_fraction,
    rng,
    add_cov_shif_test=COV_SHIFT_TEST,
    check_distribution=False
):

    # Generate independent training population
    training = generate_training_data(
        n_train=N_TRAIN,
        time_points=TIME_POINTS,
        train_k=TRAIN_K,
        rho_x1=rho_x1,
        rho_x2=rho_x2,
        rho_z=RHO_Z,
        sigma_y=SIGMA_Y,
        z_method=Z_METHOD,
        beta=BETA,
        t_post=T_POST,
        rng=rng,
        confounded=CONFOUNDED,
    )

    # Independent training holdout
    training, train_holdout = make_training_holdout(
        training,
        n_holdout=5000,
        rng=rng,
    )

    # Fit prediction model
    model = fit_prediction_model(
        training,
        random_state=RANDOM_STATE,
    )

    # In-sample R2
    in_sample_pred = predict_mu(
        model,
        training["data"]["X1"].values,
        training["data"]["X2"].values,
        training["data"]["K"].values,
    )

    in_sample_r2 = r2_score(
        training["data"]["Y"].values,
        in_sample_pred,
    )

    # Prediction R2 on independent training holdout
    holdout_pred = predict_mu(
        model,
        train_holdout["X1"],
        train_holdout["X2"],
        train_holdout["K"],
    )

    prediction_r2 = r2_score(
        train_holdout["Y"],
        holdout_pred,
    )

    # Generate observational/treatment population
    obs = generate_observational_data(
        n_obs=N_OBS,
        rho_x1=rho_x1,
        rho_x2=rho_x2,
        rho_z=RHO_Z,
        t_pre=T_PRE,
        t_post=T_POST,
        t_end=T_END,
        treatment_bump=TREATMENT_BUMP,
        sigma_y=SIGMA_Y,
        p_treat=P_TREAT,
        z_method=Z_METHOD,
        beta=BETA,
        rng=rng,
        confounded=CONFOUNDED,
    )

    # ============================================================
    # PREDICTION MAE DIAGNOSTICS
    train_mae = mean_absolute_error(
        training["data"]["Y"].values,
        in_sample_pred,
    )

    holdout_mae = mean_absolute_error(
        train_holdout["Y"],
        holdout_pred,
    )

    treated_idx_all = np.where(
        obs["D"] == 1
    )[0]

    untreated_idx_all = np.where(
        obs["D"] == 0
    )[0]

    # ----- Treated -----

    treated_x1_post = obs["X11"][
        treated_idx_all,
        obs["t_post"]
    ]

    treated_x2_post = obs["X21"][
        treated_idx_all,
        obs["t_post"]
    ]

    treated_y_post = obs["Y1"][
        treated_idx_all,
        obs["t_end"]
    ]

    treated_pred = predict_mu(
        model,
        treated_x1_post,
        treated_x2_post,
        obs["t_end"] - obs["t_post"],
    )

    treated_mae = mean_absolute_error(
        treated_y_post,
        treated_pred,
    )

    # ----- Untreated -----

    untreated_x1_post = obs["X10"][
        untreated_idx_all,
        obs["t_post"]
    ]

    untreated_x2_post = obs["X20"][
        untreated_idx_all,
        obs["t_post"]
    ]

    untreated_y_post = obs["Y0"][
        untreated_idx_all,
        obs["t_end"]
    ]

    untreated_pred = predict_mu(
        model,
        untreated_x1_post,
        untreated_x2_post,
        obs["t_end"] - obs["t_post"],
    )

    untreated_mae = mean_absolute_error(
        untreated_y_post,
        untreated_pred,
    )

    if check_distribution:
        # ============================================================
        # DISTRIBUTION DIAGNOSTICS:
        # TRAINING POPULATION VS OBSERVATIONAL POPULATION

        def summarize_distribution(x):
            x = np.asarray(x).reshape(-1)

            return {
                "mean": np.mean(x),
                "sd": np.std(x, ddof=1),
                "min": np.min(x),
                "q01": np.quantile(x, 0.01),
                "q25": np.quantile(x, 0.25),
                "median": np.median(x),
                "q75": np.quantile(x, 0.75),
                "q99": np.quantile(x, 0.99),
                "max": np.max(x),
            }

        distribution_rows = []

        # Flatten all longitudinal time points so that we compare
        # the full trajectories in the two populations.
        treated_idx = np.where(obs["D"] == 1)[0]

        variables = {

            "Y": (
                training["Y0"],
                obs["Y1"][treated_idx],
            ),

            "Z1": (
                training["Z1"],
                obs["Z11"][treated_idx],
            ),

            "Z2": (
                training["Z2"],
                obs["Z21"][treated_idx],
            ),

        }

        for variable, (train_values, obs_values) in variables.items():

            train_summary = summarize_distribution(train_values)
            obs_summary = summarize_distribution(obs_values)

            distribution_rows.append({
                "VARIABLE": variable,

                "TRAIN_MEAN": train_summary["mean"],
                "OBS_MEAN": obs_summary["mean"],
                "MEAN_DIFF": (
                    obs_summary["mean"]
                    - train_summary["mean"]
                ),

                "TRAIN_SD": train_summary["sd"],
                "OBS_SD": obs_summary["sd"],

                "TRAIN_MIN": train_summary["min"],
                "OBS_MIN": obs_summary["min"],

                "TRAIN_Q01": train_summary["q01"],
                "OBS_Q01": obs_summary["q01"],

                "TRAIN_Q25": train_summary["q25"],
                "OBS_Q25": obs_summary["q25"],

                "TRAIN_MEDIAN": train_summary["median"],
                "OBS_MEDIAN": obs_summary["median"],

                "TRAIN_Q75": train_summary["q75"],
                "OBS_Q75": obs_summary["q75"],

                "TRAIN_Q99": train_summary["q99"],
                "OBS_Q99": obs_summary["q99"],

                "TRAIN_MAX": train_summary["max"],
                "OBS_MAX": obs_summary["max"],
            })

        distribution_diagnostics = pd.DataFrame(
            distribution_rows
        )

        print("\n")
        print("=" * 80)
        print("TRAINING VS OBSERVATIONAL DISTRIBUTIONS")
        print("=" * 80)

        print(
            distribution_diagnostics.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # Select treated individuals
    treated_idx = np.where(
        obs["D"] == 1
    )[0]

    n_labeled = max(
        2,
        int(
            labeled_fraction
            * len(treated_idx)
        ),
    )

    labeled_idx = rng.choice(
        treated_idx,
        size=n_labeled,
        replace=False,
    )

    # TRUE ATT
    att_true = true_att(obs)

    # PPI WITHOUT COVARIATE SHIFT
    ppi_result = ppi_att(
        obs,
        labeled_idx,
        model,
        train_holdout,
        rng,
        use_covariate_shift=False,
    )

    att_ppi = ppi_result["ppi"]
    se_ppi = ppi_result["se"]

    # Initialize so it exists even when covariate shift is disabled
    ppi_cs_result = None

    # PPI WITH COVARIATE SHIFT
    if add_cov_shif_test:

        ppi_cs_result = ppi_att(
            obs,
            labeled_idx,
            model,
            train_holdout,
            rng,
            use_covariate_shift=True,
        )

        att_ppi_cs = ppi_cs_result["ppi"]
        se_ppi_cs = ppi_cs_result["se"]

    else:

        att_ppi_cs = np.nan
        se_ppi_cs = np.nan

    # NAIVE
    naive_result = naive_unlabeled_att(
        obs,
        labeled_idx,
        model,
    )

    att_naive = naive_result["att"]
    se_naive = naive_result["se"]

    # DIFFERENCE IN MEANS
    dim_result = dim_ate(
        obs,
        labeled_idx,
        rng
    )

    att_dim = dim_result["att"]
    se_dim = dim_result["se"]

    # ORACLE ATT
    oracle = np.mean(
        obs["Y1"][
            labeled_idx,
            obs["t_end"]
        ]
        -
        obs["Y0"][
            labeled_idx,
            obs["t_end"]
        ]
    )

    # PPI DIAGNOSTICS
    ppi_diagnostics = ppi_result[
        "diagnostics"
    ].copy()

    ppi_diagnostics["SURROGATE"] = None

    ppi_diagnostics["RHO_X1"] = rho_x1
    ppi_diagnostics["RHO_X2"] = rho_x2
    ppi_diagnostics["RHO_Z"] = RHO_Z
    ppi_diagnostics["Z_METHOD"] = Z_METHOD

    ppi_diagnostics["LABELED_FRACTION"] = labeled_fraction
    ppi_diagnostics["N_LABELED"] = n_labeled
    ppi_diagnostics["N_TREATED"] = len(treated_idx)
    ppi_diagnostics["PPI_VERSION"] = "standard"

    ppi_diagnostics[
        "POST_RECTIFIER_MEAN"
    ] = ppi_result["post_rectifier"]

    ppi_diagnostics[
        "ABS_POST_RECTIFIER_MEAN"
    ] = ppi_result["abs_post_rectifier"]

    ppi_diagnostics[
        "CONTROL_RECTIFIER_MEAN"
    ] = ppi_result["control_rectifier"]

    ppi_diagnostics[
        "ABS_CONTROL_RECTIFIER_MEAN"
    ] = ppi_result["abs_control_rectifier"]

    # Covariate-shift diagnostics
    if add_cov_shif_test:

        ppi_cs_diagnostics = ppi_cs_result[
            "diagnostics"
        ].copy()

        ppi_cs_diagnostics["SURROGATE"] = None

        ppi_cs_diagnostics["RHO_X1"] = rho_x1
        ppi_cs_diagnostics["RHO_X2"] = rho_x2
        ppi_cs_diagnostics["RHO_Z"] = RHO_Z
        ppi_cs_diagnostics["Z_METHOD"] = Z_METHOD

        ppi_cs_diagnostics[
            "LABELED_FRACTION"
        ] = labeled_fraction

        ppi_cs_diagnostics[
            "N_LABELED"
        ] = n_labeled

        ppi_cs_diagnostics[
            "N_TREATED"
        ] = len(treated_idx)

        ppi_cs_diagnostics[
            "PPI_VERSION"
        ] = "covariate_shift"

        ppi_cs_diagnostics[
            "POST_RECTIFIER_MEAN"
        ] = ppi_cs_result["post_rectifier"]

        ppi_cs_diagnostics[
            "ABS_POST_RECTIFIER_MEAN"
        ] = ppi_cs_result["abs_post_rectifier"]

        ppi_cs_diagnostics[
            "CONTROL_RECTIFIER_MEAN"
        ] = ppi_cs_result["control_rectifier"]

        ppi_cs_diagnostics[
            "ABS_CONTROL_RECTIFIER_MEAN"
        ] = ppi_cs_result["abs_control_rectifier"]

        ppi_diagnostics = pd.concat(
            [
                ppi_diagnostics,
                ppi_cs_diagnostics,
            ],
            ignore_index=True,
        )

    # Final result
    result = {

        "ATT_TRUE": att_true,

        # Standard PPI
        "ATT_PPI": att_ppi,
        "SE_PPI": se_ppi,

        # Covariate-shifted PPI
        "ATT_PPI_CS": att_ppi_cs,
        "SE_PPI_CS": se_ppi_cs,

        # Naive
        "ATT_NAIVE": att_naive,
        "SE_NAIVE": se_naive,

        # Prediction MAE
        "TRAIN_MAE": train_mae,
        "HOLDOUT_MAE": holdout_mae,

        "OBS_TREATED_MAE": treated_mae,

        "OBS_UNTREATED_MAE": untreated_mae,

        # Difference in means
        "ATT_DIM": att_dim,
        "SE_DIM": se_dim,
        
        # Oracle
        "ATT_ORACLE": oracle,

        # Prediction diagnostics
        "PRED_R2": prediction_r2,
        "IN_SAMPLE_R2": in_sample_r2,

        # Sample sizes
        "N_LABELED": n_labeled,
        "N_TREATED": len(treated_idx),

        # Simulation parameters
        "RHO_X1": rho_x1,
        "RHO_X2": rho_x2,
        "RHO_Z": RHO_Z,
        "Z_METHOD": Z_METHOD,

        "LABELED_FRACTION": labeled_fraction,
    }

    # Add covariate-shift diagnostics only when available
    if ppi_cs_result is not None:

        result.update({
            "CS_ESS":
                ppi_cs_result["ess_control"],

            "CS_MEAN_WEIGHT":
                ppi_cs_result["mean_control_weight"],

            "CS_MAX_WEIGHT":
                ppi_cs_result["max_control_weight"],

            "CS_MIN_WEIGHT":
                ppi_cs_result["min_control_weight"],
        })

    return {
        "result": result,
        "diagnostics": ppi_diagnostics,
        "training": training,
        "train_holdout": train_holdout,
        "obs": obs,
        "labeled_idx": labeled_idx,
    }

# RUN SIMULATION GRID
def run_simulation_grid(
    n_trials=N_TRIALS,
):
    """
    Main simulation grid.

    Dimensions:

        rho_x1
        rho_x2
        labeled fraction
        trial
    """

    results = []
    ppi_diagnostics = []

    for rho_x1, rho_x2 in RHO_X_COMBOS:

        surrogate_name = (
            f"X1={rho_x1:.2f}_X2={rho_x2:.2f}"
        )

        print(
            f"\nSurrogate: {surrogate_name}"
        )

        print(
            f"  rho_X1={rho_x1}"
            f"  rho_X2={rho_x2}"
        )

        for labeled_fraction in LABELED_FRACTIONS:

            print(
                f"  labeled fraction = "
                f"{labeled_fraction:.0%}"
            )

            for trial in range(n_trials):

                trial_rng = np.random.default_rng(
                    RANDOM_STATE
                    + trial
                    + int(10000 * labeled_fraction)
                    + int(100000 * rho_x1)
                    + int(1000000 * rho_x2)
                )

                trial_output = run_single_trial(
                    rho_x1=rho_x1,
                    rho_x2=rho_x2,
                    labeled_fraction=labeled_fraction,
                    rng=trial_rng,
                )

                # Main results
                result = trial_output["result"]

                result["SURROGATE"] = surrogate_name
                result["TRIAL"] = trial

                results.append(result)

                # Diagnostics
                diag = trial_output[
                    "diagnostics"
                ].copy()

                diag["SURROGATE"] = surrogate_name
                diag["TRIAL"] = trial

                ppi_diagnostics.append(diag)

    results_df = pd.DataFrame(results)

    ppi_diagnostics_df = pd.concat(
        ppi_diagnostics,
        ignore_index=True,
    )

    return (
        results_df,
        ppi_diagnostics_df,
    )


# RUN
results, ppi_diagnostics = run_simulation_grid(
    n_trials=N_TRIALS,
)

# CALCULATE 95% CI COVERAGE
results["PPI_CI_LOW"] = (
    results["ATT_PPI"]
    -
    1.96 * results["SE_PPI"]
)

results["PPI_CI_HIGH"] = (
    results["ATT_PPI"]
    +
    1.96 * results["SE_PPI"]
)

results["PPI_CS_CI_LOW"] = (
    results["ATT_PPI_CS"]
    -
    1.96 * results["SE_PPI_CS"]
)

results["PPI_CS_CI_HIGH"] = (
    results["ATT_PPI_CS"]
    +
    1.96 * results["SE_PPI_CS"]
)

results["PPI_CS_COVERED"] = (
    (results["ATT_TRUE"] >= results["PPI_CS_CI_LOW"])
    &
    (results["ATT_TRUE"] <= results["PPI_CS_CI_HIGH"])
)

results["NAIVE_CI_LOW"] = (
    results["ATT_NAIVE"]
    -
    1.96 * results["SE_NAIVE"]
)

results["NAIVE_CI_HIGH"] = (
    results["ATT_NAIVE"]
    +
    1.96 * results["SE_NAIVE"]
)

results["DIM_CI_LOW"] = (
    results["ATT_DIM"]
    -
    1.96 * results["SE_DIM"]
)

results["DIM_CI_HIGH"] = (
    results["ATT_DIM"]
    +
    1.96 * results["SE_DIM"]
)


# Did the 95% CI contain the true ATT?
results["PPI_COVERED"] = (
    (results["ATT_TRUE"] >= results["PPI_CI_LOW"])
    &
    (results["ATT_TRUE"] <= results["PPI_CI_HIGH"])
)

results["NAIVE_COVERED"] = (
    (results["ATT_TRUE"] >= results["NAIVE_CI_LOW"])
    &
    (results["ATT_TRUE"] <= results["NAIVE_CI_HIGH"])
)

results["DIM_COVERED"] = (
    (results["ATT_TRUE"] >= results["DIM_CI_LOW"])
    &
    (results["ATT_TRUE"] <= results["DIM_CI_HIGH"])
)

# Save updated results
results.to_csv(
    f"{HEADPATH}results_{N_TRIALS}_{Z_METHOD}_{RHO_Z}_{len(RHO_X_COMBOS)}_{len(LABELED_FRACTIONS)}_{EXP_NAME}.csv",
    index=False,
)

ppi_diagnostics.to_csv(
    f"{HEADPATH}ppi_diagnostics_{N_TRIALS}_{Z_METHOD}_{RHO_Z}_{len(RHO_X_COMBOS)}_{len(LABELED_FRACTIONS)}_{EXP_NAME}.csv",
    index=False,
)
summary = summarize_results(
    results
)

print("\n")
print("=" * 80)
print("SIMULATION SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# COVERAGE SUMMARY
coverage_summary = (
    results
    .groupby([
        "SURROGATE",
        "LABELED_FRACTION",
    ])
    .agg(
        PPI_COVERAGE=("PPI_COVERED", "mean"),
        PPI_CS_COVERAGE=("PPI_CS_COVERED", "mean"),

        NAIVE_COVERAGE=("NAIVE_COVERED", "mean"),
        DIM_COVERAGE=("DIM_COVERED", "mean"),

        MEAN_PPI_SE=("SE_PPI", "mean"),
        MEAN_PPI_CS_SE=("SE_PPI_CS", "mean"),

        MEAN_NAIVE_SE=("SE_NAIVE", "mean"),
        MEAN_DIM_SE=("SE_DIM", "mean"),
    )
    .reset_index()
)

coverage_summary["PPI_COVERAGE"] *= 100
coverage_summary["PPI_CS_COVERAGE"] *= 100
coverage_summary["NAIVE_COVERAGE"] *= 100
coverage_summary["DIM_COVERAGE"] *= 100

print("\n")
print("=" * 80)
print("95% CI COVERAGE")
print("=" * 80)

print(
    coverage_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

# PREDICTION QUALITY
prediction_summary = (
    results
    .groupby("SURROGATE")["IN_SAMPLE_R2"]
    .agg(["mean", "std"])
)

print("\nPrediction quality (in sample):")
print(prediction_summary)


prediction_summary = (
    results
    .groupby("SURROGATE")["PRED_R2"]
    .agg(["mean", "std"])
)

print("\nPrediction quality (out sample):")
print(prediction_summary)


prediction_summary = (
    results
    .groupby("SURROGATE")["TRAIN_MAE"]
    .agg(["mean", "std"])
)

print("\nPrediction quality (training MAE):")
print(prediction_summary)


prediction_summary = (
    results
    .groupby("SURROGATE")["HOLDOUT_MAE"]
    .agg(["mean", "std"])
)

print("\nPrediction quality (holdout MAE):")
print(prediction_summary)
