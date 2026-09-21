#######################################################################################
# Author: Franny Dean
# Function: functions for the synthetic simulation
#######################################################################################

import numpy as np
import pandas as pd
from scipy.special import expit

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

# ============================================================
# TRUE OUTCOME FUNCTION
def outcome_from_z(
    z1, 
    z2,
    noise
):
    """
    Y^t(a) = 1 + Z^t(a) + Z2^t(a) + 0.5 * (Z1^t(a) + Z2^t(a))^2 + epsilon^t
    """
    z_sum = z1 + z2

    return (
        1.0
        + z_sum
        + 0.5 * z_sum**2
        + noise
    )

def generate_outcomes_from_z(
    Z1,
    Z2,
    sigma_y,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        0,
        sigma_y,
        size=Z1.shape,
    )

    return (
        1.0
        + Z1
        + Z2
        + 0.5 * (Z1 + Z2)**2
        + noise
    )

# GENERATE TWO LONGITUDINAL LATENT TRAJECTORIES
def generate_z_trajectory(
    n,
    n_time,
    rho_z,
    method,
    beta,
    treatment_bump,
    t_post,
    confounded=False,
    gamma_c=0.5,
    rng=None,
):
    """
    Generate two longitudinal latent trajectories Z1 and Z2.

    When confounded=False:
        Z1^0 ~ N(0, 1)
        Z2^0 ~ N(0, 1)

    When confounded=True:
        C ~ N(0, 1)

        Z2^0 = gamma_c * C + eps_z2

    Thus:
        C -> Z2 -> Y

    C itself is returned so that it can also be used to
    generate treatment assignment D.
    """

    if rng is None:
        rng = np.random.default_rng()

    if method not in ["linear", "non-linear"]:
        raise ValueError(
            f"Unknown method='{method}'. "
            "Choose from ['linear', 'non-linear']."
        )

    # --------------------------------------------------------
    # Initialize
    Z10 = np.zeros((n, n_time))
    Z20 = np.zeros((n, n_time))
    Z11 = np.zeros((n, n_time))

    # Latent confounder
    if confounded:
        C = rng.normal(0, 1, size=n)

        # Independent residual component
        eps_z2 = rng.normal(0, 1, size=n)

        # C -> Z1
        Z20[:, 0] = (
            gamma_c * C
            + eps_z2
        )

    else:
        C = np.zeros(n)

        Z20[:, 0] = rng.normal(
            0,
            1,
            size=n,
        )

    # Untreated and treated start from same baseline
    Z11[:, 0] = Z10[:, 0]

    # Z1 is independent of C
    Z10[:, 0] = rng.normal(
        0,
        1,
        size=n,
    )

    # --------------------------------------------------------
    # Generate trajectories
    for t in range(1, n_time):

        if method == "linear":

            if t == t_post:
                Z11[:, t] = (
                    rho_z * Z11[:, t - 1]
                    + treatment_bump
                )
            else:
                Z11[:, t] = (
                    rho_z * Z11[:, t - 1]
                )

            Z10[:, t] = (
                rho_z * Z10[:, t - 1]
            )

            Z20[:, t] = (
                rho_z * Z20[:, t - 1]
            )

        elif method == "non-linear":

            if t == t_post:
                Z11[:, t] = (
                    rho_z * Z11[:, t - 1]
                    + beta * np.tanh(Z11[:, t - 1])
                    + treatment_bump
                )
            else:
                Z11[:, t] = (
                    rho_z * Z11[:, t - 1]
                    + beta * np.tanh(Z11[:, t - 1])
                )

            Z10[:, t] = (
                rho_z * Z10[:, t - 1]
                + beta * np.tanh(Z10[:, t - 1])
            )

            Z20[:, t] = (
                rho_z * Z20[:, t - 1]
                + beta * np.tanh(Z20[:, t - 1])
            )

    return Z10, Z20, Z11, Z20, C

# GENERATE LONGITUDINAL X
def generate_x_from_z(
    Z1,
    Z2,
    rho_x1,
    rho_x2,
    rng=None,
):
    """
    Generate a two-dimensional surrogate:

        X1^t = rho_X1 * Z1^t
               + sqrt(1 - rho_X1^2) * epsilon1^t

        X2^t = rho_X2 * Z2^t
               + sqrt(1 - rho_X2^2) * epsilon2^t

    X1 measures Z1.
    X2 measures Z2.
    """

    if rng is None:
        rng = np.random.default_rng()

    epsilon1 = rng.normal(
        size=Z1.shape
    )

    epsilon2 = rng.normal(
        size=Z2.shape
    )

    X1 = (
        rho_x1 * Z1
        + np.sqrt(1 - rho_x1**2) * epsilon1
    )

    X2 = (
        rho_x2 * Z2
        + np.sqrt(1 - rho_x2**2) * epsilon2
    )

    return X1, X2

# GENERATE TRAINING DATA
def generate_training_data(
    n_train,
    time_points,
    train_k,
    rho_x1,
    rho_x2,
    rho_z,
    sigma_y,
    z_method,
    beta,
    t_post,
    gamma_c=0.5,
    confounded=False,
    rng=None,
):
    """
    Training data consist entirely of untreated longitudinal data.

    For each trajectory we generate:

        Z1^0, ..., Z1^T
        Z2^0, ..., Z2^T
        X1^0, ..., X1^T
        X2^0, ..., X2^T
        Y^0(0), ..., Y^T(0)

    where:

        X1 measures Z1 with quality rho_x1
        X2 measures Z2 with quality rho_x2

    Training pairs are:

        (X1^t, X2^t, k) -> Y^{t+k}(0)

    for all available prediction intervals k.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_time = len(time_points)

    # Latent longitudinal trajectories
    Z10, Z20, _, _, _ = generate_z_trajectory(
        n=n_train,
        n_time=n_time,
        rho_z=rho_z,
        method=z_method,
        beta=beta,
        treatment_bump=0,
        t_post=t_post,
        rng=rng,
        confounded=confounded,
        gamma_c=gamma_c,
    )

    # Longitudinal surrogate
    X1, X2 = generate_x_from_z(
        Z10,
        Z20,
        rho_x1=rho_x1,
        rho_x2=rho_x2,
        rng=rng,
    )

    # Longitudinal untreated outcomes
    Y0 = generate_outcomes_from_z(
        Z10,
        Z20,
        sigma_y=sigma_y,
        rng=rng,
    )

    # Construct prediction training examples
    X1_train = []
    X2_train = []
    K_train = []
    Y_train = []

    for k in train_k:

        # Need t + k to exist
        for t in range(n_time - k):

            X1_train.append(X1[:, t])
            X2_train.append(X2[:, t])

            K_train.append(
                np.full(n_train, k)
            )

            Y_train.append(
                Y0[:, t + k]
            )

    X1_train = np.concatenate(X1_train)
    X2_train = np.concatenate(X2_train)
    K_train = np.concatenate(K_train)
    Y_train = np.concatenate(Y_train)

    train_df = pd.DataFrame({
        "X1": X1_train,
        "X2": X2_train,
        "K": K_train,
        "Y": Y_train,
    })

    return {
        "Z1": Z10,
        "Z2": Z20,
        "X1": X1,
        "X2": X2,
        "Y0": Y0,
        "data": train_df,
    }

# FIT MU(X1, X2, K)
def fit_prediction_model(
    training_data,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
):
    """
    Fit:

        mu_hat(X1, X2, k)
            ~= E[Y^{t+k}(0) | X1^t, X2^t]

    using a random forest.
    """

    train_df = training_data["data"]

    X_model = train_df[
        ["X1", "X2", "K"]
    ].values

    y_model = train_df["Y"].values

    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )

    model.fit(
        X_model,
        y_model,
    )

    return model


# PREDICT Y USING X1, X2 AND INTERVAL K
def predict_mu(
    model,
    X1,
    X2,
    k,
):
    """
    Apply:

        mu_hat(X1, X2, k)
    """

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim != 1:
        X1 = X1.reshape(-1)

    if X2.ndim != 1:
        X2 = X2.reshape(-1)

    if len(X1) != len(X2):
        raise ValueError(
            f"X1 and X2 must have the same length. "
            f"Got {len(X1)} and {len(X2)}."
        )

    K = np.full(
        len(X1),
        k,
    )

    model_input = np.column_stack([
        X1,
        X2,
        K,
    ])

    return model.predict(model_input)


# GENERATE TREATED / OBSERVATIONAL DATA
def generate_observational_data(
    n_obs,
    rho_x1,
    rho_x2,
    t_pre,
    t_post,
    t_end,
    rho_z,
    treatment_bump,
    sigma_y,
    p_treat,
    z_method,
    beta,
    confounded=False,
    gamma_c = 0.5,
    rng=None,
):
    """
    Generate longitudinal observational/treatment data.

    Treatment assignment is randomized

        D ~ Bernoulli(p_treat)
         
    OR its confounded by C.
        logit_p = (np.log(p_treat / (1 - p_treat)) + gamma_C * C)
        p_D = expit(logit_p)
        D  ~ Bernoulli(p_treat)

    Treatment changes ONLY the latent Z2 trajectory.

    Before treatment:

        Z1^t(1) = Z1^t(0)
        Z2^t(1) = Z2^t(0)

    After treatment:

        Z1^t^post(1) = Z1^t^post(0) + treatment_bump
        Z2^t(1) = Z2^t(0)

    Surrogates are generated separately:

        X1 = rho_x1 * Z1 + measurement noise
        X2 = rho_x2 * Z2 + measurement noise

    Y(a) is generated from the corresponding latent trajectory.

    The same outcome noise is used for Y0 and Y1, so the
    causal effect comes entirely through the treatment-induced
    change in Z1.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_time = t_end + 1

    # Untreated latent trajectory
    Z10, Z20, Z11, Z21, C = generate_z_trajectory(
        n=n_obs,
        n_time=n_time,
        rho_z=rho_z,
        method=z_method,
        beta=beta,
        treatment_bump=treatment_bump,
        t_post=t_post,
        rng=rng,
        confounded=confounded,
        gamma_c=gamma_c
    )

    # Treatment assignment
    if confounded:

        logit_p = (
            np.log(p_treat / (1 - p_treat))
            + gamma_c * C
        )

        p_D = expit(logit_p)

        D = rng.binomial(
            1,
            p_D,
            size=n_obs,
        )
    else:
        D = rng.binomial(
            1,
            p_treat,
            size=n_obs,
        )

    # Surrogate potential trajectories
    X10, X20 = generate_x_from_z(
        Z10,
        Z20,
        rho_x1=rho_x1,
        rho_x2=rho_x2,
        rng=rng,
    )

    X11, X21 = generate_x_from_z(
        Z11,
        Z21,
        rho_x1=rho_x1,
        rho_x2=rho_x2,
        rng=rng,
    )

    # Outcome potential trajectories:

    # Shared noise means:
    #
    # Y1 - Y0
    #
    # is entirely due to treatment changing Z1.

    epsilon = rng.normal(
        0,
        sigma_y,
        size=Z10.shape,
    )

    Y0 = outcome_from_z(
        Z10,
        Z20,
        epsilon,
    )

    Y1 = outcome_from_z(
        Z11,
        Z21,
        epsilon,
    )

    # Observed surrogate measurements:

    # Everyone is untreated before treatment
    X1_pre = X10[:, t_pre]
    X2_pre = X20[:, t_pre]

    # After treatment:
    #
    # treated -> treated potential surrogate
    # untreated -> untreated potential surrogate

    X1_post = np.where(
        D == 1,
        X11[:, t_post],
        X10[:, t_post],
    )

    X2_post = np.where(
        D == 1,
        X21[:, t_post],
        X20[:, t_post],
    )

    # Observed endpoint outcome
    Y_end_observed = np.where(
        D == 1,
        Y1[:, t_end],
        Y0[:, t_end],
    )

    return {
        "D": D,
        "C": C,

        # Latent potential trajectories
        "Z10": Z10,
        "Z11": Z11,
        "Z20": Z20,
        "Z21": Z21,

        # Surrogate potential trajectories
        "X10": X10,
        "X11": X11,
        "X20": X20,
        "X21": X21,

        # Outcome potential trajectories
        "Y0": Y0,
        "Y1": Y1,

        # Key observed quantities
        "X1_pre": X1_pre,
        "X2_pre": X2_pre,

        "X1_post": X1_post,
        "X2_post": X2_post,

        "Y_end_observed": Y_end_observed,

        # Time information
        "t_pre": t_pre,
        "t_post": t_post,
        "t_end": t_end,

        "k_pre_end": t_end - t_pre,
        "k_post_end": t_end - t_post,
    }


# TRUE ATT
def true_att(obs):
    """
    True ATT among treated individuals:

        E[Y(1) - Y(0) | D=1]
    """

    treated = obs["D"] == 1

    true_effect = (
        obs["Y1"][treated, obs["t_end"]]
        -
        obs["Y0"][treated, obs["t_end"]]
    )

    return np.mean(true_effect)

# NAIVE BEFORE-AND-AFTER SURROGATE
def naive_unlabeled_att(
    obs,
    labeled_idx,
    model,
):
    """
    Naive estimator using predicted treated and untreated
    endpoint outcomes.

        ATT = mean[
            Y_hat(1) - Y_hat(0)
        ]
    """
    treated_idx = np.where(obs["D"] == 1)[0]

    # We plot all data ATT but the PPI estimate uses independent samples (hence unlabeled for prediction ATT).
    unlabeled_idx = np.setdiff1d(
        treated_idx,
        labeled_idx,
    )

    X1_post = obs["X1_post"][treated_idx] # [unlabeled_idx]
    X2_post = obs["X2_post"][treated_idx] # [unlabeled_idx]

    Y1_hat = predict_mu(
        model,
        X1_post,
        X2_post,
        obs["k_post_end"],
    )

    X1_pre = obs["X1_pre"][treated_idx] # [unlabeled_idx]
    X2_pre = obs["X2_pre"][treated_idx] # [unlabeled_idx]

    Y0_hat = predict_mu(
        model,
        X1_pre,
        X2_pre,
        obs["k_pre_end"],
    )

    # Individual predicted treatment effects
    ite_hat = Y1_hat - Y0_hat

    n = len(ite_hat)

    att = np.mean(ite_hat)

    se = (
        np.std(ite_hat, ddof=1)
        / np.sqrt(n)
    )

    return {
        "att": att,
        "se": se,
        "n": n,
    }

# COVARIATE SHIFT WEIGHTS
def estimate_covariate_shift_weights(
    X_source,
    X_target,
    clip_min=0.05,
    clip_max=20.0,
    random_state=42
):
    """
    Estimate density-ratio weights:

        w(x) = p_target(x) / p_source(x)

    using logistic regression.

    X_source and X_target can be:

        shape (n,)
    
    or:

        shape (n, p)

    For the current simulation p = 2:

        X = [X1, X2]
    """

    X_source = np.asarray(X_source)
    X_target = np.asarray(X_target)

    # Convert 1D inputs to (n, 1), but preserve
    # multidimensional inputs.
    if X_source.ndim == 1:
        X_source = X_source.reshape(-1, 1)

    if X_target.ndim == 1:
        X_target = X_target.reshape(-1, 1)

    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError(
            "X_source and X_target must be 1D or 2D arrays."
        )

    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError(
            "X_source and X_target must have the same "
            f"number of features. Got "
            f"{X_source.shape[1]} and {X_target.shape[1]}."
        )

    # Source = 0
    # Target = 1
    X_combined = np.vstack([
        X_source,
        X_target,
    ])

    domain = np.concatenate([
        np.zeros(len(X_source)),
        np.ones(len(X_target)),
    ])

    # Fit classifier distinguishing target from source
    domain_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    domain_model.fit(
        X_combined,
        domain,
    )

    # P(target | X) for source observations
    p_target_given_x = domain_model.predict_proba(
        X_source
    )[:, 1]

    # Density ratio:
    #
    # p_target(x) / p_source(x)
    #
    # = [P(target|x) / P(source|x)]
    #   * [P(source) / P(target)]

    n_source = len(X_source)
    n_target = len(X_target)

    prior_source = (
        n_source
        /
        (n_source + n_target)
    )

    prior_target = (
        n_target
        /
        (n_source + n_target)
    )

    eps = 1e-8

    weights = (
        p_target_given_x
        /
        np.clip(
            1.0 - p_target_given_x,
            eps,
            1.0,
        )
    ) * (
        prior_source
        /
        prior_target
    )

    # Stabilize extreme weights
    weights = np.clip(
        weights,
        clip_min,
        clip_max,
    )

    # Normalize to mean 1
    weights = (
        weights
        /
        np.mean(weights)
    )

    return weights

# PPI ESTIMATOR + INDIVIDUAL DIAGNOSTICS
def ppi_att(
    obs,
    labeled_idx,
    model,
    training_holdout,
    rng=None,
    use_covariate_shift=False,
):
    """
    PPI estimator with individual-level prediction-error
    diagnostics.

    Surrogate is two-dimensional:

        X = (X1, X2)

    Standard:

        PRE residual =
            mean(Y0_control - Y0_hat_control)

    Covariate shifted:

        PRE residual =
            weighted mean(
                Y0_control - Y0_hat_control
            )

    where weights reweight the control distribution of
    (X1_pre, X2_pre) toward the treated distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(obs["D"])

    # INDICES
    treated_idx = np.where(
        obs["D"] == 1
    )[0]

    control_pool = np.where(obs["D"] == 0)[0]

    control_idx = rng.choice(
        control_pool,
        size=min(len(labeled_idx), len(control_pool)),
        replace=False,
    )

    unlabeled_treated_idx = np.setdiff1d(
        treated_idx,
        labeled_idx,
    )

    # PREDICTIONS FOR ALL TREATED
    X1_pre_treated = obs["X1_pre"][treated_idx]
    X2_pre_treated = obs["X2_pre"][treated_idx]

    X1_post_treated = obs["X1_post"][treated_idx]
    X2_post_treated = obs["X2_post"][treated_idx]

    Y0_hat_treated = predict_mu(
        model,
        X1_pre_treated,
        X2_pre_treated,
        obs["k_pre_end"],
    )

    Y1_hat_treated = predict_mu(
        model,
        X1_post_treated,
        X2_post_treated,
        obs["k_post_end"],
    )

    Y0_true_treated = obs["Y0"][
        treated_idx,
        obs["t_end"]
    ]

    Y1_true_treated = obs["Y1"][
        treated_idx,
        obs["t_end"]
    ]

    # INDIVIDUAL PREDICTION ERRORS
    pre_errors = (
        Y0_true_treated
        -
        Y0_hat_treated
    )

    post_errors = (
        Y1_true_treated
        -
        Y1_hat_treated
    )

    # PREDICTED ATT COMPONENT
    if len(unlabeled_treated_idx) > 0:

        X1_pre_unlabeled = obs["X1_pre"][
            unlabeled_treated_idx
        ]

        X2_pre_unlabeled = obs["X2_pre"][
            unlabeled_treated_idx
        ]

        X1_post_unlabeled = obs["X1_post"][
            unlabeled_treated_idx
        ]

        X2_post_unlabeled = obs["X2_post"][
            unlabeled_treated_idx
        ]

        Y0_hat_unlabeled = predict_mu(
            model,
            X1_pre_unlabeled,
            X2_pre_unlabeled,
            obs["k_pre_end"],
        )

        Y1_hat_unlabeled = predict_mu(
            model,
            X1_post_unlabeled,
            X2_post_unlabeled,
            obs["k_post_end"],
        )

        delta_hat = (
            Y1_hat_unlabeled
            -
            Y0_hat_unlabeled
        )

        predicted_component = np.mean(
            delta_hat
        )

        n_predicted = len(delta_hat)

        if n_predicted > 1:
            se_predicted = (
                np.std(
                    delta_hat,
                    ddof=1,
                )
                /
                np.sqrt(n_predicted)
            )
        else:
            se_predicted = 0.0

    else:

        predicted_component = 0.0
        se_predicted = 0.0
        n_predicted = 0

    # LABELED POST-TREATMENT RECTIFIER
    X1_post_labeled = obs["X1_post"][
        labeled_idx
    ]

    X2_post_labeled = obs["X2_post"][
        labeled_idx
    ]

    Y1_labeled = obs["Y1"][
        labeled_idx,
        obs["t_end"]
    ]

    Y1_hat_labeled = predict_mu(
        model,
        X1_post_labeled,
        X2_post_labeled,
        obs["k_post_end"],
    )

    post_residuals = (
        Y1_labeled
        -
        Y1_hat_labeled
    )

    post_residual = np.mean(
        post_residuals
    )

    n_post = len(post_residuals)

    if n_post > 1:
        se_post = (
            np.std(
                post_residuals,
                ddof=1,
            )
            /
            np.sqrt(n_post)
        )
    else:
        se_post = 0.0

    # CONTROL / PRE RESIDUAL
    X1_pre_control = obs["X1_pre"][
        control_idx
    ]

    X2_pre_control = obs["X2_pre"][
        control_idx
    ]

    Y0_control = obs["Y0"][
        control_idx,
        obs["t_end"]
    ]

    Y0_hat_control = predict_mu(
        model,
        X1_pre_control,
        X2_pre_control,
        obs["k_pre_end"],
    )

    control_residuals = (
        Y0_control
        -
        Y0_hat_control
    )

    n_control = len(
        control_residuals
    )

    # STANDARD CONTROL RECTIFIER
    if not use_covariate_shift:

        control_weights = np.ones(
            n_control
        )

        control_residual = np.mean(
            control_residuals
        )

        if n_control > 1:
            se_control = (
                np.std(
                    control_residuals,
                    ddof=1,
                )
                /
                np.sqrt(n_control)
            )
        else:
            se_control = 0.0

    # COVARIATE-SHIFTED CONTROL RECTIFIER
    else:

        # Source = controls
        X_source = np.column_stack([
            X1_pre_control,
            X2_pre_control,
        ])

        # Target = treated
        X_target = np.column_stack([
            obs["X1_pre"][treated_idx],
            obs["X2_pre"][treated_idx],
        ])

        control_weights = (
            estimate_covariate_shift_weights(
                X_source=X_source,
                X_target=X_target,
            )
        )

        control_residual = np.average(
            control_residuals,
            weights=control_weights,
        )

        n_eff_control = (
            np.sum(control_weights) ** 2
            /
            np.sum(control_weights ** 2)
        )

        weighted_variance = (
            np.sum(
                control_weights
                *
                (
                    control_residuals
                    -
                    control_residual
                ) ** 2
            )
            /
            np.sum(control_weights)
        )

        se_control = np.sqrt(
            weighted_variance
            /
            n_eff_control
        )

    # PPI
    ppi = (
        predicted_component
        +
        post_residual
        -
        control_residual
    )

    se_ppi = np.sqrt(
        se_predicted**2
        +
        se_post**2
        +
        se_control**2
    )

    # INDIVIDUAL DIAGNOSTICS
    labeled_lookup = set(
        labeled_idx
    )

    labeled_indicator = np.array([
        idx in labeled_lookup
        for idx in treated_idx
    ])

    diagnostics = pd.DataFrame({

        "OBS_INDEX":
            treated_idx,

        "LABELED":
            labeled_indicator,

        "Y0_TRUE":
            Y0_true_treated,

        "Y0_PRED":
            Y0_hat_treated,

        "Y1_TRUE":
            Y1_true_treated,

        "Y1_PRED":
            Y1_hat_treated,

        "PRE_ERROR":
            pre_errors,

        "POST_ERROR":
            post_errors,

        "ABS_PRE_ERROR":
            np.abs(pre_errors),

        "ABS_POST_ERROR":
            np.abs(post_errors),

        "POST_RECTIFIER":
            np.where(
                labeled_indicator,
                post_errors,
                np.nan,
            ),

        "ABS_POST_RECTIFIER":
            np.where(
                labeled_indicator,
                np.abs(post_errors),
                np.nan,
            ),

        "CONTROL_RECTIFIER":
            control_residual,

        "ABS_CONTROL_RECTIFIER":
            abs(control_residual),

        "CONTROL_SHIFTED":
            use_covariate_shift,
    })

    diagnostics["PRE_ERROR_SQUARED"] = (
        diagnostics["PRE_ERROR"] ** 2
    )

    diagnostics["POST_ERROR_SQUARED"] = (
        diagnostics["POST_ERROR"] ** 2
    )

    return {
        "ppi": ppi,
        "se": se_ppi,

        "n_predicted": n_predicted,
        "n_post": n_post,
        "n_control": n_control,

        "se_predicted": se_predicted,
        "se_post": se_post,
        "se_control": se_control,

        "post_rectifier": post_residual,
        "abs_post_rectifier": abs(post_residual),

        "control_rectifier": control_residual,
        "abs_control_rectifier": abs(control_residual),

        "use_covariate_shift": use_covariate_shift,

        "mean_control_weight":
            np.mean(control_weights),

        "sd_control_weight":
            np.std(control_weights),

        "max_control_weight":
            np.max(control_weights),

        "min_control_weight":
            np.min(control_weights),

        "ess_control":
            (
                np.sum(control_weights) ** 2
                /
                np.sum(control_weights ** 2)
            ),

        "diagnostics": diagnostics,
    }

# DIFFERENCE-IN-MEANS ATE
def dim_ate(
    obs,
    labeled_idx,
    rng=None
):
    """
    Difference-in-means estimator.

    Uses:
        labeled treated endpoint outcomes
        vs.
        all untreated endpoint outcomes

    ATT_DIM =
        mean[Y(1) | labeled treated]
        -
        mean[Y(0) | controls]
    """

    if rng is None:
        rng = np.random.default_rng()

    treated_idx = labeled_idx

    control_pool = np.where(obs["D"] == 0)[0]

    control_idx = rng.choice(
        control_pool,
        size=min(len(labeled_idx), len(control_pool)),
        replace=False,
    )

    Y_treated = obs["Y1"][
        treated_idx,
        obs["t_end"]
    ]

    Y_control = obs["Y0"][
        control_idx,
        obs["t_end"]
    ]

    att = (
        np.mean(Y_treated)
        -
        np.mean(Y_control)
    )

    se_treated = (
        np.std(Y_treated, ddof=1)
        /
        np.sqrt(len(Y_treated))
    )

    se_control = (
        np.std(Y_control, ddof=1)
        /
        np.sqrt(len(Y_control))
    )

    se = np.sqrt(
        se_treated**2
        +
        se_control**2
    )

    return {
        "att": att,
        "se": se,
        "n_treated": len(Y_treated),
        "n_control": len(Y_control),
    }

# CREATE TRAINING HOLDOUT
def make_training_holdout(
    training_data,
    n_holdout=4000,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()

    df = training_data["data"]

    n_holdout = min(
        n_holdout,
        len(df),
    )

    holdout_idx = rng.choice(
        len(df),
        size=n_holdout,
        replace=False,
    )

    holdout_mask = np.zeros(
        len(df),
        dtype=bool,
    )

    holdout_mask[holdout_idx] = True

    holdout_df = df.iloc[
        holdout_idx
    ].copy()

    train_df = df.iloc[
        ~holdout_mask
    ].copy()

    training_for_model = {
        **training_data,
        "data": train_df,
    }

    train_holdout = {
        "X1": holdout_df["X1"].values,
        "X2": holdout_df["X2"].values,
        "K": holdout_df["K"].values,
        "Y": holdout_df["Y"].values,
    }

    return (
        training_for_model,
        train_holdout,
    )
