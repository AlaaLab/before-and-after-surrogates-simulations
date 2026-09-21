#######################################################################################
# Author: Franny Dean
# Function: runs the plots after all simulations for tables 
#######################################################################################

import pandas as pd
import numpy as np
import argparse

from PLOTTING import *

parser = argparse.ArgumentParser()

# INPUT FILEPATHS
parser.add_argument("--results_linear_table1",  type=str, default="")
parser.add_argument("--results_nonlinear_table1",  type=str, default="")

parser.add_argument("--results_linear_table2_good", type=str, default="")
parser.add_argument("--results_nonlinear_table2_good", type=str, default="")

parser.add_argument("--results_linear_table2_bad", type=str, default="")
parser.add_argument("--results_nonlinear_table2_bad", type=str, default="")

parser.add_argument("--rho_x1_bad", type=float, default=0.5)
parser.add_argument("--rho_x2_bad", type=str, default=1.0)
parser.add_argument("--rho_x1_good", type=float, default=1.0)
parser.add_argument("--rho_x2_good", type=str, default=1.0)

parser.add_argument("--rho_z_linear", type=float, default=1.01)
parser.add_argument("--rho_z_nonlinear", type=str, default=0.85)

args = parser.parse_args()
# SETTINGS

# ============================================================

RESULTS_LINEAR_TABLE1 = args.results_linear_table1
RESULTS_NONLINEAR_TABLE1 = args.results_nonlinear_table1

RESULTS_LINEAR_TABLE2_GOOD = args.results_linear_table2_good
RESULTS_NONLINEAR_TABLE2_GOOD = args.results_nonlinear_table2_good

RESULTS_LINEAR_TABLE2_BAD = args.results_linear_table2_bad
RESULTS_NONLINEAR_TABLE2_BAD = args.results_nonlinear_table2_bad

RHO_X1_BAD = args.rho_x1_bad
RHO_X2_BAD = args.rho_x2_bad
RHO_X1_GOOD = args.rho_x1_good
RHO_X2_GOOD = args.rho_x2_good

RHO_Z_LINEAR = args.rho_z_linear
RHO_Z_NONLINEAR = args.rho_z_nonlinear

# ============================================================
# TABLE 1
results_linear = pd.read_csv(f'{RESULTS_LINEAR_TABLE1}.csv')
results_linear['DGP'] = 'linear'
results_nonlinear = pd.read_csv(f'{RESULTS_NONLINEAR_TABLE1}.csv')
results_nonlinear['DGP'] = 'non-linear'
results = pd.concat([results_nonlinear, results_linear])

table = make_coverage_bias_table(results)

publication_table = format_table1(table)

publication_table.to_csv(
    "TABLE1.csv"
)

latex = make_latex_table(publication_table)

print(latex)

# TABLE 2 -bad
results_linear = pd.read_csv(f'{RESULTS_LINEAR_TABLE2_BAD}.csv')
results_linear['DGP'] = 'linear'
results_nonlinear = pd.read_csv(f'{RESULTS_NONLINEAR_TABLE2_BAD}.csv')
results_nonlinear['DGP'] = 'non-linear'
results = pd.concat([results_nonlinear, results_linear])

table1 = make_estimate_table2_by_labeled_fraction(
    results,
    rho_x1=RHO_X1_BAD,
    rho_x2=RHO_X2_BAD,
    rho_z=RHO_Z_LINEAR,
)
table2 = make_estimate_table2_by_labeled_fraction(
    results,
    rho_x1=RHO_X1_BAD,
    rho_x2=RHO_X2_BAD,
    rho_z=RHO_Z_NONLINEAR,
)

# SAVE
table2_bad = pd.concat(
    [
        table1["linear"].assign(DGP="Linear"),
        table2["non-linear"].assign(DGP="Non-linear"),
    ]
)
table2_bad.to_csv("TABLE2_bad.csv")

latex_table = combine_linear_and_nonlinear_to_latex(
    table1,
    table2,
)

print(latex_table)

# TABLE 2 - good
results_linear = pd.read_csv(f'{RESULTS_LINEAR_TABLE2_GOOD}.csv')
results_linear['DGP'] = 'linear'
results_nonlinear = pd.read_csv(f'{RESULTS_NONLINEAR_TABLE2_GOOD}.csv')
results_nonlinear['DGP'] = 'non-linear'
results = pd.concat([results_nonlinear, results_linear])

table1 = make_estimate_table2_by_labeled_fraction(
    results,
    rho_x1=RHO_X1_GOOD,
    rho_x2=RHO_X2_GOOD,
    rho_z=RHO_Z_LINEAR,
)
table2 = make_estimate_table2_by_labeled_fraction(
    results,
    rho_x1=RHO_X1_GOOD,
    rho_x2=RHO_X2_GOOD,
    rho_z=RHO_Z_NONLINEAR,
)

# SAVE
table2_good = pd.concat(
    [
        table1["linear"].assign(DGP="Linear"),
        table2["non-linear"].assign(DGP="Non-linear"),
    ]
)
table2_good.to_csv("TABLE2_good.csv")

latex_table = combine_linear_and_nonlinear_to_latex(
    table1,
    table2,
)

print(latex_table)