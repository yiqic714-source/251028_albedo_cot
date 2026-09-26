# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, least_squares


# ============================================================
# Settings
# ============================================================

CSV_FILE = (
    "/home/chenyiqi/251028_albedo_cot/project0904/"
    "uniform_fov_product/rsl_fov_202001_west.csv"
)


COT_MID = np.arange(0.5, 76.0, 1.0)


LOG10_B_MIN = -5
LOG10_B_MAX = 5

K_MIN = 0.05
K_MAX = 10


FIG_DIR = "./fig"
os.makedirs(FIG_DIR, exist_ok=True)



# ============================================================
# Model
# ============================================================

def albedo_from_cot(cot, b, k):

    """
    Ac = b*COT^k /(1+b*COT^k)
    """

    cot_k = np.power(cot, k)

    return (
        b * cot_k
        /
        (1.0 + b * cot_k)
    )



# ============================================================
# Read data
# ============================================================


df = pd.read_csv(CSV_FILE)


cot_cols = [
    f"cot_freq_{i}_{i+1}"
    for i in range(76)
]


freq = df[cot_cols].to_numpy(
    dtype=np.float64
)


obs_albedo = df["ret_albedo"].to_numpy(
    dtype=np.float64
)


freq_sum = np.sum(
    freq,
    axis=1
)


valid = (

    np.isfinite(obs_albedo)

    &

    np.all(
        np.isfinite(freq),
        axis=1
    )

    &

    (freq_sum > 0)

    &

    (obs_albedo >= 0)

    &

    (obs_albedo <= 1)

)


freq = freq[valid]

obs_albedo = obs_albedo[valid]


prob = (
    freq
    /
    np.sum(freq, axis=1)[:,None]
)


print(
    "Valid footprints:",
    len(obs_albedo)
)



# ============================================================
# COT standard deviation
# ============================================================


cot_mean_all = (
    prob @ COT_MID
)


# ============================================================
# COT standard deviation
# ============================================================

cot_mean_all = prob @ COT_MID


cot_std_all = np.sqrt(
    np.sum(
        prob *
        (COT_MID[None, :] - cot_mean_all[:, None])**2,
        axis=1
    )
)


# ============================================================
# Select extreme COT variability cases
# ============================================================

p90_std = np.percentile(
    cot_std_all,
    90
)

p10_std = np.percentile(
    cot_std_all,
    10
)


high_var = (
    cot_std_all >= p90_std
)


low_var = (
    cot_std_all <= p10_std
)


print(
    "COT STD 90 percentile:",
    p90_std
)


print(
    "COT STD 10 percentile:",
    p10_std
)


print(
    "High STD samples:",
    np.sum(high_var)
)


print(
    "Low STD samples:",
    np.sum(low_var)
)



# ============================================================
# Optimization
# ============================================================


def unpack_parameters(x):

    log10_b, k = x

    b = 10 ** log10_b

    return b,k



def optimize_model(
        predict_function,
        prob_input,
        obs_input,
        cot_mean_input,
        cot_logmean_input):


    def residual(x):

        b,k = unpack_parameters(x)


        pred = predict_function(
            b,
            k,
            prob_input,
            cot_mean_input,
            cot_logmean_input
        )


        return pred - obs_input



    def objective(x):

        r = residual(x)

        return np.mean(
            r*r
        )



    global_result = differential_evolution(

        objective,

        bounds=[

            (
                LOG10_B_MIN,
                LOG10_B_MAX
            ),

            (
                K_MIN,
                K_MAX
            )

        ],

        seed=42,

        popsize=20,

        maxiter=500,

        tol=1e-10

    )



    ls_result = least_squares(

        residual,

        global_result.x,

        bounds=(

            [
                LOG10_B_MIN,
                K_MIN
            ],

            [
                LOG10_B_MAX,
                K_MAX
            ]

        ),

        max_nfev=10000

    )


    b,k = unpack_parameters(
        ls_result.x
    )


    return b,k



# ============================================================
# Three methods
# ============================================================


def predict_distribution(
        b,
        k,
        prob_input,
        cot_mean_input,
        cot_logmean_input):


    bin_A = albedo_from_cot(
        COT_MID,
        b,
        k
    )


    return (
        prob_input
        @
        bin_A
    )



def predict_mean_cot(
        b,
        k,
        prob_input,
        cot_mean_input,
        cot_logmean_input):


    return albedo_from_cot(
        cot_mean_input,
        b,
        k
    )



def predict_log_mean_cot(
        b,
        k,
        prob_input,
        cot_mean_input,
        cot_logmean_input):


    return albedo_from_cot(
        cot_logmean_input,
        b,
        k
    )



methods = {

    "Distribution":
        predict_distribution,

    "Mean_COT":
        predict_mean_cot,

    "Log_mean_COT":
        predict_log_mean_cot

}



# ============================================================
# Fit function
# ============================================================


def fit_group(mask, name):


    prob_g = prob[mask]

    obs_g = obs_albedo[mask]


    cot_mean_g = (
        prob_g
        @
        COT_MID
    )


    cot_logmean_g = np.exp(

        prob_g
        @
        np.log(COT_MID)

    )


    results={}


    for method,func in methods.items():


        b,k = optimize_model(

            func,

            prob_g,

            obs_g,

            cot_mean_g,

            cot_logmean_g

        )


        results[method]=(b,k)


        print(
            "\n",
            name,
            method
        )

        print(
            "b =",
            b
        )

        print(
            "k =",
            k
        )


    return results



# ============================================================
# Run fitting
# ============================================================


result_high = fit_group(
    high_var,
    "High COT STD"
)


result_low = fit_group(
    low_var,
    "Low COT STD"
)



# ============================================================
# Plot
# ============================================================


cot_plot = np.linspace(
    0.01,
    60,
    500
)


plt.figure(
    figsize=(9,7)
)


line_style = {

    "Distribution":"-",

    "Mean_COT":"--",

    "Log_mean_COT":":"

}



for method in methods:


    b,k = result_high[method]


    plt.plot(

        cot_plot,

        albedo_from_cot(
            cot_plot,
            b,
            k
        ),

        linestyle=line_style[method],

        linewidth=2,

        label=f"High STD - {method}"

    )



for method in methods:


    b,k = result_low[method]


    plt.plot(

        cot_plot,

        albedo_from_cot(
            cot_plot,
            b,
            k
        ),

        linestyle=line_style[method],

        linewidth=2,

        label=f"Low STD - {method}"

    )



plt.xlabel(
    "COT"
)


plt.ylabel(
    "Cloud albedo"
)


plt.xlim(
    0,
    60
)


plt.ylim(
    0,
    1
)


plt.legend(
    fontsize=8
)


plt.tight_layout()


outfile = os.path.join(
    FIG_DIR,
    "Ac_COT_COTstd_high_low.png"
)


plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "\nSaved figure:"
)

print(
    outfile
)