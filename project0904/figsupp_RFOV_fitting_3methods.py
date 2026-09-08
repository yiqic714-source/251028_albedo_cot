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
    "RFOV_product/rsl_fov_202001_west.csv"
)

COT_MID = np.arange(0.5, 60.0, 1.0)

LOG10_B_MIN = -5
LOG10_B_MAX = 5

K_MIN = 0.05
K_MAX = 10


FIG_DIR = "./figs"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Model
# ============================================================

def albedo_from_cot(cot, b, k):
    """
    Ac = b*COT^k / (1+b*COT^k)
    """
    cot_k = np.power(cot, k)

    return (b * cot_k) / (1.0 + b * cot_k)



# ============================================================
# Read data
# ============================================================

df = pd.read_csv(CSV_FILE)


cot_cols = [
    f"cot_freq_{i}_{i+1}"
    for i in range(60)
]


freq = df[cot_cols].to_numpy(dtype=np.float64)

obs_albedo = df["ret_albedo"].to_numpy(dtype=np.float64)


freq_sum = np.sum(freq, axis=1)


valid = (
    np.isfinite(obs_albedo)
    &
    np.all(np.isfinite(freq), axis=1)
    &
    (freq_sum > 0)
    &
    (obs_albedo >= 0)
    &
    (obs_albedo <= 1)
)


freq = freq[valid]
obs_albedo = obs_albedo[valid]


prob = freq / np.sum(freq, axis=1)[:, None]


print("Valid footprints:", len(obs_albedo))


# ============================================================
# Three effective COT representations
# ============================================================


# 方法2：算术平均 COT

cot_mean = prob @ COT_MID



# 方法3：log mean COT

log_cot_mean = np.exp(
    prob @ np.log(COT_MID)
)



# ============================================================
# Optimization framework
# ============================================================


def unpack(x):

    log10_b, k = x

    b = 10 ** log10_b

    return b, k



def optimize_model(predict_function):


    def residual(x):

        b, k = unpack(x)

        pred = predict_function(
            b,
            k
        )

        return pred - obs_albedo


    def mse(x):

        r = residual(x)

        return np.mean(r*r)


    result_global = differential_evolution(
        mse,
        [
            (LOG10_B_MIN, LOG10_B_MAX),
            (K_MIN, K_MAX)
        ],
        seed=42,
        popsize=20,
        maxiter=500,
        tol=1e-10,
    )


    result_ls = least_squares(
        residual,
        result_global.x,
        bounds=(
            [LOG10_B_MIN,K_MIN],
            [LOG10_B_MAX,K_MAX]
        ),
        max_nfev=10000
    )


    b,k = unpack(result_ls.x)

    pred = predict_function(
        b,
        k
    )


    rmse=np.sqrt(
        np.mean((pred-obs_albedo)**2)
    )


    return b,k,rmse



# ============================================================
# Method 1
# Full COT distribution
# ============================================================


def predict_distribution(b,k):

    bin_A = albedo_from_cot(
        COT_MID,
        b,
        k
    )

    return prob @ bin_A



b1,k1,rmse1 = optimize_model(
    predict_distribution
)



# ============================================================
# Method 2
# Arithmetic mean COT
# ============================================================


def predict_mean_cot(b,k):

    return albedo_from_cot(
        cot_mean,
        b,
        k
    )


b2,k2,rmse2 = optimize_model(
    predict_mean_cot
)



# ============================================================
# Method 3
# Log mean COT
# ============================================================


def predict_log_cot(b,k):

    return albedo_from_cot(
        log_cot_mean,
        b,
        k
    )


b3,k3,rmse3 = optimize_model(
    predict_log_cot
)



# ============================================================
# Print results
# ============================================================

print("\n============================")
print("Method 1: COT Distribution")
print("============================")
print("b =",b1)
print("k =",k1)
print("RMSE =",rmse1)



print("\n============================")
print("Method 2: Mean COT")
print("============================")
print("b =",b2)
print("k =",k2)
print("RMSE =",rmse2)



print("\n============================")
print("Method 3: Log mean COT")
print("============================")
print("b =",b3)
print("k =",k3)
print("RMSE =",rmse3)



# ============================================================
# Plot Ac-COT curves
# ============================================================

cot_plot = np.linspace(
    0.01,
    80,
    500
)


A1 = albedo_from_cot(
    cot_plot,
    b1,
    k1
)

A2 = albedo_from_cot(
    cot_plot,
    b2,
    k2
)

A3 = albedo_from_cot(
    cot_plot,
    b3,
    k3
)



plt.figure(figsize=(5, 4.1))


plt.plot(
    cot_plot,
    A1, lw=1.5,
    label=f"COT Distribution\nb={b1:.3g}, k={k1:.3f}"
)

plt.plot(
    cot_plot,
    A3, ':', lw=1.8,
    label=f"Log mean COT\nb={b3:.3g}, k={k3:.3f}"
)

plt.plot(
    cot_plot,
    A2, '--', lw=1.5,
    label=f"Mean COT\nb={b2:.3g}, k={k2:.3f}"
)


plt.xlabel("COT", fontsize=14)
plt.ylabel(r'$A_{\mathrm{c}}$', fontsize=14)
plt.grid(alpha=0.25)

plt.xlim(0,60)
plt.ylim(0,0.7)

plt.legend()

plt.tight_layout()


outfile=os.path.join(
    FIG_DIR,
    "figsupp_3_mean_methods.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved figure:")
print(outfile)