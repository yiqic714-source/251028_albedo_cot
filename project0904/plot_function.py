import os
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# Create output directory
# ==========================
out_dir = "./figs"
os.makedirs(out_dir, exist_ok=True)


# ==========================
# COT range
# ==========================
COT = np.linspace(2, 60, 500)

# cloud fraction
f = 0.5


# ==========================
# Ac = 0.13 * COT/(1+0.13*COT)
# ==========================
Ac_simple = 0.13 * COT / (1 + 0.13 * COT)


# ==========================
# Ac=(a3+a4*f*COT)^a6
# ==========================
params = {
    "NPO": [0.00163, 0.0052, 0.337],
    "NAO": [0.00101, 0.0052, 0.325],
    "TPO": [0.00017, 0.0064, 0.405],
    "TAO": [0.00027, 0.0071, 0.423],
    "TIO": [0.00016, 0.0069, 0.425],
    "SPO": [0.00013, 0.0062, 0.342],
    "SAO": [0.00024, 0.0057, 0.333],
    "SIO": [0.00028, 0.0053, 0.324],
}


# calculate
Ac_region = {}

for region, (a3, a4, a6) in params.items():
    Ac_region[region] = (a3 + a4 * f * COT) ** a6


# ==========================
# Plot
# ==========================
plt.figure(figsize=(8, 6), dpi=300)


# simple relationship
plt.plot(
    COT,
    Ac_simple,
    color="black",
    linewidth=2.5,
    label=r"$A_c=0.13COT/(1+0.13COT)$"
)


# nonlinear curves
for region, Ac in Ac_region.items():
    plt.plot(
        COT,
        Ac,
        linewidth=1.8,
        label=region,
        alpha=0.7
    )


plt.xlabel("Cloud optical thickness (COT)", fontsize=13)
plt.ylabel("Cloud albedo ($A_c$)", fontsize=13)

plt.xlim(0, 60)
plt.ylim(0, 1)

plt.grid(alpha=0.3)

plt.legend(
    fontsize=9,
    frameon=False,
    ncol=2
)

plt.tight_layout()


# ==========================
# Save figure
# ==========================
fig_path = os.path.join(out_dir, "Ac_COT_comparison_f0p5.png")

plt.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure saved to: {fig_path}")