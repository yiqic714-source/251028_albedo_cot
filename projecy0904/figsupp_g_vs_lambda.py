import os
import miepython
import numpy as np
import matplotlib.pyplot as plt

# Parameters
diameters = [7.0, 10.0, 13.0]  # droplet diameters in micrometers
m_real = 1.33                  # real part of refractive index of water
m_imag = 0.0                   # imaginary part is approximately zero in the visible/NIR
m = complex(m_real, m_imag)

wavelengths = np.arange(0.01, 6.0, 0.01)

plt.figure(figsize=(5, 4))

for d in diameters:
    g_values = []

    for lam in wavelengths:
        x = (np.pi * d) / lam
        qext, qsca, qback, g = miepython.efficiencies_mx(m, x)
        g_values.append(g)

    plt.plot(wavelengths, g_values, alpha=0.75, linewidth=1.2, label=f'd = {d:.0f} µm')

boundary_lines = [
    (0.3, 'gray'),
    (0.4, 'tab:red'),
    (0.7, 'tab:red'),
    (5.0, 'gray'),
]

for xline, color in boundary_lines:
    plt.axvline(xline, color=color, linestyle='--', linewidth=1, alpha=0.85)
    if xline in [0.3]:
        plt.text(
            xline,
            -0.01,
            f'{xline:.1f}',
            rotation=90,
            ha='right',
            va='top',
            color=color,
            fontsize=9,
            transform=plt.gca().get_xaxis_transform()
        )
    if xline in [0.4, 0.7]:
        plt.text(
            xline,
            -0.01,
            f'{xline:.1f}',
            rotation=90,
            ha='left',
            va='top',
            color=color,
            fontsize=9,
            transform=plt.gca().get_xaxis_transform()
        )

plt.xlim([0, 6])
plt.xlabel(r'$\lambda$ (µm)', fontsize=13)
plt.ylabel(r'$g$', fontsize=13)
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)

os.makedirs('figs', exist_ok=True)
plt.tight_layout()
plt.savefig('figs/figsupp_g_vs_lambda.png', dpi=300, bbox_inches='tight')
plt.show()