import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-4, 4, 1000)
y1 = np.cos(x + np.pi / 6)
y2 = np.cos(x - np.pi / 6)
y3 = np.cos(x + 5 * np.pi / 6)

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(x, y1)
ax.plot(x, y2)
ax.plot(x, y3)

ax.set_xlim(-4, 4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(-1.3, 1.3)

for spine in ax.spines.values():
	spine.set_visible(False)

ax.annotate('', xy=(4, 0), xytext=(-4, 0), arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0))
ax.annotate('', xy=(0, 1.3), xytext=(0, -1.3), arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0))

fig.savefig('/home/chenyiqi/260320_ship_emission/test.png', bbox_inches='tight')
plt.close(fig)
