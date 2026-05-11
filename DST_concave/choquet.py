import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from env import get_true_reference_pf

true_pf = get_true_reference_pf()
pareto  = np.array(true_pf)

MU12 = 1.0

def choquet_explorer(mu1, mu2):
    a_cost = -pareto[:, 0]
    b_cost = -pareto[:, 1]
    C = np.where(a_cost <= b_cost,
                 a_cost + (b_cost - a_cost) * mu1,
                 b_cost + (a_cost - b_cost) * mu2)
    idx = np.argmin(C)
    return pareto[idx, 0], pareto[idx, 1], C[idx]

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.20)

ax.scatter(pareto[:, 0], pareto[:, 1],
           s=25, color='#378ADD', zorder=3, label='True Pareto front')
sol, = ax.plot([], [], 'o', color='#D85A30', ms=12, zorder=5)
info = ax.text(0.02, 0.04, '', transform=ax.transAxes, fontsize=9,
               va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#cccccc'))
ax.set_xlabel('Time Cost')
ax.set_ylabel('Treasure')
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

ax_mu1 = plt.axes([0.15, 0.10, 0.70, 0.03])
ax_mu2 = plt.axes([0.15, 0.04, 0.70, 0.03])
sl_mu1 = Slider(ax_mu1, 'μ₁ (speed)',    0.0, 1.0, valinit=0.5, valstep=0.01)
sl_mu2 = Slider(ax_mu2, 'μ₂ (treasure)', 0.0, 1.0, valinit=0.5, valstep=0.01)

def update(val):
    mu1 = sl_mu1.val
    mu2 = sl_mu2.val
    sf1, sf2, cmin = choquet_explorer(mu1, mu2)
    sol.set_data([sf1], [sf2])
    sol.set_label(f'Choquet (time={sf1:.0f}, treasure={sf2:.0f})')
    ax.set_title(f'Choquet on DST Concave  [μ₁₂=1.0]\nμ₁={mu1:.2f}  μ₂={mu2:.2f}', fontweight='bold')
    info.set_text(f'μ₁={mu1:.2f}  μ₂={mu2:.2f}  μ₁₂={MU12}\nChoquet min={cmin:.4f}')
    ax.legend(loc='upper right', fontsize=9)
    fig.canvas.draw_idle()

sl_mu1.on_changed(update)
sl_mu2.on_changed(update)
update(None)

plt.show()