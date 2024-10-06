import matplotlib.pyplot as plt
import numpy as np

import utils

p = np.array([10, 10, 10]) # point
O = np.zeros(3) # origin
P_1 = np.array([3, 0, 5]) # l_1 intercept
v = np.array([1, -2, 0]) # l_1 vector

fig, ax = utils.create_3d_plot()

utils.plot_line(ax, P_1, v, label="l_1")
utils.plot_point(ax, p, label="p")
utils.plot_vector(ax, P_1, v, color="g", label="v")
utils.plot_vector(ax, O, p, color="m", label="Op")
utils.plot_vector(ax, P_1, p, color="c", label="(P_1)p")


# projections are vectors
P_l_p = np.dot(p - O, v) / np.dot(v, v) * v # projection of Op onto l_1
P_l_P_1 = np.dot(p - P_1, v) / np.dot(v, v) * v # projection of (P_1)p onto l_1
# let's first plot these projections as if they originate from the origin
utils.plot_vector(ax, O, P_l_p, color="r", label="P_l_p")
utils.plot_vector(ax, O, P_l_P_1, color="g", label="P_l_P_1")


# Add legend
ax.legend()

# Optionally, you can adjust the legend position and font size
# ax.legend(loc='upper right', fontsize='small')

plt.show()
