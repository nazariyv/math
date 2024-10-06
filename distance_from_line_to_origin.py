import matplotlib.pyplot as plt
import numpy as np

import utils


def distance_from_line_to_origin(p_0, v):
    """Calculate the shortest distance from a line to the origin.

    Args:
        p_0 (np.ndarray): A point on the line.
        v (np.ndarray): Direction vector of the line.

    Returns:
        float: The shortest distance from the line to the origin.

    Example:
        >>> import numpy as np
        >>> p_0 = np.array([1, 1, 1])
        >>> v = np.array([1, 0, 0])
        >>> round(distance_from_line_to_origin(p_0, v), 2)
        1.41
    """
    P_l_p_0 = (np.dot(p_0, v) / np.dot(v, v)) * v
    P_l_perp_p_0 = p_0 - P_l_p_0
    return utils.vector_length(P_l_perp_p_0)


# Example usage and visualization
p_0 = np.array([1, 1, 1])
v = np.array([2, 1, 5])
distance = distance_from_line_to_origin(p_0, v)

fig, ax = utils.create_3d_plot()

utils.plot_line(ax, p_0, v, label="Original Line")
utils.plot_point(ax, p_0, label="p_0")
utils.plot_vector(ax, p_0, v, color="g", label="v")

utils.plot_line(
    ax, np.zeros(3), v, color="c", linestyle="--", label="Parallel Line through Origin"
)

closest_point = p_0 - (np.dot(p_0, v) / np.dot(v, v)) * v
utils.plot_point(ax, closest_point, color="m", label="Closest Point")

utils.plot_shortest_distance(ax, np.zeros(3), closest_point, label="Shortest Distance")

midpoint = closest_point / 2
utils.add_text_3d(
    ax, midpoint, f"Distance: {distance:.2f}", bbox={"facecolor": "white", "alpha": 0.7}
)

utils.set_plot_limits(ax, np.vstack((p_0, v, closest_point)))
utils.finalize_plot(ax, "Distance from Line to Origin Visualization")

plt.show()
