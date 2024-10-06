import matplotlib.pyplot as plt
import numpy as np

import utils


def distance_from_point_to_line(point, line_point, line_direction):
    """Calculate the shortest distance from a point to a line.

    Args:
        point (np.ndarray): The point to calculate the distance from.
        line_point (np.ndarray): A point on the line.
        line_direction (np.ndarray): Direction vector of the line.

    Returns:
        float: The shortest distance from the point to the line.

    Example:
        >>> import numpy as np
        >>> point = np.array([1, 1, 1])
        >>> line_point = np.array([0, 0, 0])
        >>> line_direction = np.array([1, 0, 0])
        >>> round(distance_from_point_to_line(point, line_point, line_direction), 2)
        1.41
    """
    line_direction = utils.unit_vector(line_direction)
    point_vector = point - line_point
    projection = np.dot(point_vector, line_direction) * line_direction
    perpendicular = point_vector - projection
    return utils.vector_length(perpendicular)


# Example usage and visualization
point = np.array([10, 10, 10])
line_point = np.array([3, 0, 5])
line_direction = np.array([1, -2, 0])

distance = distance_from_point_to_line(point, line_point, line_direction)

fig, ax = utils.create_3d_plot()

utils.plot_line(ax, line_point, line_direction, label="Line")
utils.plot_point(ax, point, label="Point")
utils.plot_point(ax, line_point, color="g", label="Line Intercept")
utils.plot_vector(ax, line_point, line_direction, color="m", label="Line Direction")

closest_point = line_point + np.dot(
    point - line_point, utils.unit_vector(line_direction)
) * utils.unit_vector(line_direction)
utils.plot_point(ax, closest_point, color="c", label="Closest Point")

utils.plot_shortest_distance(ax, point, closest_point, label="Shortest Distance")

midpoint = (point + closest_point) / 2
utils.add_text_3d(
    ax, midpoint, f"Distance: {distance:.2f}", bbox={"facecolor": "white", "alpha": 0.7}
)

utils.set_plot_limits(ax, np.vstack((point, line_point, line_direction, closest_point)))
utils.finalize_plot(ax, "Distance from Point to Line Visualization")

plt.show()
