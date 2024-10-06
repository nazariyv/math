import matplotlib.pyplot as plt
import numpy as np


def vector_length(v):
    """Calculate the length of a vector.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        float: Length of the vector.

    Example:
        >>> import numpy as np
        >>> v = np.array([3, 4])
        >>> vector_length(v)
        5.0
    """
    return np.linalg.norm(v)


def unit_vector(vector):
    """Calculate the unit vector of a given vector.

    Args:
        vector (np.ndarray): Input vector.

    Returns:
        np.ndarray: Unit vector.

    Example:
        >>> import numpy as np
        >>> v = np.array([3, 4])
        >>> unit_vector(v)
        array([0.6, 0.8])
    """
    return vector / vector_length(vector)


def angle_between(v1, v2):
    """Calculate the angle between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        float: Angle between the vectors in radians.

    Example:
        >>> import numpy as np
        >>> v1 = np.array([1, 0])
        >>> v2 = np.array([0, 1])
        >>> round(angle_between(v1, v2), 2)
        1.57
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def create_3d_plot(figsize=(12, 10)):
    """Create a 3D plot.

    Args:
        figsize (tuple): Figure size (width, height).

    Returns:
        tuple: Figure and Axes objects.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def plot_vector(ax, start, vector, color="r", label=None):
    """Plot a vector in 3D space.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        start (np.ndarray): Starting point of the vector.
        vector (np.ndarray): Vector to plot.
        color (str): Color of the vector.
        label (str): Label for the vector.
    """
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        label=label,
    )


def plot_line(
    ax,
    start,
    direction,
    t_range=(-2, 2),
    num_points=100,
    color="b",
    linestyle="-",
    label=None,
):
    """Plot a line in 3D space.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        start (np.ndarray): Starting point of the line.
        direction (np.ndarray): Direction vector of the line.
        t_range (tuple): Range of the parameter t.
        num_points (int): Number of points to plot.
        color (str): Color of the line.
        linestyle (str): Style of the line.
        label (str): Label for the line.
    """
    t = np.linspace(t_range[0], t_range[1], num_points)
    line_points = start[:, np.newaxis] + direction[:, np.newaxis] * t
    ax.plot(
        line_points[0],
        line_points[1],
        line_points[2],
        color=color,
        linestyle=linestyle,
        label=label,
    )


def plot_point(ax, point, color="r", size=100, label=None):
    """Plot a point in 3D space.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        point (np.ndarray): Point to plot.
        color (str): Color of the point.
        size (float): Size of the point.
        label (str): Label for the point.
    """
    ax.scatter(*point, color=color, s=size, label=label)


def plot_shortest_distance(ax, point1, point2, color="r", linestyle="--", label=None):
    """Plot the shortest distance between two points in 3D space.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.
        color (str): Color of the line.
        linestyle (str): Style of the line.
        label (str): Label for the line.
    """
    ax.plot(
        [point1[0], point2[0]],
        [point1[1], point2[1]],
        [point1[2], point2[2]],
        color=color,
        linestyle=linestyle,
        label=label,
    )


def add_text_3d(ax, position, text, fontsize=10, ha="center", va="center", bbox=None):
    """Add text to a 3D plot.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        position (np.ndarray): Position of the text.
        text (str): Text to add.
        fontsize (int): Font size of the text.
        ha (str): Horizontal alignment.
        va (str): Vertical alignment.
        bbox (dict): Bounding box properties.
    """
    ax.text(*position, text, fontsize=fontsize, ha=ha, va=va, bbox=bbox)


def set_plot_limits(ax, points, scale=1.2):
    """Set the limits of a 3D plot based on the given points.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        points (np.ndarray): Array of points.
        scale (float): Scale factor for the limits.
    """
    max_limit = np.max(np.abs(points)) * scale
    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])


def finalize_plot(ax, title, xlabel="X", ylabel="Y", zlabel="Z"):
    """Finalize the 3D plot by setting labels, title, and legend.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()
