import matplotlib.pyplot as plt
import numpy as np

import utils


def plot_angle_arc(ax, v1, v2, radius=0.5, num_points=100):
    """Plot an arc representing the angle between two vectors.

    Args:
        ax (Axes): Matplotlib 3D axes object.
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        radius (float): Radius of the arc.
        num_points (int): Number of points to plot on the arc.
    """
    v1_u = utils.unit_vector(v1)
    v2_u = utils.unit_vector(v2)
    ortho = utils.unit_vector(np.cross(v1_u, v2_u))
    # angle between two vectors is defined as: cos theta = (v1.v2) / (|v1| * |v2|)
    # i.e. dot product of two vectors divided by product of their magnitudes
    # and so theta = arccos((v1.v2) / (|v1| * |v2|))
    angle_rad = utils.angle_between(v1, v2)
    t = np.linspace(0, angle_rad, num_points)
    rotation_matrix = lambda angle: np.array(
        [
            [
                np.cos(angle) + ortho[0] ** 2 * (1 - np.cos(angle)),
                ortho[0] * ortho[1] * (1 - np.cos(angle)) - ortho[2] * np.sin(angle),
                ortho[0] * ortho[2] * (1 - np.cos(angle)) + ortho[1] * np.sin(angle),
            ],
            [
                ortho[1] * ortho[0] * (1 - np.cos(angle)) + ortho[2] * np.sin(angle),
                np.cos(angle) + ortho[1] ** 2 * (1 - np.cos(angle)),
                ortho[1] * ortho[2] * (1 - np.cos(angle)) - ortho[0] * np.sin(angle),
            ],
            [
                ortho[2] * ortho[0] * (1 - np.cos(angle)) - ortho[1] * np.sin(angle),
                ortho[2] * ortho[1] * (1 - np.cos(angle)) + ortho[0] * np.sin(angle),
                np.cos(angle) + ortho[2] ** 2 * (1 - np.cos(angle)),
            ],
        ]
    )
    arc_points = np.array([np.dot(rotation_matrix(angle), v1_u) for angle in t])
    arc_points *= radius
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color="g", lw=2)
    mid_point = arc_points[len(arc_points) // 2]
    utils.add_text_3d(ax, mid_point, f"{angle_rad:.2f} rad", fontsize=10)


# Example usage and visualization
vector1 = np.array([1, 1, 1])
vector2 = np.array([-1, -1, -0.9])

print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")
print(f"Length of vector 1: {utils.vector_length(vector1)}")
print(f"Length of vector 2: {utils.vector_length(vector2)}")

dot_product = np.dot(vector1, vector2)
print(f"Dot product of vectors: {dot_product}")

angle_rad = utils.angle_between(vector1, vector2)
print(f"Angle between vectors: {angle_rad} radians")

fig, ax = utils.create_3d_plot(figsize=(10, 8))

utils.plot_vector(ax, np.zeros(3), vector1, color="r", label="Vector 1")
utils.plot_vector(ax, np.zeros(3), vector2, color="b", label="Vector 2")

plot_angle_arc(ax, vector1, vector2)

midpoint = (vector1 + vector2) / 2
utils.add_text_3d(
    ax,
    midpoint,
    f"Dot product: {dot_product:.2f}",
    bbox={"facecolor": "white", "alpha": 0.7},
)

utils.set_plot_limits(ax, np.vstack((vector1, vector2)))
utils.finalize_plot(ax, "3D Vector Visualization with Angle Arc and Dot Product")

plt.show()
