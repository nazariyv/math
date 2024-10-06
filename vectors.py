import numpy as np
import matplotlib.pyplot as plt
from utils import *

vector1 = np.array([1, 1, 1])
vector2 = np.array([-1, -1, -0.9])

print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")
print(f"Length of vector 1: {vector_length(vector1)}")
print(f"Length of vector 2: {vector_length(vector2)}")

dot_product = np.dot(vector1, vector2)
print(f"Dot product of vectors: {dot_product}")

# angle between two vectors is defined as: cos theta = (v1.v2) / (|v1| * |v2|)
# i.e. dot product of two vectors divided by product of their magnitudes
# and so theta = arccos((v1.v2) / (|v1| * |v2|))
angle_rad = angle_between(vector1, vector2)
print(f"Angle between vectors: {angle_rad} radians")

fig, ax = create_3d_plot(figsize=(10, 8))

plot_vector(ax, np.zeros(3), vector1, color='r', label='Vector 1')
plot_vector(ax, np.zeros(3), vector2, color='b', label='Vector 2')

def plot_angle_arc(ax, v1, v2, radius=0.5, num_points=100):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ortho = unit_vector(np.cross(v1_u, v2_u))
    t = np.linspace(0, angle_rad, num_points)
    rotation_matrix = lambda angle: np.array([
        [np.cos(angle) + ortho[0]**2 * (1 - np.cos(angle)),
         ortho[0] * ortho[1] * (1 - np.cos(angle)) - ortho[2] * np.sin(angle),
         ortho[0] * ortho[2] * (1 - np.cos(angle)) + ortho[1] * np.sin(angle)],
        [ortho[1] * ortho[0] * (1 - np.cos(angle)) + ortho[2] * np.sin(angle),
         np.cos(angle) + ortho[1]**2 * (1 - np.cos(angle)),
         ortho[1] * ortho[2] * (1 - np.cos(angle)) - ortho[0] * np.sin(angle)],
        [ortho[2] * ortho[0] * (1 - np.cos(angle)) - ortho[1] * np.sin(angle),
         ortho[2] * ortho[1] * (1 - np.cos(angle)) + ortho[0] * np.sin(angle),
         np.cos(angle) + ortho[2]**2 * (1 - np.cos(angle))]
    ])
    arc_points = np.array([np.dot(rotation_matrix(angle), v1_u) for angle in t])
    arc_points *= radius
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='g', lw=2)
    mid_point = arc_points[len(arc_points)//2]
    add_text_3d(ax, mid_point, f'{angle_rad:.2f} rad', fontsize=10)

plot_angle_arc(ax, vector1, vector2)

midpoint = (vector1 + vector2) / 2
add_text_3d(ax, midpoint, f'Dot product: {dot_product:.2f}', bbox=dict(facecolor='white', alpha=0.7))

set_plot_limits(ax, np.vstack((vector1, vector2)))
finalize_plot(ax, '3D Vector Visualization with Angle Arc and Dot Product')

plt.show()
