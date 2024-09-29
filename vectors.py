import numpy as np
import matplotlib.pyplot as plt

def vector_length(v):
    return np.linalg.norm(v)

def unit_vector(vector):
    return vector / vector_length(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Define two 3D vectors
vector1 = np.array([1, 1, 1])
vector2 = np.array([-1, -1, -0.9])

print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")
print(f"Length of vector 1: {vector_length(vector1)}")
print(f"Length of vector 2: {vector_length(vector2)}")

# Calculate dot product
dot_product = np.dot(vector1, vector2)
print(f"Dot product of vectors: {dot_product}")

# angle between two vectors is defined as: cos theta = (v1.v2) / (|v1| * |v2|)
# i.e. dot product of two vectors divided by product of their magnitudes
# and so theta = arccos((v1.v2) / (|v1| * |v2|))
angle = np.arccos(dot_product / (vector_length(vector1) * vector_length(vector2)))
print(f"Angle between vectors: {angle} radians")

# Calculate the angle between vectors
angle_rad = angle_between(vector1, vector2)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot vector1
ax.quiver(0, 0, 0, vector1[0], vector1[1], vector1[2], color='r', label='Vector 1')

# Plot vector2
ax.quiver(0, 0, 0, vector2[0], vector2[1], vector2[2], color='b', label='Vector 2')

# Create and add the arc
def plot_angle_arc(ax, v1, v2, radius=0.5, num_points=100):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    # Calculate the orthogonal vector to define the plane of rotation
    ortho = np.cross(v1_u, v2_u)
    ortho = unit_vector(ortho)
    
    # Create points along the arc
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
    
    # Plot the arc
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='g', lw=2)
    
    # Add angle label
    mid_point = arc_points[len(arc_points)//2]
    ax.text(mid_point[0], mid_point[1], mid_point[2], f'{angle_rad:.2f} rad', 
            fontsize=10, ha='center', va='center')

plot_angle_arc(ax, vector1, vector2)

# Add dot product label
# We'll position it at the midpoint between the tips of the two vectors
midpoint = (vector1 + vector2) / 2
ax.text(midpoint[0], midpoint[1], midpoint[2], f'Dot product: {dot_product:.2f}', 
        fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Visualization with Angle Arc and Dot Product')

# Set axis limits
max_limit = max(np.max(np.abs(vector1)), np.max(np.abs(vector2)))
ax.set_xlim([-max_limit, max_limit])
ax.set_ylim([-max_limit, max_limit])
ax.set_zlim([-max_limit, max_limit])

# Add legend
ax.legend()

# Show the plot
plt.show()