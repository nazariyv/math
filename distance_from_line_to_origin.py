# TODO: write a readme with math for each of these files
# line is described by a point and a vector
# denote the point by p_0 and the vector by v
# then line (call it l) is described by the equation l = p_0 + t*v
# denote shortest distance from origin to this line by d(l, O)
# now recall the concept of projection
# consider line l' which is parallel to l, but it passes through origin
# so it will be described by the equation l' = 0 + t*v
# now, d(l, O) = d(l', p_0)
# if you think of p_0 as a vector from origin to that point,
# then d(l', p_0) = || P_{vector v perpendicular}(p_0) ||
# i.e. perpendicular projection of p_0 on l' (because vector v is parallel to l')
# where P_{vector v perpendicular}(p_0) = p_0 - P_{v perp}(p_0)
# where P_{v perp}(p_0) = (p_0 \cdot v / ||v||^2) * v

import numpy as np
import matplotlib.pyplot as plt

def vector_length(v):
    return np.linalg.norm(v)

def unit_vector(vector):
    return vector / vector_length(vector)

def distance_from_line_to_origin(p_0, v):
    # Calculate the projection of p_0 onto v
    P_l_p_0 = (np.dot(p_0, v) / np.dot(v, v)) * v
    
    # Calculate the perpendicular component
    P_l_perp_p_0 = p_0 - P_l_p_0
    
    # The distance is the length of the perpendicular component
    return vector_length(P_l_perp_p_0)

# Define a point and a vector for the line
p_0 = np.array([1, 1, 1])
v = np.array([2, 1, 5])

# Calculate the distance
distance = distance_from_line_to_origin(p_0, v)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the original line
t = np.linspace(-2, 2, 100)
line_points = p_0[:, np.newaxis] + v[:, np.newaxis] * t
ax.plot(line_points[0], line_points[1], line_points[2], 'b', label='Original Line')

# Plot the point p_0
ax.scatter(*p_0, color='r', s=100, label='p_0')

# Plot the vector v from p_0
ax.quiver(*p_0, *v, color='g', label='v')

# Plot the line passing through the origin parallel to the original line
origin_line_points = v[:, np.newaxis] * t
ax.plot(origin_line_points[0], origin_line_points[1], origin_line_points[2], 'c--', label='Parallel Line through Origin')

# Calculate and plot the closest point on the line to the origin
closest_point = p_0 - (np.dot(p_0, v) / np.dot(v, v)) * v
ax.scatter(*closest_point, color='m', s=100, label='Closest Point')

# Plot the line from the origin to the closest point
ax.plot([0, closest_point[0]], [0, closest_point[1]], [0, closest_point[2]], 'r--', label='Shortest Distance')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Distance from Line to Origin Visualization')

# Add distance label
midpoint = closest_point / 2
ax.text(*midpoint, f'Distance: {distance:.2f}', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

# Set axis limits
max_limit = max(np.max(np.abs(line_points)), np.max(np.abs(origin_line_points))) * 1.2
ax.set_xlim([-max_limit, max_limit])
ax.set_ylim([-max_limit, max_limit])
ax.set_zlim([-max_limit, max_limit])

# Add legend
ax.legend()

# Show the plot
plt.show()