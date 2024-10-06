import numpy as np
import matplotlib.pyplot as plt

def vector_length(v):
    return np.linalg.norm(v)

def unit_vector(vector):
    return vector / vector_length(vector)

def distance_from_point_to_line(point, line_point, line_direction):
    # Convert line_direction to a unit vector
    line_direction = unit_vector(line_direction)
    
    # Vector from line_point to the given point
    point_vector = point - line_point
    
    # Calculate the projection of point_vector onto the line direction
    projection = np.dot(point_vector, line_direction) * line_direction
    
    # Calculate the perpendicular component
    perpendicular = point_vector - projection
    
    # The distance is the length of the perpendicular component
    return vector_length(perpendicular)

# Define the point and line
point = np.array([2, 3, 1])
line_point = np.array([0, 0, 0])  # Line intercept
line_direction = np.array([1, 1, 1])  # Line direction

# Calculate the distance
distance = distance_from_point_to_line(point, line_point, line_direction)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the line
t = np.linspace(-2, 2, 100)
line_points = line_point[:, np.newaxis] + line_direction[:, np.newaxis] * t
ax.plot(line_points[0], line_points[1], line_points[2], 'b', label='Line')

# Plot the point
ax.scatter(*point, color='r', s=100, label='Point')

# Plot the line intercept point
ax.scatter(*line_point, color='g', s=100, label='Line Intercept')

# Plot the line direction vector
ax.quiver(*line_point, *line_direction, color='m', label='Line Direction')

# Calculate and plot the closest point on the line to the given point
closest_point = line_point + np.dot(point - line_point, unit_vector(line_direction)) * unit_vector(line_direction)
ax.scatter(*closest_point, color='c', s=100, label='Closest Point')

# Plot the line from the point to the closest point (shortest distance)
ax.plot([point[0], closest_point[0]], [point[1], closest_point[1]], [point[2], closest_point[2]], 'r--', label='Shortest Distance')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Distance from Point to Line Visualization')

# Add distance label
midpoint = (point + closest_point) / 2
ax.text(*midpoint, f'Distance: {distance:.2f}', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

# Set axis limits
max_limit = max(np.max(np.abs(line_points)), np.max(np.abs(point))) * 1.2
ax.set_xlim([-max_limit, max_limit])
ax.set_ylim([-max_limit, max_limit])
ax.set_zlim([-max_limit, max_limit])

# Add legend
ax.legend()

# Show the plot
plt.show()