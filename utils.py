
# utils.py
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

def create_3d_plot(figsize=(12, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def plot_vector(ax, start, vector, color='r', label=None):
    ax.quiver(start[0], start[1], start[2], vector[0], vector[1], vector[2], color=color, label=label)

def plot_line(ax, start, direction, t_range=(-2, 2), num_points=100, color='b', linestyle='-', label=None):
    t = np.linspace(t_range[0], t_range[1], num_points)
    line_points = start[:, np.newaxis] + direction[:, np.newaxis] * t
    ax.plot(line_points[0], line_points[1], line_points[2], color=color, linestyle=linestyle, label=label)

def plot_point(ax, point, color='r', size=100, label=None):
    ax.scatter(*point, color=color, s=size, label=label)

def plot_shortest_distance(ax, point1, point2, color='r', linestyle='--', label=None):
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=color, linestyle=linestyle, label=label)

def add_text_3d(ax, position, text, fontsize=10, ha='center', va='center', bbox=None):
    ax.text(*position, text, fontsize=fontsize, ha=ha, va=va, bbox=bbox)

def set_plot_limits(ax, points, scale=1.2):
    max_limit = np.max(np.abs(points)) * scale
    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])

def finalize_plot(ax, title, xlabel='X', ylabel='Y', zlabel='Z'):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()