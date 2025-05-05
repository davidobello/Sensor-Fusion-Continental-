import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants
SPEED_OF_SOUND = 343  # m/s
SENSOR_ANGLES = np.deg2rad([45, -45, -135, 135])  # angles relative to robot heading
ROBOT_RADIUS = 0.1  # meters (approx. center to sensor)
MAP_SIZE = 100
CELL_SIZE = 0.1  # each cell is 10cm
P_HIT = 0.7
P_MISS = 0.4
L_HIT = np.log(P_HIT / (1 - P_HIT))
L_MISS = np.log(P_MISS / (1 - P_MISS))
L0 = 0  # prior log-odds
MAX_RANGE = 3.0  # max sensor range in meters

# Load data
df = pd.read_csv("robot.csv")
df.iloc[:, 4:] = df.iloc[:, 4:] * SPEED_OF_SOUND / 2  # Convert TOF to distance

# Initialize occupancy grid in log-odds
log_odds_grid = np.zeros((MAP_SIZE, MAP_SIZE))

def world_to_grid(x, y):
    return int(x / CELL_SIZE) + MAP_SIZE // 2, int(y / CELL_SIZE) + MAP_SIZE // 2

def bresenham(x0, y0, x1, y1):
    """ Bresenham's Line Algorithm for ray tracing """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def update_map(row):
    x_r, y_r, theta = row[1], row[2], row[3]
    distances = row[4:].values

    for i, d in enumerate(distances):
        if d > MAX_RANGE:  # Skip invalid readings
            continue

        # Sensor position in world frame
        angle = theta + SENSOR_ANGLES[i]
        x_s = x_r + ROBOT_RADIUS * math.cos(angle)
        y_s = y_r + ROBOT_RADIUS * math.sin(angle)

        # Obstacle position in world frame
        x_o = x_s + d * math.cos(angle)
        y_o = y_s + d * math.sin(angle)

        # Convert to grid
        gx_s, gy_s = world_to_grid(x_s, y_s)
        gx_o, gy_o = world_to_grid(x_o, y_o)

        # Cells along the ray: decrease log-odds (free)
        ray_cells = bresenham(gx_s, gy_s, gx_o, gy_o)[:-1]  # exclude obstacle
        for (x, y) in ray_cells:
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                log_odds_grid[y, x] += L_MISS

        # Obstacle cell: increase log-odds (occupied)
        if 0 <= gx_o < MAP_SIZE and 0 <= gy_o < MAP_SIZE:
            log_odds_grid[gy_o, gx_o] += L_HIT

# Process all measurements
for _, row in df.iterrows():
    update_map(row)

# Convert log-odds to probability for display
prob_grid = 1 - 1 / (1 + np.exp(log_odds_grid))

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(prob_grid, cmap="gray_r", origin="lower")
plt.title("Bayesian Occupancy Grid Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="P(Occupied)")
plt.show()
