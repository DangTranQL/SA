import json5
import matplotlib.pyplot as plt

# Load the JSON5 file
with open('.cache/000000001.json5', 'r') as f:
    data = json5.load(f)

# Extract all 'f' points
points_f = [point['f'] for point in data.values()]

# Separate x and y for plotting
x = [pt[0] for pt in points_f]
y = [pt[1] for pt in points_f]

# Plot
plt.scatter(x, y, color='blue', label='Points f')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of all points f')
plt.legend()
plt.show()