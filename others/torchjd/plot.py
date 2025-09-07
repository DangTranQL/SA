import numpy as np
import matplotlib.pyplot as plt

coords = np.loadtxt("points.csv", delimiter=",", skiprows=1)

x, y = coords[:, 0], coords[:, 1]
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Saved Coordinates")
plt.grid(True)
plt.show()