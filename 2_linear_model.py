import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

x_data = [2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# Prepare lists to store w, b, and mse values for 3D plotting
w_list = []
b_list = []
mse_list = []

# Calculate mse for a range of w and b values
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-4.0, 0.1, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val
        mse = l_sum / len(x_data)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# Convert lists to numpy arrays for plotting
w_array = np.array(w_list)
b_array = np.array(b_list)
mse_array = np.array(mse_list)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.scatter(w_array, b_array, mse_array, c='r', marker='o')

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Mean Squared Error (MSE)')

# Show the plot
plt.show()