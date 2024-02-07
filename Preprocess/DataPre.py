import numpy as np
import matplotlib.pyplot as plt

# Load data from the datasetCh2_stu.txt file
with open('datasetCh2_stu.txt', 'r') as file:
    data = [list(map(int, line.split())) for line in file.readlines()]

# Extract the first and second columns as X and Y values
x_values, y_values = zip(*data)

xp = np.arange(min(x_values), max(x_values) + 1, 0.01)  # array with 100 points between min(x_values) and max(x_values)

# Values of X for which you want to find Y in graphs
x_to_find = [200, 350, 400]

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Iterate through polynomial degrees
for i in range(1, 7, 1):
    p = np.polyfit(x_values, y_values, i)
    p = np.poly1d(p)
    
    # Calculate polynomial values for the entire range
    yp = np.polyval(p, xp)
    
    # Calculate error
    e = np.polyval(p, x_values) - y_values
    error = np.sum(e**2)
    
    # Plot the data and the polynomial fit
    ax = axs[(i-1)//3, (i-1)%3]
    ax.plot(xp, yp, label=f'Degree {i} Polynomial')
    ax.scatter(x_values, y_values, color='red', label='Data points', marker='o')
    
    # Plot points for which you want to find Y
    y_to_find = np.polyval(p, x_to_find)
    ax.scatter(x_to_find, y_to_find, color='blue', marker='x', label='Y Values to Find')
    
    # X limit for showing in answer 4
    ax.set_xlim([min(x_values), 400])
    
    # Add labels and legend
    ax.set_title(f'Degree {i} Polynomial \nError: {error:.2f}')
    ax.set_xlabel('X Attribute')
    ax.set_ylabel('Y Attribute')
    ax.legend()


# Adjust layout for better spacing
plt.tight_layout()
plt.show()
