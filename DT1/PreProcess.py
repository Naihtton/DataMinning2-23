import numpy as np
import csv

# Read the Iris dataset
with open("datasets/iris.data", "r") as f:
    lines = f.readlines()

# Process the data
X = []
y = []
for line in lines:
    if line.strip():  # Check if the line is not empty
        data = line.strip().split(',')
        # Apply thresholding
        sepal_length = "<=5.4" if float(data[0]) <= 5.4 else ("5.4 -> 6.5" if float(data[0]) <= 6.5 else ">6.5")
        sepal_width = "<=2.8" if float(data[1]) <= 2.8 else ("2.8 -> 3.6" if float(data[1]) <= 3.6 else ">3.6")
        petal_length = "<=3" if float(data[2]) <= 3.0 else ("3 -> 4.9" if float(data[2]) <= 4.9 else ">4.9")
        petal_width = "<=0.9" if float(data[3]) <= 0.9 else ("0.9 -> 1.7" if float(data[3]) <= 1.7 else ">1.7")
        X.append([sepal_length, sepal_width, petal_length, petal_width])  # Features
        y.append(data[-1])  # Target labels

# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# Define the output CSV file path
output_csv_file = "datasets/iris_processed.csv"

# Write the processed data to a CSV file with header
with open(output_csv_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    # Write the processed data
    for i in range(len(X)):
        writer.writerow([X[i][0], X[i][1], X[i][2], X[i][3], y[i]])
    print("Iris dataset is pre-processed")
