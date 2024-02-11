import csv
import numpy as np

def preprocess_data(file_path):
    """
    Preprocesses the Iris dataset and saves it as a CSV file.
    """
    X = []
    y = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip():
            data = line.strip().split(',')
            X.append(data[:-1])  # Features
            y.append(data[-1])   # Target labels

    # Convert strings to floats
    X = np.array(X, dtype=float)
    y = np.array(y)

    # Define thresholds for each feature
    thresholds = [
        np.percentile(X[:, 0], [33, 66]),  # sepal length
        np.percentile(X[:, 1], [33, 66]),  # sepal width
        np.percentile(X[:, 2], [33, 66]),  # petal length
        np.percentile(X[:, 3], [33, 66])   # petal width
    ]

    # Discretize features based on thresholds
    for i in range(len(X)):
        for j in range(len(thresholds)):
            if X[i, j] <= thresholds[j][0]:
                X[i, j] = 0
            elif X[i, j] <= thresholds[j][1]:
                X[i, j] = 1
            else:
                X[i, j] = 2

    # Save the preprocessed data as CSV
    output_file_path = file_path.replace('.data', '_preprocessed.csv')
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        for i in range(len(X)):
            writer.writerow(list(X[i]) + [y[i]])

    return output_file_path

# Example usage:
preprocessed_file_path = preprocess_data("iris/iris.data")
print("Preprocessed file saved at:", preprocessed_file_path)
