import numpy as np
from Dtreefunc import entropy, inforD

# Read the Iris dataset
with open("iris/iris.data", "r") as f:
    lines = f.readlines()

# Process the data
X = []
y = []
for line in lines:
    if line.strip():  # Check if the line is not empty
        data = line.strip().split(',')
        # Apply thresholding
        sepal_length = 1 if float(data[0]) <= 5.5 else (2 if float(data[0]) <= 6.7 else 3)
        sepal_width = 1 if float(data[1]) <= 2.8 else (2 if float(data[1]) <= 3.6 else 3)
        petal_length = 1 if float(data[2]) <= 3.0 else (2 if float(data[2]) <= 4.9 else 3)
        petal_width = 1 if float(data[3]) <= 0.9 else (2 if float(data[3]) <= 1.7 else 3)
        X.append([sepal_length, sepal_width, petal_length, petal_width])  # Features
        y.append(data[-1])  # Target labels

# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# Define feature names
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

# Initialize arrays to store counts and information related to different attributes and classes
unique_classes = np.unique(y)
class_counts = np.zeros(len(unique_classes))  # For counting classes

# Calculate class counts
for i, c in enumerate(unique_classes):
    class_counts[i] = np.sum(y == c)

# Calculate entropy for the entire dataset
InD = entropy(class_counts[0], class_counts[1] + class_counts[2])

# Print info(D)
print(f"Info(D): {InD}")

# Calculate information gain for each feature and find the one with the highest gain
max_info_gain = float('-inf')
best_feature_index = None
best_threshold = None

for i in range(X.shape[1]):
    feature_values = X[:, i]
    unique_values = np.unique(feature_values)
    feature_entropy = np.zeros(len(unique_values))
    m = [np.sum(feature_values == value) for value in unique_values]

    for j, value in enumerate(unique_values):
        subset_indices = feature_values == value
        class_counts = np.zeros(len(unique_classes))

        for k, c in enumerate(unique_classes):
            class_counts[k] = np.sum(y[subset_indices] == c)
        
        feature_entropy[j] = entropy(class_counts[0], class_counts[1] + class_counts[2])

    # Calculate the information gain for the feature
    feature_InD = inforD(m, feature_entropy)
    feature_info_gain = InD-feature_InD
    print(f"Information Gain for {feature_names[i]}: {feature_info_gain}")

    # Check if this feature has the highest information gain
    if feature_info_gain > max_info_gain:
        max_info_gain = feature_info_gain
        best_feature_index = i

        # Determine the thresholding condition for this feature
        best_threshold = unique_values[np.argmax(feature_entropy)]
        
        
#  # Print attribute counts
#     print(f"Attribute counts for {feature_names[i]}:")
#     for j, value in enumerate(unique_values):
#         print(f"{value}: {m[j]}")
#     print() 
    
    
# Print the rule for the root node
print(f"Rule for the root node: {feature_names[best_feature_index]} <= threshold {best_threshold}")
