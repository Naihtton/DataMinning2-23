import numpy as np
from Dtreefunc import entropy, inforD
from PreProcess import preprocess_data

# Define a class for each node
class TreeNode:
    def __init__(self, feature=None, threshold=None, pos=None, depth=None):
        self.feature = feature
        self.threshold = threshold
        self.pos = pos
        self.depth = depth
        self.left = None
        self.right = None

# id3 function for calculating info_gains, rules, best threshold, and recursively run through depth.
def id3(X, y, feature_names, depth=0, max_depth=4):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Calculate entropy for the current node
    InD = entropy(*class_counts)

    # Print info(D) for the current node
    print(f"\nInfo(D) at depth {depth}: %5.3f" % InD)
    
    # class distribution print
    class_distribution = {k: v for k, v in zip(unique_classes, class_counts) if k != 'class'}
    print(f"Class distribution at depth {depth}: {class_distribution}")


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

            feature_entropy[j] = entropy(*class_counts)

        # Calculate the information gain for the feature
        feature_InD = inforD(m, feature_entropy)
        feature_info_gain = InD - feature_InD
        print(f"Information Gain for {feature_names[i]}: %5.3f" % feature_info_gain)

        # Check if this feature has the highest information gain
        if feature_info_gain > max_info_gain:
            max_info_gain = feature_info_gain
            best_feature_index = i

            # Determine the thresholding condition for this feature
            if len(unique_values) > 0:
                best_threshold = unique_values[np.argmax(feature_entropy)]
            else:
                best_threshold = None  # or choose a default threshold

    # Print the rule for the current node
    print(f"Rule for node at depth {depth}: {feature_names[best_feature_index]} where the pos is {best_threshold}")

    if depth < max_depth and best_threshold is not None:
        # Split the dataset based on the best feature and threshold
        feature_values = X[:, best_feature_index]
        left_indices = feature_values <= best_threshold
        right_indices = feature_values > best_threshold

        # Recursively build subtrees
        print(f"\nBuilding left subtree at depth {depth+1}:")
        id3(X[left_indices], y[left_indices], feature_names, depth+1, max_depth)

        print(f"\nBuilding right subtree at depth {depth+1}:")
        id3(X[right_indices], y[right_indices], feature_names, depth+1, max_depth)


preprocessed_file_path = preprocess_data("iris/iris.data")

# Read the preprocessed data from the CSV file
with open(preprocessed_file_path, "r") as f:
    lines = f.readlines()

# Process the preprocessed data
X = []
y = []
for line in lines:
    if line.strip():  # Check if the line is not empty
        data = line.strip().split(',')
        X.append(data[:-1])  # Features
        y.append(data[-1])  # Target labels

# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# Define feature names
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

# Call the id3 function to build the decision tree
id3(X, y, feature_names)
