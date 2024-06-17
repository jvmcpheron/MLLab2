import pandas as pd
import numpy as np
import arff  # Use the liac-arff library
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# Load the data using liac-arff
with open('sick.arff', 'r') as file:
    dataset = arff.load(file)

data = dataset['data']
attributes = dataset['attributes']
attribute_names = [attribute[0] for attribute in attributes]

# Create a DataFrame
dfX = pd.DataFrame(data, columns=attribute_names)

# The target variable is the third feature
target_index = 2

# Extract the target variable
ynew = dfX[attribute_names[target_index]].values

# Debug: Check data types
print(f"Attribute names: {attribute_names}")
print(f"First few labels (ynew): {ynew[:5]}")

# Convert byte labels to string labels if necessary
if isinstance(ynew[0], bytes):
    ynew = [label.decode('utf-8') for label in ynew]

# Ensure attribute names are strings
attribute_names = [str(attribute) for attribute in attribute_names]

# Debug: Check label conversion
print(f"First few decoded labels: {ynew[:5]}")

# Turn the labels into simple integers
labelenc = LabelEncoder()
labelenc.fit(ynew)
ynew = labelenc.transform(ynew)

# Debug: Check label encoding
print(f"Encoded labels: {ynew[:5]}")
print(f"Class names: {labelenc.classes_}")

# Extract input features (excluding the target feature)
Xnew = dfX.drop(columns=[attribute_names[target_index]])

# Convert categorical features to numerical using one-hot encoding
Xnew = pd.get_dummies(Xnew)

# Debug: Check the first few rows of the transformed features
print(f"First few rows of Xnew after encoding:\n{Xnew.head()}")

# Learn the decision tree
dtlearner = DecisionTreeClassifier()
dtlearner.fit(Xnew, ynew)

# Print the decision tree to a file
dot_data = StringIO()
export_graphviz(
    dtlearner,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=Xnew.columns,
    class_names=[str(cls) for cls in labelenc.classes_]
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('sicktree.png')

# Display the image
Image(graph.create_png())

