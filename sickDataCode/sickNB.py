import openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

# Retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('sick', download_data=True, download_qualities=True,
                                      download_features_meta_data=True)
X, y, _, feature_names = dataset.get_data(target='Class')
X = np.array(X)
y = y.to_numpy().ravel()

# Determine which features are string (discrete)
string_cols = dataset.get_features_by_type('string')
nominal_cols = dataset.get_features_by_type('nominal')
categorical_cols = list(set(string_cols + nominal_cols))
non_categorical_cols = list(set(range(len(X[0]))) - set(categorical_cols))

# Ensure categorical_cols indices are within valid range
max_index = X.shape[1] - 1
categorical_cols = [col for col in categorical_cols if col <= max_index]

# Print for debugging
print(f"Categorical columns: {categorical_cols}")
print(f"Non-categorical columns: {non_categorical_cols}")

# Encode categorical features using OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
preproc = ColumnTransformer(transformers=[('onehot', onehot, categorical_cols)], remainder='passthrough')
Xnew = preproc.fit_transform(X)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
Xnew = imputer.fit_transform(Xnew)

# If any features got mapped, they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()
for i in categorical_cols:
    onehot_names = [sub.replace(f'onehot__x{i}', feature_names[i]) for sub in onehot_names]
for i in non_categorical_cols:
    onehot_names = [sub.replace(f'remainder__x{i}', feature_names[i]) for sub in onehot_names]

# Set the number of folds, repeats, and the different sizes for the training data used
nfolds = 10
nrepeats = 10
nsizes = 10
first_seed = 42

# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# For each of a number of repeats
for r in range(nrepeats):
    # Divide the data using a stratified k-fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # Keep track of the fold number
    f = 0

    # Get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # Extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]

        # Determine the size of the train set
        num_examples = Xtrain.shape[0]

        # For each of the sizes (1/nsizes)*num_examples, (2/nsizes)*num_examples, ..., (nsizes/nsizes) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)

            # Use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            # Learn a decision tree
            dtlearner = GaussianNB()
            dtlearner.fit(Xtrainsub, ytrainsub)

            # Keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# Create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size = np.zeros((nsizes,))
std_by_size = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))

for z in range(nsizes):
    for r in range(nrepeats):
        # Combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])

    # Calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size[z] = np.std(accuracies_across_folds[z, :])

    # Calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size[z]} and std is {std_by_size[z]}')

# Plot the average accuracy showing the standard deviation as an error bar by the size of the train set
fig = plt.figure()
plt.errorbar(average_size, accuracies_by_size, std_by_size)
plt.xlabel("Size of training data")
plt.ylabel("Accuracy")
plt.show()

