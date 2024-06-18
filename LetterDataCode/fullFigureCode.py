import openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

# retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('letter', download_data=True, download_qualities=True, download_features_meta_data=True)
X, y, cat_features, feature_names = dataset.get_data(target='class')
X = np.array(X)
y = y.to_numpy().reshape(-1,)

# determine which features are string (discrete)
string_cols = dataset.get_features_by_type('string')
non_string_cols = list(set(range(0, len(X[0]))) - set(string_cols))
non_string_cols.reverse()

# scikit-learns decision tree system does not work for nominal features
# so encode such features using a OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, drop='if_binary')
preproc = ColumnTransformer(transformers=[('onehot', onehot, string_cols)], remainder='passthrough')
Xnew = preproc.fit_transform(X)

# If any features got mapped they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()

for i in string_cols:
    onehot_names = [sub.replace('onehot__x'+str(i)+'_', feature_names[i]+'_') for sub in onehot_names]
for i in non_string_cols:
    onehot_names = [sub.replace('remainder__x'+str(i), feature_names[i]) for sub in onehot_names]



# Set the number of folds, repeats and the different sizes for the training data used
nfolds = 10
nrepeats = 10

# This amount of sizes for the training set
nsizes = 10

# The data is randomly shuffled and we would like to use the same seeds in creating the
# folds so that we can repeat the code and get the same random folds
first_seed = 42


#DEFAULT TREE
# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# for each of a number of repeats
for r in range(nrepeats):
    # divide the data using a stratified k fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # keep track of the fold number
    f = 0
    # get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]
        # determine the size of the train set
        num_examples = Xtrain.shape[0]
        # for each of the sizes (1/nsizes)*num_examples, (2/nsize)*num_examples, ..., (nsizes/nsize) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)
            # use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            #THIS IS TREE
            # and learn a decision tree
            dtlearner = DecisionTreeClassifier()
            dtlearner.fit(Xtrainsub, ytrainsub)
            # keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt = np.zeros((nsizes,))
std_by_size_dt = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))
for z in range(nsizes):
    for r in range(nrepeats):
        # combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])
    # calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt[z] = np.std(accuracies_across_folds[z, :])
    # calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt[z]} and std is {std_by_size_dt[z]}')


#ALT TREE 2

# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# for each of a number of repeats
for r in range(nrepeats):
    # divide the data using a stratified k fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # keep track of the fold number
    f = 0
    # get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]
        # determine the size of the train set
        num_examples = Xtrain.shape[0]
        # for each of the sizes (1/nsizes)*num_examples, (2/nsize)*num_examples, ..., (nsizes/nsize) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)
            # use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            #THIS IS TREE
            # and learn a decision tree
            dtlearner = DecisionTreeClassifier(min_samples_split=100)
            dtlearner.fit(Xtrainsub, ytrainsub)
            # keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt2 = np.zeros((nsizes,))
std_by_size_dt2 = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))
for z in range(nsizes):
    for r in range(nrepeats):
        # combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])
    # calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt2[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt2[z] = np.std(accuracies_across_folds[z, :])
    # calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt2[z]} and std is {std_by_size_dt2[z]}')


#ALT TREE 3

# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# for each of a number of repeats
for r in range(nrepeats):
    # divide the data using a stratified k fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # keep track of the fold number
    f = 0
    # get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]
        # determine the size of the train set
        num_examples = Xtrain.shape[0]
        # for each of the sizes (1/nsizes)*num_examples, (2/nsize)*num_examples, ..., (nsizes/nsize) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)
            # use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            #THIS IS TREE
            # and learn a decision tree
            dtlearner = DecisionTreeClassifier(min_samples_leaf=50)
            dtlearner.fit(Xtrainsub, ytrainsub)
            # keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_dt3 = np.zeros((nsizes,))
std_by_size_dt3 = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))
for z in range(nsizes):
    for r in range(nrepeats):
        # combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])
    # calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_dt3[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_dt3[z] = np.std(accuracies_across_folds[z, :])
    # calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_dt3[z]} and std is {std_by_size_dt3[z]}')


#NAIVE BAYES

# Create the arrays to hold the accuracy and size for each size, repeat, fold combination
accuracies = np.zeros((nsizes, nrepeats, nfolds))
sizes = np.zeros((nsizes, nrepeats, nfolds))

# for each of a number of repeats
for r in range(nrepeats):
    # divide the data using a stratified k fold strategy
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=r + first_seed)

    # keep track of the fold number
    f = 0
    # get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew, y):
        # extract the train and test data from the original Xnew and y
        Xtrain = Xnew[train_index, :]
        ytrain = y[train_index]
        Xtest = Xnew[test_index, :]
        ytest = y[test_index]
        # determine the size of the train set
        num_examples = Xtrain.shape[0]
        # for each of the sizes (1/nsizes)*num_examples, (2/nsize)*num_examples, ..., (nsizes/nsize) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)
            # use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size, :]
            ytrainsub = ytrain[0:curr_size]

            #THIS IS TREE
            # and learn a decision tree
            dtlearner = GaussianNB()
            dtlearner.fit(Xtrainsub, ytrainsub)
            # keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z, r, f] = curr_size
            accuracies[z, r, f] = dtlearner.score(Xtest, ytest)
            print(f'Accuracy for size {curr_size}, repeat {r}, fold {f} is {accuracies[z, r, f]}')
        f += 1

# create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes, nrepeats))
accuracies_by_size_nb = np.zeros((nsizes,))
std_by_size_nb = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))
for z in range(nsizes):
    for r in range(nrepeats):
        # combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z, r, :])
    # calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size_nb[z] = np.mean(accuracies_across_folds[z, :])
    std_by_size_nb[z] = np.std(accuracies_across_folds[z, :])
    # calculate the average size
    average_size[z] = np.mean(sizes[z, :, :])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size_nb[z]} and std is {std_by_size_nb[z]}')



# Plot Decision Tree accuracy with green color
plt.errorbar(average_size, accuracies_by_size_dt, std_by_size_dt, color='green', label='Default Tree')
plt.errorbar(average_size, accuracies_by_size_dt2, std_by_size_dt2, color='blue', label='Split Edit Tree')
plt.errorbar(average_size, accuracies_by_size_dt3, std_by_size_dt3, color='purple', label='Leaf Edit Tree')
plt.errorbar(average_size, accuracies_by_size_nb, std_by_size_nb, color='red', label='Naive Bayes')


plt.xlabel("Size of training data")
plt.ylabel("Accuracy")
plt.legend()
plt.show()