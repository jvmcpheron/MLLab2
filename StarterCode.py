import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# retrieve the data and extract the input features (X) and target (y)
dataset = openml.datasets.get_dataset('lisbon-house-prices',download_data=True,download_qualities=True,download_features_meta_data=True)
X, y, cat_features, feature_names = dataset.get_data(target='Price')
X = np.array(X)
y = y.to_numpy().reshape(-1,1)

# determine which features are string (discrete)
string_cols = dataset.get_features_by_type('string')
non_string_cols = list(set(range(0,len(X[0]))) - set(string_cols))
non_string_cols.reverse()

# scikit-learns decision tree system does not work for nominal features
# so encode such features using a OneHotEncoder
onehot = OneHotEncoder(sparse_output=False,drop='if_binary')
preproc = ColumnTransformer(transformers=[('onehot', onehot, string_cols)],remainder='passthrough')
Xnew = preproc.fit_transform(X)

# If any features got mapped they will have new names, rework the names to include original
onehot_names = preproc.get_feature_names_out()

for i in string_cols:
    onehot_names = [sub.replace('onehot__x'+str(i)+'_',feature_names[i]+'_') for sub in onehot_names]
for i in non_string_cols:
    onehot_names = [sub.replace('remainder__x'+str(i),feature_names[i]) for sub in onehot_names]

# This problem is regression, let's make it classification by dividing the prices into quintiles
kbins = KBinsDiscretizer([5],encode='ordinal').fit(y)
ynew = kbins.transform(y)

# Set the number of folds, repeats and the different sizes for the training data used
nfolds = 10
nrepeats = 10
nsizes = 10

# The data is randomly shuffled and we would like to use the same seeds in creating the
# folds so that we can repeat the code and get the same random folds
first_seed = 42

# Create the arrays to hold the accuracy and size for each size,repeat,fold combination
accuracies = np.zeros((nsizes,nrepeats,nfolds))
sizes = np.zeros((nsizes,nrepeats,nfolds))

# for each of a number of repeats
for r in range(nrepeats):
    # divide the data using a stratified k fold strategy (I have labeled the Lisbon data
    #   with 5 classes by binning the cost, so stratified makes sense)
    skf = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=r+first_seed)

    # keep track of the fold number
    f = 0
    # get the indices of the overall data for each fold
    for train_index, test_index in skf.split(Xnew,ynew):
        # extract the train and test data from the original Xnew and ynew
        Xtrain = Xnew[train_index,:]
        ytrain = ynew[train_index,:]
        Xtest = Xnew[test_index,:]
        ytest = ynew[test_index,:]
        # determine the size of the train set
        num_examples = Xtrain.shape[0]
        # for each of the sizes (1/nsizes)*num_examples, (2/nsize)*num_examples, ..., (nsizes/nsize) * num_examples
        for z in range(nsizes):
            curr_size = round(((z + 1) / nsizes) * num_examples)
            # use the first curr_size number of points from the train set
            Xtrainsub = Xtrain[0:curr_size,:]
            ytrainsub = ytrain[0:curr_size,:]
            # and learn a decision tree
            dtlearner = DecisionTreeClassifier()
            dtlearner.fit(Xtrainsub,ytrainsub)
            # keep track of this size and calculate the accuracy for this size/repeat/fold combination
            sizes[z,r,f] = curr_size
            accuracies[z,r,f] = dtlearner.score(Xtest,ytest)
            print(f'Accuracy for size{z}:{curr_size} , repeat{r}, fold{f} is {accuracies[z,r,f]}')
        f += 1

# create some matrices to store combined values
accuracies_across_folds = np.zeros((nsizes,nrepeats))
accuracies_by_size = np.zeros((nsizes,))
std_by_size = np.zeros((nsizes,))
average_size = np.zeros((nsizes,))
for z in range(nsizes):
    for r in range(nrepeats):
        # combine the accuracy across all the folds for a size/repeat combination
        accuracies_across_folds[z][r] = np.mean(accuracies[z,r,:])
    # calculate the mean and std of the accuracy for some size (calculating over repeats)
    accuracies_by_size[z] = np.mean(accuracies_across_folds[z,:])
    std_by_size[z] = np.std(accuracies_across_folds[z,:])
    # calculate the average size
    average_size[z] = np.mean(sizes[z,:,:])
    print(f'For size {average_size[z]} avg accuracy is {accuracies_by_size[z]} and std is {std_by_size[z]}')

# plot the average accuracy showing the standard deviation as an error bar by the size of the train set
fig = plt.figure()
plt.errorbar(average_size,accuracies_by_size,std_by_size)
plt.xlabel("Size of training data")
plt.ylabel("Accuracy")
plt.show()
