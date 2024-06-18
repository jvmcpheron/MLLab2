# Tasks
For each of the datasets from Lab 1 I want you to produce a learning curve plot, but where you have multiple lines to compare the performances of some varied learning methods:

1. Decision trees as in Lab 1 using the default settings

2. Decision trees, but where you choose some alternate values of min_samples_split and/or min_samples_leaf (if you set these numbers higher than the defaults which are 2 and 1 respectively this causes the algorithm to stop splitting at that point, which causes the tree to be shorter)

3. Decision trees using a different set of values for min_samples_split and/or min_samples_leaf from options 1 and 2

4. Decision trees using a different set of values for min_samples_split and/or min_samples_leaf from options 1, 2, and 3.

5. Gaussian naive Bayes on the same set of data.

Note that you will need to update the code generating the graph to show each of the methods above (1-5) differently (perhaps using different colored lines) and include a legend on the graph.