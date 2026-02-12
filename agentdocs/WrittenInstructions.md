1. Download the Iris dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Randomly place 80% of the datapoints into a training dataset and the remaining 20%
into a test dataset (80/20 split), representing labels as three-element one-hot vectors.
Set np.random.seed(0) prior to performing the random split.
a. Using a network with 5 hidden units, create a plot of average loss across all
training samples (e.g., cross-entropy after you perform softmax) per-epoch, over
10 training epochs. Include four lines on your plot, one for each learning rate of 1,
1e-2, 1e-3 and 1e-8. Do not vary any other hyperparameters. Make sure your
plot includes a title, x/y axis labels, and a legend.
b. According to this plot, which LR seems to produce the best model? Explain your
reasoning.
c. Give the average loss across all test set datapoints for each of these four models
(make sure not to perform any further model parameter updates!). According to
test set loss, which LR seems to produce the best model? Explain your reasoning
and compare these findings to the findings produced by the training set losses.
d. Using an LR of 1e-2, create a plot of average loss by training epoch over 10
epochs. Include four lines on your plot, one for each hidden layer size of 2, 8, 16
and 32 units. Do not vary any other hyperparameters. Make sure your plot
includes a title, x/y axis labels, and a legend.
e. Give the average test loss for these four models, as well as the models’ accuracy
on the test set in predicting the species of Iris (accuracy is defined as the fraction
of predictions where argmax(output) equals the true label). Describe what
patterns you notice in performance across these model sizes in both the training
and the test losses.
2. Repeat subparts a-e again for an 80/20 split of the California housing dataset:
raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
Use linear output with mean squared error (MSE) loss and “median_house_value” as the
target variable. In part e, provide MSE instead of accuracy. Standardize input features to
zero mean and variance of 1 before training to ensure stable gradients.
3. Repeat subparts a-e again for the MNIST dataset. Download the train/test images and
labels from: https://github.com/cvdfoundation/mnist
Because MNIST datapoints are images (represented by matrices), we will need28 × 28
to convert each datapoint into a -element vector. To do so, use the28 × 28 = 784
MNIST starter code provided at the end of this assignment. This is a classification
dataset, so use cross-entropy and represent labels as 10-element one-hot vectors. Note:
we will use the official 60,000 / 10,000 train/test split. Do not create a custom 80/20 split
for MNIST. You may train on a subset of 20,000 MNIST training examples if runtime
becomes an issue–just make sure to note it in your Written submission.
4. Compare the performance of the network across datasets. Does any of the datasets
seem easier or harder for the model to learn than the others? Use your plots and other
results as evidence. Why do you think this might be, and what steps would you consider
taking to help the model perform better?
5. During implementation of the neural network, what are three bugs/errors you
encountered? For each bug, paste a few lines of output/error printout showcasing the
issue, describe the cause and how you went about debugging the issue.
