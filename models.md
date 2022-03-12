## ML Model Training

- Logistic Regression: with penalty, C, solver

- Support Vector Classification: C, kernel, degree, gamma, coef0

- Decision Tree Classifier: criterion(default), splitter(default), max_depth,
min_samples_split, min_samples_leaf (influences only the feature importance list),
max_features

- Random Forest Classifier: criterion, n_estimators, max_depth, min_samples_split,
min_samples_leaf, max_features, max_leaf_nodes

- KNN: n_neighbors, weights, algorithm, leaf_size, metric - default worked best

- ANN with keras and tensorflow.

### Logistic Regression: penalty, C, solver parameter were used.

The logistic regression model is a binary classifier, uses the S-shaped logistic function:
estimates the probability of the positive class. If the probability is greater than 0.5, the
instance belongs to the positive class, if it is less, then the model predicts that it belongs
to the negative class. To calculate the probability, the model computes the weighted
sum of the input features (plus a bias term), and it outputs the logistic of this result.
Since this value is always between 0 and 1, the model then easily computes the
probability.

First, I trained a Logistic Regression model with default values and checked the
accuracy, precision and recall scores. These scores were the same, and relatively high.
We also used the trained model on the training set to check if the model is overfitted.
Since the metrics came back very close to the test set scores, I can conclude that the
model was not overfitted. I plotted a confusion matrix to check false positives and false
negatives in the prediction. The model predicted 11 positive cases for heart attack, but
only 4 were true positives. 7 were false positives, and 13 were false negatives.
After that I performed a grid search to find the best parameters for penalty, C, solver,
starting with broad intervals then narrowing them in the following grid searches. After 2
grid searches I found the best parameters: 'C=206.913808111479, penalty= l2, solver=
liblinear. We trained a new model with the best parameters and checked the
generalization error with the same metrics and MSE. After grid search, the confusion
matrix was the same.

I also checked the feature importances to understand the model better. The top six
features in predicting a heart attack are: angina, chest pain, stroke, chest pressure, high
triglyceride levels and high cholesterol.

Best parameters:

C = 207, penalty = 'l2', solver = 'liblinear'

### Support Vector Classification: with C, kernel, degree, gamma, coef0

The second model was a vector classification model. The Support Vector Classifier is a
linear model, with an algorithm that creates a line or a hyperplane to separate the data
into classes. Since the gamma value is low in the best model, that means that even the
data points far away from the boundary get considerable weight and the curve is more
linear.

After the grid search, I trained a model with the best parameters, and checked the
performance metrics. Accuracy, precision, and recall scores came back the same, 98.48
and MSE=0.015. The model predicted 8 positive cases for heart attack, 4 true positives
and 4 false positives. There were also 13 false negative cases.
Then I performed dimensionality reduction with PCA to see how the results change
while preserving 95% variance. The performance metrics were the same after the PCA,
but the confusion matrix was slightly different. Now the model predicted zero positive
cases, and 17 false negatives.

Best parameters:

C=5

coef0=1

gamma=0.005

kernel='poly'

### Decision Tree Classifier:

The Decision Tree algorithm splits the nodes to have the lowest gini scores. If the model
is not constrained, it splits nodes until it reaches the lowest gini scores. Then it stops
splitting further and the node becomes a leaf. In practice, the gini score is calculated for
every node, then the weighted average of these scores becomes the gini of the given
feature/question. Then after every features’ gini score is calculated this way, the feature
with the lowest gini score will be the next feature that splits the dataset.

The decision tree classifier model was trained with criterion(default), splitter(default),
max_depth, min_samples_split, min_samples_leaf (influences only the feature
importance list), and max_features. However, I first checked how a default model
performs: the metrics were slightly lower than with the previous models: accuracy,
precision, and recall scores = 0.97056. The model predicted 22 positive cases, from
which 19 were false positives. There were also 14 false negative cases. The feature
importance list was a little different from the logistic regression model: the top six
features were angina, BMI, systolic blood pressure, walking time during work, number of
alcoholic drinks per day, and sleep hours per night.

After performing grid search, I trained a new model with the best parameters and
checked the performance metrics: they slightly improved: accuracy score, precision,
and recall scores = 0.984 and MSE=0.016. From the confusion matrix I learned that the
model predicted 3 positive cases from which 2 were false positives. And there were also
16 false negatives. After further tuning the model by hand, I found another model which
performed slightly better: accuracy, precision, and recall scores were 0.9866. The
model predicted 8 positive cases, but 3 were false positives. There were also 12 false
negative cases. The feature importances also changed.

The best parameters were:

max_depth=8

max_leaf_nodes=30

min_samples_split=20

max_features=12

min_samples_leaf=12

criterion(default)=gini

splitter(default)=best

### Random Forest Classifier: 

with criterion, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes

The random forest model is a tree based classifier that splits nodes to minimize the gini
score or entropy. While one single decision tree trains fast, a random forest takes much
longer as it grows many random trees.

The first model with default parameters predicted 2 true positive cases and 15 false
negatives. The metrics were (accuracy, precision, recall scores): 0.9866.
After grid search, the scores slightly decreased to 0.9848 and the MSE=0.151.
The feature importances was similar to the best decision tree model.

The best parameters are:

max_depth=32

max_leaf_nodes=18

min_samples_leaf=2

n_estimators=100

criterion='gini'

### KNN Classifier: 

For this algorithm, n_neighbors, weights, algorithm, leaf_size, metric
parameters were used, and the default parameters gave the same result as the best
parameters. The KNN algorithm classifies a new datapoint by comparing it to a given
number of its nearest neighbors. The models predicted 2 positive cases from which 1
was false positive. There were 16 false negatives as well. The accuracy, precision,
recall scores were 0.9848. 

The best parameters were:

n_neighbors=5

algorithm= 'auto'

leaf_size=10

metric='minkowski'

weights= 'uniform'

### ANN with keras and tensorflow:

For the ANN model, the training data was further divided into training and validation
sets. 450 data points were used for cross validation. After fine tuning the parameters,
the following model performed the best: I trained a model with four dense layers, 1 input
layer, 2 hidden layers and 1 output layer, with ‘relu’ and ‘softmax’ activations. The layers
have 5, 3, 3, and 2 neurons. We used 400 epochs with early stopping and with
patience=10. The MSE is computed on the validation samples. I also used learning
curve graphs to check if the validation loss got close to the training loss. The accuracy
score was 0.9848 and MSE=0.0152.

[Home](http://piringer.github.io/heartdisease/index)

[Introduction](http://piringer.github.io/heartdisease/intro)

[Data Description](http://piringer.github.io/heartdisease/Project.pdf)

[Data Exploration and Preparation](http://piringer.github.io/heartdisease/exploration)

[ML Model Training](http://piringer.github.io/heartdisease/models)

[Results and Discussion](http://piringer.github.io/heartdisease/results)

[User Interface](http://piringer.github.io/heartdisease/ui)

[Jupyter Notebook](https://github.com/piringer/heartdisease/blob/main/australian2.ipynb)

[Walkthrough Video](https://www.youtube.com/watch?v=18eQWJJu3tA)

[Web App](http://ec2-52-54-129-72.compute-1.amazonaws.com:8501/)

