## ML Model Results and Discussion:

The hyperparameters were fine-tuned using grid search to find the best parameters.
Finding the best parameters to constrain the models helps prevent overfitting. Our goal
when training ML models is not to get the best performance metric scores, but to have a
model that generalizes well on new, unseen data. I also added some visuals to help
understand the results: tree plots and confusion matrices. I compared the performance
metrics (accuracy, precision, recall, MSE) which is summarized in the table below:

| Models | Accuracy | Precision | Recall | MSE |
| :----------- | :----- | :----: | :----: | :----: |
| Logistic Regression | 0.9822 | 0.9822 | 0.9822 |0.0152 |
| SVM Classifier | 0.9857 | 0.9857 | 0.9857 |0.0143|
| Decision Tree | 0.9848 | 0.9848 | 0.9848 |0.0152 |
| Random Forest | 0.9848 | 0.9848 | 0.9848 |0.0152 |
| KNN Classifier | 0.9848 | 0.9848 | 0.9848 |0.0152 |
| ANN | 0.9848 |  |  |0.0152 |

As we can see, the performance metric scores are identical or very similar in every
model. However, the confusion matrices are different and offer additional insight: I
chose the SVM Classifier model for the web application since it predicted the least
numbers of false negatives and false positives. I think when we try to predict a deadly
disease, it is important to keep the false negative numbers to a minimum. In this way,
we won’t miss those patients who might be in danger of having a heart attack.
While the scores were very similar, some of the models did not predict any positive
cases. I think this is due to the fact that the dataset contained only 108 positive cases in
the 5603 data points. Considering this relatively low number, I think most models had
difficulty learning to predict positive cases. We could probably improve this result by
combining several years of data into one dataset, and dropping some of the negative
cases so the ratio of positive and negative cases could improve.

| Models | True Positive | True Negative | False Positive | False Negative |
| :----------- | :----- | :----: | :----: | :----: |
| Logistic Regression | 4 | 1097 | 7 |13 |
| SVM Classifier | 4 | 1100 | 4 |13|
| Decision Tree | 1 | 1102 | 2 |16 |
| Random Forest | 0 | 1104 | 0 |17 |
| KNN Classifier | 1 |1103 | 1 |16 |
| ANN | 1 | 1103 | 1 |16 |

After I picked the best model for the web application, I used the pickle library to save the
model for FastAPI.

[Home](http://piringer.github.io/heartdisease/index)

[Introduction](http://piringer.github.io/heartdisease/intro)

[Data Description](http://piringer.github.io/heartdisease/Project.pdf)

[Data Exploration and Preparation](http://piringer.github.io/heartdisease/exploration)

[ML Model Training](http://piringer.github.io/heartdisease/models)

[Results and Discussion](http://piringer.github.io/heartdisease/results)

[User Interface](http://piringer.github.io/heartdisease/ui)

[Jupyter Notebook](https://github.com/piringer/heartdisease/blob/main/austral_nb.ipynb)

[Walkthrough Video](https://youtu.be/aUX2eIEG-tU)

[Web App](http://ec2-52-54-129-72.compute-1.amazonaws.com:8501/)




