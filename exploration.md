## Data Exploration and Preparation

First, I got familiar with the dataset. I thoroughly read the survey documentation and
explored the variables of the dataset. After choosing the variables I planned to work
with, I started the data exploration with descriptive statistics and data visualization.
Graphs helped to find and understand relationships between the variables. We also
checked the distribution and skewness of the data. Boxplots are useful for detecting
outliers, histograms present distribution, and scatter plots show relationships between
two continuous variables.

Next I cleaned and prepared the data for the ML algorithms. I addressed the missing
values with imputation. Because the dataset is relatively small, I didn’t drop any missing
values, but for both categorical and continuous features, I used KNN Imputer. For the
categorical features, missing values were “nan”, while for the continuous data it was
zero. Therefore, I replaced zeros with “nan” for easy imputation. The categorical
features first were ordinal encoded then imputed and finally I got the labels back by
reversing ordinal encoding. After the continuous features were imputed too, I joined all
features back together into one dataframe. After I had no missing values, I continued
data exploration with the info(), describe(), and corr() methods. Since I wanted to predict
heart disease, I compared patients' data with and without heart attack using boxplots,
barplots, scatterplots and cat plots with the seaborn library. Finally, I scaled the
continuous data using the Standard Scaler from scikit-learn preprocessing library.

After I had a clean and preprocessed dataset, I was ready to move on to model building.
I created two dataframes: features and target. The target data frame contains only one
column, the feature I want to predict: “heart_attack”. The features dataframe has the
remaining 33 features as described above in the Data Description part.
Then, I split the data into training data and test data using train_test_split and 20% test
set size. I used the test set after the hyperparameters were fine-tuned.

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

