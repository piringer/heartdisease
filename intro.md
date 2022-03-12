## Heart Disease Prediction

One of the leading causes of death in the United States and around the world is heart disease and heart attacks (CDC, NCHS Data Brief, 2021). According to the CDC, in 2020, about 670 thousand people died from heart disease, and the numbers are growing every year. There are several well documented risk factors that doctors use to diagnose ischemic heart disease: cholesterol levels, high blood pressure, age, obesity, diabetes, angina, a type of chest pain, or ECG abnormalities. In addition to these risk factors, there are several more lifestyle related factors that are worth considering like physical activity, sleep length and quality, medications, vitamins, fruit, vegetables, meat consumption, smoking and alcohol use.

Taking advantage of the large amount of data available today and Machine Learning algorithms, it makes sense to use this technology in diagnostics. Prevention and early diagnosis can be crucial in such a deadly disease, especially since in many cases there are no physical symptoms before the first heart attack. The use of an effective ML model could aid doctors to identify high risk patients and start intervention early either with treatment or minor lifestyle changes.

The National Heart Foundation of Australia (N.H.F.O.) conducts a comprehensive health survey regularly to study the collected data and evaluate risk factors and additional lifestyle choices. I used the 1980 dataset: “Risk Factor Prevalence Study, 1980”. From the 169 columns of the dataset, I used 35 as features to predict a heart attack in a classification problem.

The project was conducted in this Jupyter Notebook with Python. For the web application, FastAPI provides the backend and Streamlit provides the frontend. Docker desktop app and Docker Hub is used to manage the images that are pushed to AWS to an EC2 instance which serves the ML model as microservices.

First, documentation and data files of the survey were downloaded from the Australian Data Archive. The next step was to explore the data, read through the survey, and select questions that might be relevant to predicting the risk of a heart attack. Then I visualized the different features and looked for correlations. After that, I needed to clean and prepare the data for the machine learning models. I imputed missing data, renamed attributes, and scaled continuous features. Next, I trained several ML models and compared their performance metrics. The models I compared are: Logistic Regression, SVC, Decision Tree Classifier, Random Forest Classifier, KNN Classifier, and ANN with tensorflow. After analyzing the results, I chose the best performing model and built a web application with microservices. The final step was to deploy the web application using a cloud service, AWS.

[Introduction](http://piringer.github.io/heartdisease/intro)

[Data Description](http://piringer.github.io/heartdisease/Project.pdf)

[Data Exploration and Preparation](http://piringer.github.io/heartdisease/exploration)

[ML Model Training](http://piringer.github.io/heartdisease/models)

[Results and Discussion](http://piringer.github.io/heartdisease/results)

[Jupyter Notebook](https://github.com/piringer/heartdisease/blob/main/australian2.ipynb)

[Walkthrough Video](https://www.youtube.com/watch?v=18eQWJJu3tA)

[Web App](http://ec2-52-54-129-72.compute-1.amazonaws.com:8501/)
