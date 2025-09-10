# Adult-Income-Prediction-App

This application predicts whether an individual’s annual income is above or below $50K.  
The prediction is based on personal and demographic factors such as age, workclass, education, marital status, 
occupation, relationship, gender, capital gain, capital loss, weekly working hours, and native country.  

The dataset used in this project is the well-known UCI Adult Income dataset, which contains 48,842 records 
with 14 descriptive attributes and one target variable, *Income*. The target has two classes: "<=50K" and ">50K".  

In this study, multiple machine learning models were tested, including Logistic Regression, Support Vector Classifier, 
Decision Tree, Random Forest, Gaussian Naive Bayes, and XGBoost. After evaluation, the XGBoost Classifier achieved 
the highest accuracy of **87.32%**, and it was selected as the final model for making predictions in this app. 
