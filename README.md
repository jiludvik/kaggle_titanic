##Titanic Survival Prediction

This project contains code and notebooks for Kaggle Titanic challenge [https://www.kaggle.com/c/titanic]. 

My personal objective set for this challenge were to:
* Bridge my knowledge of data wrangling and machine learning libraries from R to Python
* Develop clean and reusable Python code supporting multiple prediction algorithms 
* Learn PyCharm

Python script does the following
* Module 1: Data loading, imputation, feature engineering, and basic data exploration 
* Module 2: Hyperparameter tuning of 5 algorithms (LGBMClassifier, DecisionTreeCLassifier, RandomForestClassifier, GradientBoostinGClassifier, XGBoost) using GridSearchCV:
* Module 3: Recursive Feature Elimination
* Module 4: Hyperparameter tuning using BayesSearchCV (alternative to module 2)
* Module 5: Training of a model combining 5 algorithms using VotingClassifier
* Module 6: Generate predictions from test data

Tested on MacOS Catalina with:
* Python 3.8
* Numpy 1.19.1
* Pandas 1.1.1
* Matplotlib 3.3.1
* Seaborn 0.10.1
* Scikit-Learn 0.23.2 
* Scikit-Optimize 0.8.1
* LightGBM 3.0.0 (pip install)
* XGBoost 1.2 (pip install)

