# TITANIC ML - Final Version With All Options Tested


#%% MODULE 1. Data Loading, Cleansing, Feature Engineering and Exploration
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#  Step 1.1 Read test & training data

train_data = pd.read_csv(Path().joinpath('data', 'train.csv'))
train_data.name='train_data'
test_data = pd.read_csv(Path().joinpath('data', 'test.csv'))
test_data.name='test_data'
combined_data= train_data.append(test_data) #appended data frames
all_datasets = [train_data, test_data] #list with two data frames

#  Step 1.2 Look at data structure and summarise missing values- OPTIONAL
for dataset in [train_data,test_data]:
    print(dataset.name)
    print(dataset.shape)  # Dimensions
    print(dataset.dtypes)  # Data types
    print(dataset.isnull().sum())
    print()
# most missing are these are Cabin, Age, Embarked
#Considering more people have not survived, we will stick to ROC_AUC that is better at handling unbalanced data sets

#  Step 1.3 Impute missing values - Stage 1: Embarked, Fare, Cabin

for dataset in all_datasets:
    dataset.Embarked.fillna('C', inplace=True)
    dataset['Fare'] = SimpleImputer(strategy='median').fit_transform(dataset[['Fare']])
    random_cabinfloor_vector = pd.Series(np.random.choice(['F','G'], size=len(dataset['Cabin'].index)))
    dataset['Cabin'] = dataset['Cabin'].fillna(random_cabinfloor_vector)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['PeopleInTicket'] = dataset['Ticket'].map(combined_data['Ticket'].value_counts())
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major'], 'Officer')
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Don', 'Dr', 'Rev',
                                                 'Sir', 'Jonkheer', 'Dona'], 'Lady/Sir')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Ms','Mme'], 'Mrs')
    dataset.loc[(dataset.Title == 'Miss') & (dataset.Parch != 0) & (dataset.PeopleInTicket > 1), 'Title'] = "FemaleChild"
    dataset['Deck'] = dataset['Cabin'].str.findall('[a-zA-Z]').apply(lambda x: min(x))
    dataset['GroupSize'] = dataset[['FamilySize', 'PeopleInTicket']].max(axis=1)
    print(dataset.name)
    print(dataset[['Embarked','Fare','Cabin']].isnull().sum())
    print()

#  Step 1.4 Create new features: FamilySize, IsAlone, PeopleInTicket, Title and GroupSize - MANDATORY
# Used largely approach from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
for dataset in all_datasets:
    print(dataset.name)
    for col in dataset[['FamilySize', 'GroupSize', 'IsAlone','PeopleInTicket','Title', 'Deck']]:
        print(col,': ', dataset[col].unique())
    print()

# Step 1.5 Impute missing values - Stage 2: Age using mean age for corresponding Pclass, Sex and Title - MANDATORY
# Reused code from https://www.kaggle.com/allohvk/titanic-missing-age-imputation-tutorial-advanced

grp = train_data.groupby(['Pclass','Sex','Title'])['Age'].mean().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
def fill_age(x):
    return grp[(grp.Pclass==x.Pclass)&(grp.Sex==x.Sex)&(grp.Title==x.Title)]['Age'].values[0]

train_data['Age'], test_data['Age'] = [df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1) for df in [train_data, test_data]]
# The above would beenfit from rewriting into normal for cycle

print(train_data['Age'].isnull().sum())
print(test_data['Age'].isnull().sum())

# Step 1.6 Generate Farebin and AgeBin

# (as per https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
for dataset in all_datasets: #generate FareBin and AgeBin
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=(1, 2, 3, 4)).astype(int)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5, labels=(1, 2, 3, 4, 5)).astype(int)

#  Step 1.7 Map categorical to numerical variables - MANDATORY

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "FemaleChild": 5, "Officer":6, "Lady/Sir": 7}
cabinfloor_mapping={"T": 7, "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}

for dataset in all_datasets:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    dataset['Deck'] = dataset['Deck'].map(cabinfloor_mapping)
    print(dataset.name)
    print(dataset.dtypes)
    for col in dataset[['Sex','Title','Embarked','Deck']]:
        print(col,':', dataset[col].unique())
    print()

#  Step 1.8 Feature Selection

drop_elements_train = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'PeopleInTicket', 'FamilySize', 'Fare', 'Age']
drop_elements_test = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'PeopleInTicket', 'FamilySize', 'Fare', 'Age']

train_data = train_data.drop(drop_elements_train, axis=1)
test_data = test_data.drop(drop_elements_test, axis=1)

#  Step 1.9 Plot feature correlation heatmap

def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train_data)
plt.show()

# variables with highest correlation with Survived are Sex, Title, Pclass and Fare
# variables with lowest correlation with Survived are Age, Embarked, FamilySize
# Covariates (medium strength): Sex-Title & Fare-Pclass

# %% MODULE 2. Fit models, cross-validate and tune hyperparameters using GridSearchCV - MANDATORY

# Step 2.1. Load libraries for cross-validation and hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV

# Step 2.2 Initialise variables and constants
# Define random seed
my_rand_state = 0

# Split training data set to predictor (X) and response (y)
X = train_data.drop(['Survived'], axis=1)
y = np.ravel(train_data[['Survived']])

# Define classifiers to be explored
classifiers = [
    LGBMClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
]
# Define scoring method
scoring_method = 'roc_auc' # works better for unbalanced data set

# Define cross-validation strategy
n_splits=10
n_repeats=5
cv_method = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=my_rand_state)

# Initialise an empty dataframe to store cross-validation results
cv_score_cols = ['Classifier', 'Test Score Mean', 'Test Score 2xStd', 'Time', 'Parameters', 'CV Method']
cv_score = pd.DataFrame(columns=cv_score_cols)

# initialise a list to store trained model
grid_searchs = []

# Initialise an empty dataframe to store results of hyperparameter tuning
gridsearch_score_cols = ['Classifier', 'Best Score', 'Best Score 2xStd', 'CV Score Diff', 'Time', 'Parameters', 'CV Method']
gridsearch_score = pd.DataFrame(columns=gridsearch_score_cols)

# Initialise data frame to store predictions
predictions=pd.DataFrame()

# Step 2.3 Define hyperparameters - comments include values tested during tuning

# LGBMClassifier : https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
#params_lgbm1 =[{'num_leaves':range(2,7), 'max_depth':range(1,7), 'random_state': [my_rand_state]}]
# result: num_leaves=5 max_depth=5 +0.011921
#params_lgbm2 =[{'num_leaves':[5], 'max_depth':[5], 'min_data_in_leaf': range(1,100,5), 'random_state': [my_rand_state]}]
# result: mindatainleaf=21: +0.011704
#params_lgbm3 =[{'num_leaves':[5], 'max_depth':[5], 'min_data_in_leaf': [21], 'learning_rate': np.linspace(0.01,0.5,20, endpoint=True), 'random_state': [my_rand_state]}]
# result: learning_rate=0.08736 +0.012752
# Best parameters

# DecisionTreeClassifier: # #https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use and
# #https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

# params_dtc1 = [{'max_depth': range(3,20), 'random_state': [my_rand_state]}]
# max_depth=4 : +0.096423
# params_dtc2 = [{'max_depth': [4], 'min_samples_leaf':range(1,10), 'min_samples_split' : np.linspace(0.01, 0.21, 20, endpoint=True), 'random_state': [my_rand_state]}]
# min_samples_leaf=1, min_samples_split=0.031  +0.1005 -
# params_dtc3 = [{'max_depth': [4], 'min_samples_leaf':[1], 'min_samples_split' : [0.031], 'max_features': list(range(1,X.shape[1])), 'random_state': [my_rand_state]}]
# max_features=8 +0.100883
# Best parameters

# RandomForestClassifier: #https://scikit-learn.org/stable/modules/ensemble.html#parameters and
# #https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
#params_rfc1 = [{'n_estimators': range(10,300,5), 'random_state': [my_rand_state]}]
#n_estimators=220: +0.0002
#params_rfc2 = [{'n_estimators': [220], 'max_features':['sqrt',None], 'max_depth': range(1,10), 'random_state': [my_rand_state]}]
#max_features=sqrt +0.0002 - ignore
#params_rfc3 = [{'n_estimators': [220], 'max_depth': range(1,10), 'random_state': [my_rand_state]}]
#max_depth=7 +0.020789
#params_rfc4 = [{'n_estimators': [220], 'max_depth': [7], 'min_samples_split': np.linspace(0.01, 0.31, 30, endpoint=True), 'random_state': [my_rand_state]}]
#min_samples_split=0.030689 +0.022182
#params_rfc5 = [{'n_estimators': [220], 'max_depth': [7], 'min_samples_split':[0.030689] , 'min_samples_leaf': np.linspace(0.001, 0.02, 20, endpoint=True), 'random_state': [my_rand_state]}]
#'min_samples_leaf=0.002 +0.022927

# GradienBoostingClassifier: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
#params_gbc1 = [{'learning_rate': np.linspace(0.1, 0.11, 10, endpoint=False), 'random_state': [my_rand_state]}]
# learning_rate default
#params_gbc2 = [{'n_estimators': range(1,30), 'random_state': [my_rand_state]}]
# n_estimators=19, +0.004325
#params_gbc3 = [{'n_estimators': [19], 'max_depth':range(1, 20), 'random_state': [my_rand_state]}]
# max_depth= 4: +0.006144
#params_gbc4 = [{'n_estimators': [19], 'max_depth':[4], 'min_samples_split' :range(50,150) , 'random_state': [my_rand_state]}]
# min_samples_split: 129 +0.011524
#params_gbc5 = [{'n_estimators': [19], 'max_depth':[4], 'min_samples_split' :[129] , 'max_features': ['sqrt','log2',None], 'random_state': [my_rand_state]}]
# max_features=None - default
#params_gbc6 = [{'n_estimators': [19], 'max_depth':[4], 'min_samples_split' :[129], 'min_samples_leaf':range(1,100), 'random_state': [my_rand_state]}]
# min_samples_leaf=1 - default
# Best parameters +0.011524

#XGBClassifier: https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
#params_xgbc1 = [{'max_depth': range(1,100), 'random_state': [my_rand_state]}]
#max_depth=2  +0.00216
#params_xgbc2 = [{'max_depth': [2], 'min_child_weight': range(2,30), 'random_state': [my_rand_state]}]
# max_depth=4 +0.000459 worse than xgbc1
#params_xgbc3 = [{'max_depth': [2], 'subsample': np.linspace(0.5, 1.0, 20, endpoint=True), 'colsample_bytree' : np.linspace(0.5, 1.0, 20, endpoint=True), 'random_state': [my_rand_state]}]
# subsample=0.89474 colsample_bytree=0.89473 +0.00351
#params_xgbc4 = [{'max_depth': [2], 'subsample': [0.89474], 'colsample_bytree' : [0.89473], 'learning_rate' : np.linspace(0.01, 0.21, 20, endpoint=True), 'random_state': [my_rand_state]}]
# learning_rate=0.09421 +0.00348 - worse than xgbc3

# Define tuned hyperparameters (based on the above tuning)
params_lgbm =[{'num_leaves':[5], 'max_depth':[5], 'min_data_in_leaf': [21], 'learning_rate': [0.087368], 'random_state': [my_rand_state]}]
params_dtc = [{'max_depth': [4], 'min_samples_leaf':[1], 'min_samples_split' : [0.031], 'random_state': [my_rand_state]}]
params_rfc = [{'n_estimators': [220], 'max_depth': [7], 'min_samples_split':[0.030689] , 'min_samples_leaf': [0.002], 'random_state': [my_rand_state]}]
params_gbc = [{'n_estimators': [19], 'max_depth':[4], 'min_samples_split' :[129], 'random_state': [my_rand_state]}]
params_xgbc = [{'max_depth': [2], 'subsample': [0.89474], 'colsample_bytree' : [0.89473], 'random_state': [my_rand_state]}]

parameters = [
        params_lgbm,
        params_dtc,
        params_rfc,
        params_gbc,
        params_xgbc
]

# Step 2.4 Cross-validate, tune model hyperparameters using GridSearchCV and fit modelss
i=-1
for model in classifiers:
    i=i+1
    name = model.__class__.__name__
    print('Processing:', name,'...')
    print('Step 1. Cross-Validation')
    cv_result = cross_validate(model, X, y, cv=cv_method, scoring=scoring_method, n_jobs=-1)
    cv_result_summary = pd.DataFrame([[name,
                                       cv_result['test_score'].mean(),
                                       cv_result['test_score'].std()*2,
                                       cv_result['fit_time'].mean(),
                                       str(model.get_params()),
                                       cv_method]],
                                      columns=cv_score_cols)
    cv_score = cv_score.append(cv_result_summary, ignore_index=True)
    print('\nStep 2. Tuning Hyperparameters & Fitting The Best Model')
    grid_searchs.append(GridSearchCV(estimator=classifiers[i],
                                     param_grid=parameters[i],
                                     scoring=scoring_method,
                                     cv=cv_method,
                                     n_jobs=-1))
    grid_searchs[i].fit(X, y)
    corresponding_cv_score= cv_score.loc[cv_score['Classifier'] == name, 'Test Score Mean'].values[0]
    gridsearch_result = pd.DataFrame([[name,
                                       grid_searchs[i].best_score_,
                                       grid_searchs[i].cv_results_['std_test_score'][grid_searchs[i].best_index_]*2,
                                       grid_searchs[i].best_score_ - corresponding_cv_score,
                                       grid_searchs[i].refit_time_,
                                       grid_searchs[i].best_params_,
                                       cv_method]],
                                columns=gridsearch_score_cols)
    gridsearch_score = gridsearch_score.append(gridsearch_result, ignore_index=True)
    print('\nStep 3. Generating Predictions From Training Data')
    predictions[name]= grid_searchs[i].predict(X)

print()
print('Cross-Validation Summary')
print(cv_score.iloc[:, :4])
print('\nHyperparameter Tuning Summary')
print(gridsearch_score.iloc[:, :4])
print('\nTraining Data Survived Prediction Count')
print(predictions.sum(),'\n')

#Step 2.5 Feature correlation heatmap using predictions
correlation_heatmap(predictions)
plt.show()

#RESULTS
#                   Classifier  Best Score  Best Score 2xStd  CV Score Diff
#0              LGBMClassifier    0.876326          0.072827       0.020350
#1      DecisionTreeClassifier    0.872506          0.075281       0.114143
#2      RandomForestClassifier    0.874327          0.074220       0.024006
#3  GradientBoostingClassifier    0.883137          0.068900       0.011818
#4               XGBClassifier    0.872381          0.078163       0.035441

# Best algorithm is GradientBoostingClassifier with ROC_AUC Score 0.882798 and 2*score std 0.069156
# All other algorithms are pretty close behind

#%% MODULE 3. Feature Elimination Using Recursive Feature Elimination
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
# https://stats.stackexchange.com/questions/359553/choice-of-hyper-parameters-for-recursive-feature-elimination-svm/360667
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

best_features=[]

for g in grid_searchs:
    model=g.best_estimator_
    name = model.__class__.__name__
    rfecv = RFECV(estimator=DecisionTreeClassifier(), scoring=scoring_method) #, cv=cv_method)
    pipeline = Pipeline(steps=[('s', rfecv), ('m', model)])
    # Evaluate model
    cv_score = cross_val_score(pipeline, X, y, scoring=scoring_method, cv=cv_method, n_jobs=-1, error_score='raise')
    # Report on performance
    mean_cv_score=np.mean(cv_score)
    std2_cv_score=np.std(cv_score)*2
    print("Model:",model)
    print('CV Score (Mean):', mean_cv_score)
    print('CV Score (2*Std):', std2_cv_score)
    print ('CV Score(PostRFE)- Score (Pre-RFE):', mean_cv_score-g.best_score_)

# Most of models tuned with RFE seem to have worse cv score than models without tuning.
# We could probably fix this with further model tuning but that looks like too much work for little gain

# %% MODULE 4. Fit models, cross-validate and tune hyperparameters using BayesSearchCV
# This is an alternative to MODULE 3 based on BayesSearchCV
# https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/

# Step 4.1 Load libraries for cross-validation and hyperparameter tuning

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


# Step 4.2 Initialise variables and constants
# Define random seed
my_rand_state = 0

# Split training data set to predictor (X) and response (y)
X = train_data.drop(['Survived'], axis=1)
y = np.ravel(train_data[['Survived']])

# Define classifiers to be explored
classifiers = [
     LGBMClassifier(),
     DecisionTreeClassifier(),
     RandomForestClassifier(),
     GradientBoostingClassifier(),
     XGBClassifier()
]
# Define scoring method
scoring_method = 'roc_auc' # works better for unbalanced data set

# Define cross-validation strategy
n_splits=10
n_repeats=3
cv_method = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=my_rand_state)

# Initialise an empty dataframe to store cross-validation results
cv_score_cols = ['Classifier', 'Test Score Mean', 'Test Score 2xStd', 'Time', 'Parameters', 'CV Method']
cv_score = pd.DataFrame(columns=cv_score_cols)

# initialise a list to store trained model
bayes_searchs = []

# Initialise an empty dataframe to store results of hyperparameter tuning
bayessearch_score_cols = ['Classifier', 'Best Score', 'Best Score 2xStd', 'CV Score Diff', 'Parameters', 'CV Method']
bayessearch_score = pd.DataFrame(columns=bayessearch_score_cols)

# Initialise data frame to store predictions
predictions=pd.DataFrame()

# Step 4.3 Define Search Space for BayesSearchCV

# LGBMClassifier : https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
params_lgbm={ 'num_leaves': Integer(2,100, prior='log-uniform'),
              'max_depth' :Integer(2,train_data.shape[1]),
              'min_data_in_leaf' : Integer(2,100, prior='log-uniform'),
              'learning_rate': Real(1e-3, 1, prior='log-uniform'),
              'boosting_type': Categorical(['gbdt', 'dart']),
              'random_state': [my_rand_state]
  }

# DecisionTreeClassifier: # #https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use and
# #https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3

params_dtc = {
    'criterion': Categorical(['gini', 'entropy']),
    'max_depth' :Integer(2,50, prior='log-uniform'),
    'min_samples_leaf':Real(0.1,0.5),
    'min_samples_split' : Real(0.1,1),
    'random_state': [my_rand_state]
}

# RandomForestClassifier: #https://scikit-learn.org/stable/modules/ensemble.html#parameters and
# #https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
params_rfc = {
    'n_estimators': Integer(2,300 ),
    'max_depth': Integer(2,30, prior='log-uniform'),
    'min_samples_leaf': Real(0.1, 0.5, prior='log-uniform'),
    'min_samples_split': Real(0.001, 0.5, prior='log-uniform'),
    'random_state': [my_rand_state]
}
# GradienBoostingClassifier: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
params_gbc = {
    'n_estimators': Integer(2, 50),
    'max_depth': Integer(2,30, prior='log-uniform'),
    'min_samples_leaf': Real(0.1, 0.5),
    'min_samples_split': Real(0.001, 0.5),
    'random_state': [my_rand_state]
}
#XGBClassifier: https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
params_xgbc={
    'max_depth' :Integer(1,100, prior='log-uniform'),
    'min_child_weight' : Integer(2,50),
    'subsample': Real(0.1, 1),
    'colsample_bytree': Real(0.1,1),
    'learning_rate': Real(1e-3, 1),
    'random_state': [my_rand_state]
}

parameters = [
        params_lgbm,
        params_dtc,
        params_rfc,
        params_gbc,
        params_xgbc
]

# Step 4.4. Cross-validate, tune model hyperparameters, fit models uing BayesSearchCV
i=-1 #index
for model in classifiers:
    i=i+1
    name = model.__class__.__name__
    print('Processing:', name,'...')
    print('Step 1. Cross-Validation')
    cv_result = cross_validate(model, X, y, cv=cv_method, scoring=scoring_method, n_jobs=-1)
    cv_result_summary = pd.DataFrame([[name,
#                                      scoring_method,
                                       cv_result['test_score'].mean(),
                                       cv_result['test_score'].std()*2,
                                       cv_result['fit_time'].mean(),
                                       str(model.get_params()),
                                       cv_method]],
                                      columns=cv_score_cols)
    cv_score = cv_score.append(cv_result_summary, ignore_index=True)
    print('\nStep 2. Tuning Hyperparameters & Fitting The Best Model')
    bayes_searchs.append(BayesSearchCV(estimator=classifiers[i],
                                     search_spaces=parameters[i],
                                     scoring=scoring_method,
                                     cv=cv_method,
                                     n_jobs=-1))
    bayes_searchs[i].fit(X, y)
    corresponding_cv_score= cv_score.loc[cv_score['Classifier'] == name, 'Test Score Mean'].values[0]
    bayessearch_result = pd.DataFrame([[name,
                                       bayes_searchs[i].best_score_,
                                       bayes_searchs[i].cv_results_['std_test_score'][bayes_searchs[i].best_index_]*2,
                                       bayes_searchs[i].best_score_ - corresponding_cv_score,
                                       bayes_searchs[i].best_params_,
                                       cv_method]],
                                columns=bayessearch_score_cols)
    bayessearch_score = bayessearch_score.append(bayessearch_result, ignore_index=True)
    print('\nStep 3. Generating Predictions From Training Data')
    predictions[name]= bayes_searchs[i].predict(X)

print()
print('Cross-Validation Summary')
print(cv_score.iloc[:, :4])
print('\nHyperparameter Tuning Summary')
print(bayessearch_score.iloc[:, :4])
print('\nTraining Data Survived Prediction Count')
print(predictions.sum(),'\n')

#RESULTS
#                   Classifier  Best Score  Best Score 2xStd  CV Score Diff
#0              LGBMClassifier    0.875544          0.058218       0.017106
#1      DecisionTreeClassifier    0.851555          0.079246       0.088992
#2      RandomForestClassifier    0.848748          0.078220      -0.000494
#3  GradientBoostingClassifier    0.867934          0.079625      -0.004406
#4               XGBClassifier    0.877091          0.073681       0.037219

#In almost all  cases, hyperparameters tuned using BayesSearchCV seem to be generate worst performance than
# models with default parameters. We won't therefore proceed with this approach

#%%  MODULE 5. Combine models tuned using GridSearchCV using VotingClassifier
from sklearn.ensemble import VotingClassifier

vote_est = [
    ('lgbm', grid_searchs[0].best_estimator_),
    ('dtc', grid_searchs[1].best_estimator_),
    ('rfc', grid_searchs[2].best_estimator_),
    ('gbc', grid_searchs[3].best_estimator_),
    ('xgb', grid_searchs[4].best_estimator_)
]

# Initialise an empty dataframe to store cross-validation results
vote_cv_score_cols = ['Classifier', 'Voting Method', 'Test Score Mean', 'Test Score 2xStd', 'Time', 'CV Method']
vote_cv_score = pd.DataFrame(columns=vote_cv_score_cols)

vote=[]
i=-1
for voting_method in ['soft','hard']:
    i=i+1
    vote.append(VotingClassifier(estimators=vote_est, voting=voting_method, n_jobs=-1))
    vote_cv = cross_validate(vote[i], X, y, cv=cv_method,  n_jobs=-1)
    vote[i].fit(X, y)
    name = vote.__class__.__name__
    vote_summary = pd.DataFrame([[name,
                              voting_method,
                              vote_cv['test_score'].mean(),
                              vote_cv['test_score'].std()*2,
                              vote_cv['fit_time'].mean(),
                              cv_method]],
                            columns=vote_cv_score_cols)
    vote_cv_score = vote_cv_score.append(vote_summary, ignore_index=True)
    print(vote_summary.transpose())

print('Accuracy Using Voting Classifier\n',vote_cv_score.iloc[:, :4])

# Results of the combined algorithm are similar to the accuracy scores of individual algorithms,
# so there is little point using the VotingClassifier


#%% MODULE 6. Generate predictions using test data
X_test=test_data.drop(['PassengerId'], axis=1)

# Step 6.1 Generate predictions for 5 models fitted in MODULE 2
filename=['data/prediction_lgbm.csv',
          'data/prediction_dtc.csv',
          'data/prediction_rfc.csv',
          'data/prediction_gbc.csv',
          'data/prediction_xgb.csv']

for i in range(0, len(filename)-1):
    test_predictions=test_data[['PassengerId']]
    test_predictions = test_predictions.assign(Survived=pd.Series(grid_searchs[i].predict(X_test)))
    test_predictions.to_csv(filename[i],index=False)

# Step 6.2 Generate predictions for the Voting CLassifier fitted in MODULE 5
filename=['data/prediction_votesoft.csv',
          'data/prediction_votehard.csv']

for i in range(0, len(filename)-1):
    test_predictions=test_data[['PassengerId']]
    test_predictions = test_predictions.assign(Survived=pd.Series(vote[1].predict(X_test)))
    test_predictions.to_csv(filename[i],index=False)

#%% Kaggle Scores
# Random Forest CLassifier: 0.78708 #Leaderboard position 2726 - top 7%
# GradientBoostingClassifier: 0.77990
# XGBC: 0.77751
# VotingClassifier /5 algos soft voting: 0.77751
# LGBMClassifier: 0.77033
# Decision Tree Classifier: 0.76315
# Baseline: 0.76

