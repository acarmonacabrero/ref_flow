from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from itertools import combinations
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, \
    precision_recall_curve, PrecisionRecallDisplay
import subprocess
from sklearn.ensemble import RandomForestClassifier
from math import floor
import matplotlib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import shap
from functions import parser_call, feature_filter
matplotlib.use('Agg')

#####                     #####
##### Analysis parameters #####
#####                     #####
run_id = 1  # This cannot be 0
n_iter = 2
sam_method = 'eRS'
n_gsa_runs = 15000  # GSA sample size
shap_sample_size = 2000
nruns = 1
threshold = 0
k = 10  # Number of folds


#####                   #####
##### Analysis settings #####
#####                   #####
data = pd.read_csv('agg_5_year_classification_data_for_analysis.csv')
country_year = pd.read_csv('country_existence.csv')

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
# TODO: increase this
params = {'n_estimators': [50, 100, 200, 300, 400, 500],
          'learning_rate': [0.0025, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          'max_depth': [2, 3, 4, 5, 6, 8, 10],
          'min_child_weight': [1, 3, 5, 7, 9, 11, 13, 15],
          'gamma': [0.0, 0.1, 0.5, 1, 5, 10, 15, 20],
          'reg_lambda': [0, 1, 3, 5, 7, 9, 11],
          'reg_alpha': [0, 1, 3, 5, 7, 9, 11],
          'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
          'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
          'scale_pos_weight': np.nan,  # TODO: change this parameter for each year
          # https://datascience.stackexchange.com/questions/16342/unbalanced-multiclass-data-with-xgboost  OR
          # https://datascience.stackexchange.com/questions/9488/xgboost-give-more-importance-to-recent-samples/9493#9493
          }


#####           #####
##### Functions #####
#####           #####


#####        #####
##### Parser #####
#####        #####
# study_year, lag, MLmodel = parser_call()
study_year, lag, MLmodel = 2016, 0, 'XGB'

#####                #####
##### Initialization #####
#####                #####
year_seed = 1004 + study_year  # seed for random state
feature_importance = []
scores = {}
# first_s_indices = pd.DataFrame()
# bias_si = pd.DataFrame()
# min_ci_si = pd.DataFrame()
# max_ci_si = pd.DataFrame()
# second_s_indices = pd.DataFrame()
# bias_sij = pd.DataFrame()
# min_ci_sij = pd.DataFrame()
# max_ci_sij = pd.DataFrame()

#####                  #####
##### Data Preparation #####
#####                  #####
# Filter variables by year and lag and slipt variables
X, y = feature_filter(data, country_year, study_year, lag=lag, prior_ref=False)  # Do not include prior refugee flow

# X, y = threshold_filter(X, y, threshold=log_transformation(threshold, transf))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=100)

# Save a copy of the train data for the explainer
data_ = X_train.copy()
data_['target'] = y_train
data_.to_csv('{}_data_lag{}.csv'.format(study_year, lag), index=False)
del data_
# TODO: add standarization here? If so, implement Pipelines
X = X.drop(['state_destination_name', 'state_origin_name', 'iid'], axis=1)
X_train = X_train.drop(['state_destination_name', 'state_origin_name', 'iid'], axis=1)
X_test = X_test.drop(['state_destination_name', 'state_origin_name', 'iid'], axis=1)

# Save a copy of the data for GSA
data_ = X.copy()
data_.to_csv('X.csv', index=False)
del data_

#####           #####
##### Model fit #####
#####           #####
params['scale_pos_weight'] = [y.value_counts()[1]/y.value_counts()[0]]
if MLmodel == 'XGB':
    r_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'),
                                  param_distributions=params,
                                  n_iter=n_iter,
                                  scoring='recall',  # Test for 'precision', 'recall', 'f1' and 'roc_auc'
                                  n_jobs=-1,
                                  cv=k,
                                  verbose=0,
                                  random_state=year_seed
                                  )
    # TODO: reconsider changing X, y below to X_train, y_train
    r_search.fit(X_train, y_train)
    model = r_search.best_estimator_
    best_params = r_search.best_params_
    test_probabilities = pd.DataFrame(model.predict_proba(X_test), columns=['No flow', 'Flow'])
    probabilities = test_probabilities.copy()
    probabilities['Y test'] = y_test.reset_index(drop=True)
    probabilities.to_csv('probabilities.csv', index=False)
elif MLmodel == 'RF':
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

#####              #####
##### SHAP and GSA #####
#####              #####
# Save model and explainer for SHAP figures
f = open('{}_model_lag{}.sav'.format(study_year, lag), 'wb')
pickle.dump(model, f)
f.close()
# SHAP values
background = shap.maskers.Independent(X_train, max_samples=shap_sample_size)
explainer = shap.Explainer(model.predict, background)
shap_values = explainer(X_train)
f = open('{}explainer_lag{}.exp'.format(study_year, lag), 'wb')
pickle.dump(shap_values, f)
f.close()


# Feature importance
feature_importance.append(dict(zip([feature[:-5] for feature in X.columns.to_list()],
                                   model.feature_importances_)))
feature_importance = pd.DataFrame(feature_importance, index=[study_year])
feature_importance.to_csv('f_imp_' + str(run_id) + '_' + str(study_year) + '.csv')


#####            #####
##### Statistics #####
#####            #####
# Scores
scores['Train accuracy'] = accuracy_score(y_train, model.predict(X_train))
scores['Test accuracy'] = accuracy_score(y_test, model.predict(X_test))
scores['CV accuracy'] = cross_val_score(model, X, y, cv=5, scoring='accuracy')
scores['Confusion matrix'] = confusion_matrix(y_test, model.predict(X_test))
scores['Classification report'] = classification_report(y_test, model.predict(X_test),
                                                        target_names=['No flow', 'Flow'])
fpr, tpr, thresholds = roc_curve(y_test, test_probabilities['Flow'])
scores['Flow FPR'] = fpr
scores['Flow TPR'] = tpr
scores['Flow ROC curve - thresholds'] = thresholds
scores['Flow ROC AUC'] = auc(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(y_test, test_probabilities['Flow'])
pr_display = PrecisionRecallDisplay.from_predictions(y_test, test_probabilities['Flow'], name=MLmodel)
_ = pr_display.ax_.set_title("Precision-Recall curve")
plt.savefig('precision_recall.png')
scores['Flow precision'] = precision
scores['Flow recall'] = recall
scores['Flow precision-recall - thresholds'] = thresholds

scores_series = pd.Series(scores)
scores_series.to_csv('scores.csv')

#####         #####
##### Figures #####
#####         #####
# SHAP plots
sample_ind = 1
for var in X_train.columns:
    shap.plots.partial_dependence(var, model.predict, X_train,
                                  ice=False,
                                  model_expected_value=True,
                                  feature_expected_value=True,
                                  show=False,
                                  shap_values=shap_values[sample_ind:sample_ind + 1, :],
                                  )
    plt.savefig('pd{}_{}_lag{}.png'.format(study_year, var, lag))

# ROC curve
plt.figure('roc auc')
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
plt.savefig('ROC_AUC.png')
