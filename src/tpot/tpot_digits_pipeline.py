import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

df_classification_binary_4_days = pd.read_csv("binary_danger_window_4_days_df.csv", index_col="timestamp")

X = df_classification_binary_4_days.iloc[:, :-1]
y = df_classification_binary_4_days.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, shuffle=False)

# Average CV score on the training set was: 0.7974530381406535


#Generation 1 - Current best internal CV score: 0.7969381596671167


#TPOT closed during evaluation in one generation.
#WARNING: TPOT may not provide a good pipeline if TPOT is stopped/interrupted in a early #generation.
#
#
#TPOT closed prematurely. Will use the current best pipeline.
#
#Best pipeline: ExtraTreesClassifier(MaxAbsScaler(input_matrix), bootstrap=False, #criterion=gini, max_features=0.1, min_samples_leaf=19, min_samples_split=2, #n_estimators=100)
#Confusion Matrix:
#[[127650    242]
# [ 25495    695]]
#Classification Report:
#              precision    recall  f1-score   support
#
#       False       0.83      1.00      0.91    127892
#        True       0.74      0.03      0.05     26190
#
#    accuracy                           0.83    154082
#   macro avg       0.79      0.51      0.48    154082
##weighted avg       0.82      0.83      0.76    154082
#
#TPOT:
#0.8329655637907089


exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.1, min_samples_leaf=19, min_samples_split=2, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
