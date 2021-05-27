import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_classification_binary_4_days = pd.read_csv("binary_danger_window_4_days_df.csv", index_col="timestamp")

X = df_classification_binary_4_days.iloc[:, :-1]
y = df_classification_binary_4_days.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, shuffle=False)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Average CV score on the training set was: 0.979377515438679

Generation 1 - Current best internal CV score: 0.9499999198949958

# Generation 2 - Current best internal CV score: 0.9789751742215963

# Generation 3 - Current best internal CV score: 0.9789751742215963

# Generation 4 - Current best internal CV score: 0.979377515438679

# Generation 5 - Current best internal CV score: 0.979377515438679

# Best pipeline: DecisionTreeClassifier(input_matrix, criterion=gini, max_depth=10, min_samples_leaf=12, min_samples_split=9)
# Optimized Metric: precision
# Confusion Matrix:
# [[109449  18443]
#  [ 15323  10867]]
# Classification Report:
#               precision    recall  f1-score   support

#        False       0.88      0.86      0.87    127892
#         True       0.37      0.41      0.39     26190

#     accuracy                           0.78    154082
#    macro avg       0.62      0.64      0.63    154082
# weighted avg       0.79      0.78      0.79    154082

# TPOT:
# 0.3707608324803821

exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=12, min_samples_split=9)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
