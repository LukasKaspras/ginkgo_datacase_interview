import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df_classification_binary_4_days = pd.read_csv("binary_danger_window_4_days_df.csv", index_col="timestamp")

X = df_classification_binary_4_days.iloc[:, :-1]
y = df_classification_binary_4_days.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, shuffle=False)

# Average CV score on the training set was: 0.5013625929572856
#
# Evaluation on the test set gave:
#
# Confusion Matrix:
# [[110625  17267]
# [ 19106   7084]]
#Classification Report:
#              precision    recall  f1-score   support
#
#       False       0.85      0.86      0.86    127892
#        True       0.29      0.27      0.28     26190
#
#    accuracy                           0.76    154082
#   macro avg       0.57      0.57      0.57    154082
#weighted avg       0.76      0.76      0.76    154082
#
#TPOT:
#0.2803268633386755

exported_pipeline = KNeighborsClassifier(n_neighbors=100, p=1, weights="uniform")
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
