import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import seaborn as sns
from pathlib import Path

project_dir = str(Path(__file__).resolve().parents[2])

path = project_dir + "/data/processed/binary_danger_window_4_days_df.csv"

df_classification_binary_4_days = pd.read_csv(path, index_col=0)

X = df_classification_binary_4_days.iloc[:, :-1]
y = df_classification_binary_4_days.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, shuffle=False)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Average CV score on the training set was: 0.9751753365584482

# Generation 1 - Current best internal CV score: 0.9895154876863262

# Generation 2 - Current best internal CV score: 0.9895154876863262

# Generation 3 - Current best internal CV score: 0.9895154876863262

# Generation 4 - Current best internal CV score: 0.9895154876863262

# Generation 5 - Current best internal CV score: 0.9895154876863262

# Best pipeline: ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=gini, max_features=0.1, min_samples_leaf=19, min_samples_split=2, n_estimators=100)
# Optimized Metric: accuracy
# Confusion Matrix:
# [[127690    202]
#  [ 25511    679]]
# Classification Report:
#               precision    recall  f1-score   support

#        False       0.83      1.00      0.91    127892
#         True       0.77      0.03      0.05     26190

#     accuracy                           0.83    154082
#    macro avg       0.80      0.51      0.48    154082
# weighted avg       0.82      0.83      0.76    154082

# TPOT:
# 0.8331213250087616

classifier = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.1, min_samples_leaf=19, min_samples_split=2, n_estimators=100, random_state=42)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_mat.ravel()
cm = [[tn,fp],[fn,tp]]
print("Confusion Matrix:")
print(confusion_mat)
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:",)
print (classification_rep)


fig, ax = plt.subplots()
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.color": '#676C73', "ytick.color": '#676C73'})

heatmap = sns.heatmap(cm, annot=True, fmt = "d", cmap="YlGn", ax=ax)

fontdict = {'color':'#676C73'}

# labels, title and ticks
heatmap.set_xlabel('PREDICTED LABELS', fontdict= fontdict)
heatmap.set_ylabel('ACTUAL LABELS', fontdict= fontdict)
heatmap.set_title('Confusion Matrix', fontdict= { 'fontsize': 24, 'color':'#676C73'})
heatmap.xaxis.set_ticklabels(['No Danger', 'Danger'])
heatmap.yaxis.set_ticklabels(['No Danger', 'Danger'])

for tick in heatmap.get_xticklabels():
    tick.set_color("#676C73")

for tick in heatmap.get_yticklabels():
    tick.set_color("#676C73")

plt.savefig("ConfusionMatrix_binary_4_train30.png", dpi=900)