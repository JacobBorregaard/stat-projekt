from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd


filename = "horse_data23.txt"

df = pd.read_table(filename)

df_trans = df.copy()



df_trans.loc[df_trans.lameLeg == "left:hind", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "right:fore", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "left:fore", "lameLeg"] = "left:right"
df_trans.loc[df_trans.lameLeg == "right:hind", "lameLeg"] = "left:right"



y = df_trans["lameLeg"]
X = df[["pc3","pc4","A","W"]]


clf = DecisionTreeClassifier(random_state=0)
y_pred = cross_val_predict(clf, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)
scores = cross_val_score(clf, X, y, cv=5)
print(conf_mat)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

