from sklearn.model_selection import LeaveOneOut, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = "horse_data23.txt"

df = pd.read_table(filename)

df_trans = df.copy()

df_trans.loc[df_trans.lameLeg == "left:hind", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "right:fore", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "left:fore", "lameLeg"] = "left:right"
df_trans.loc[df_trans.lameLeg == "right:hind", "lameLeg"] = "left:right"

y = df_trans["lameLeg"]

X1 = df_trans[["pc3","pc4","A","W"]]

logistic_regression = LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=5)

gkf = GroupKFold(n_splits=8)

total_acc1 = []
total_acc11 = []
true_pred = []
b_acc = []


# Loop through the indices of the samples
for train_index, test_index in gkf.split(X1,y,groups=df["horse"]):
    # Get the training and test data
    X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # logistic regression model
    logistic_regression.fit(X_train, y_train)

    #decision tree model
    decision_tree.fit(X_train, y_train)

    #baseline model
    unique, counts = np.unique(y_train, return_counts=True)
    baseline = unique[np.argmax(counts)]


    #logistic regression prediction
    y_pred = logistic_regression.predict(X_test)

    #decision tree prediction
    d_pred = decision_tree.predict(X_test)

    #baseline_predicition
    b_pred = (baseline == y_test) * 1

    true_pred = (y_pred == y_test)*1

    true_pred_d = (d_pred == y_test)*1

    # Append the prediction to the list
    for i in true_pred:
        total_acc1.append(i)

    for i in true_pred_d:
        total_acc11.append(i)

    for i in b_pred:
        b_acc.append(i)


print("model 1 logistic pred")
true_pred1 = np.array(total_acc1)
print((np.sum(true_pred1 == 1)) / (len(true_pred1)))

print("baseline acc")
b_acc = np.array(b_acc)
print((np.sum(b_acc == 1)) / (len(b_acc)))

print("model 1 decision tree pred")
true_pred11 = np.array(total_acc11)
print((np.sum(true_pred11 == 1)) / (len(true_pred11)))

#table for logistic versus decision tree model
table1 = [[sum((true_pred1 == 1) & (true_pred11 == 1)), sum((true_pred1 == 0) & (true_pred11 == 1))],
         [sum((true_pred1 == 1) & (true_pred11 == 0)), sum((true_pred1 == 0) & (true_pred11 == 0))]]


#table for baseline versus decision tree model
table2 = [[sum((b_acc == 1) & (true_pred11 == 1)), sum((b_acc == 0) & (true_pred11 == 1))],
         [sum((b_acc == 1) & (true_pred11 == 0)), sum((b_acc == 0) & (true_pred11 == 0))]]

#table for baseline versus decision tree model
table3 = [[sum((b_acc == 1) & (true_pred1 == 1)), sum((b_acc == 0) & (true_pred1 == 1))],
         [sum((b_acc == 1) & (true_pred1 == 0)), sum((b_acc == 0) & (true_pred1 == 0))]]


result1 = mcnemar(table1, exact=False)
result2 = mcnemar(table2, exact=False)
result3 = mcnemar(table3, exact=False)


print('p-value for Mcnemar test between logistic and tree model :', result1.pvalue)

print('p-value for Mcnemar test between baseline and tree model :', result2.pvalue)

print('p-value for Mcnemar test between logistic and baseline model :', result3.pvalue)



