from sklearn.model_selection import LeaveOneOut, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
filename = "horse_data23.txt"

df = pd.read_table(filename)

df_trans = df.copy()

df_trans.loc[df_trans.lameLeg == "left:hind", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "right:fore", "lameLeg"] = "right:left"
df_trans.loc[df_trans.lameLeg == "left:fore", "lameLeg"] = "left:right"
df_trans.loc[df_trans.lameLeg == "right:hind", "lameLeg"] = "left:right"






y = df["lameLeg"]
y2 = df_trans["lameLeg"]
X1 = df[["pc3","pc4","A","W"]]
X2 = df[["A","W"]]

logistic_regression = LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=0)


total_acc1 = []
total_acc11 = []
total_acc2 = []
loo = LeaveOneOut()
true_pred = []
b_acc = []
gkf = GroupKFold(n_splits=8)



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
    baseline = max(y_train)


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


for train_index, test_index in gkf.split(X2,y,groups=df["horse"]):
    # Get the training and test data
    X_train, X_test = X2.iloc[train_index], X2.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # Fit the model on the training data
    logistic_regression.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = logistic_regression.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Append the prediction to the list

    true_pred = (y_pred == y_test)*1

    for i in true_pred:
        total_acc2.append(i)



print("model 1 logistic pred")
true_pred1 = np.array(total_acc1)
print((np.sum(true_pred1 == 1)) / (len(true_pred1)))

print("model 2 logistic pred")
true_pred2 = np.array(total_acc2)
print((np.sum(true_pred2 == 1)) / (len(true_pred2)))

print("baseline acc")
b_acc = np.array(b_acc)
print((np.sum(b_acc == 1)) / (len(b_acc)))

print("model 1 decision pred")
true_pred11 = np.array(total_acc11)
print((np.sum(true_pred11 == 1)) / (len(true_pred11)))


#table for the two logistic regression models
table = [[sum((true_pred1 == 1) & (true_pred2 == 1)), sum((true_pred1 == 0) & (true_pred2 == 1))],
         [sum((true_pred1 == 1) & (true_pred2 == 0)), sum((true_pred1 == 0) & (true_pred2 == 0))]]

#table for logistic versus decision tree model
table2 = [[sum((true_pred1 == 1) & (true_pred11 == 1)), sum((true_pred1 == 0) & (true_pred11 == 1))],
         [sum((true_pred1 == 1) & (true_pred11 == 0)), sum((true_pred1 == 0) & (true_pred11 == 0))]]


#table for baseline versus decision tree model
table3 = [[sum((b_acc == 1) & (true_pred11 == 1)), sum((b_acc == 0) & (true_pred11 == 1))],
         [sum((b_acc == 1) & (true_pred11 == 0)), sum((b_acc == 0) & (true_pred11 == 0))]]

result = mcnemar(table2, exact=True)

print('McNemar\'s test statistic:', result.statistic)
print('p-value:', result.pvalue)

