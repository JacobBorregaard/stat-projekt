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

y2 = df_trans["lameLeg"]

X2 = df_trans[["A","W"]]


y = df["lameLeg"]

X1 = df[["A","W"]]




logistic_regression = LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=5)

gkf = GroupKFold(n_splits=8)

total_acc1 = []
total_acc11 = []
true_pred = []
b_acc1 = []
w_acc_l = 0
w_acc_d = 0
w_acc_b = 0


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
    b_pred = (baseline == y_test)*1

    true_pred = (y_pred == y_test)*1

    true_pred_d = (d_pred == y_test)*1

    #accuracy score
    w_acc_l = w_acc_l + ((accuracy_score(y_test, y_pred)) * (len(y_test) / len(y)))
    w_acc_d = w_acc_d + ((accuracy_score(y_test, d_pred)) * (len(y_test) / len(y)))
    w_acc_b = w_acc_b + ((accuracy_score(y_test, b_pred)) * (len(y_test) / len(y)))


    # Append the prediction to the list
    for i in true_pred:
        total_acc1.append(i)

    for i in true_pred_d:
        total_acc11.append(i)

    for i in b_pred:
        b_acc1.append(i)




print("model 1 logistic pred")
true_pred1 = np.array(total_acc1)

print(w_acc_l)
print("baseline acc")

#print(w_acc_b)
print("model 1 decision tree pred")
true_pred11 = np.array(total_acc11)
print(w_acc_d)


X1 = df[["pc3","pc4"]]

logistic_regression = LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=5)

gkf = GroupKFold(n_splits=8)

total_acc2 = []
total_acc22 = []
true_pred = []
b_acc2 = []


# Loop through the indices of the samples
for train_index, test_index in gkf.split(X2,y2,groups=df["horse"]):
    # Get the training and test data
    X_train, X_test = X2.iloc[train_index], X2.iloc[test_index]
    y_train, y_test = y2.iloc[train_index], y2.iloc[test_index]

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
        total_acc2.append(i)

    for i in true_pred_d:
        total_acc22.append(i)

    for i in b_pred:
        b_acc2.append(i)


print("model 2 logistic pred")
true_pred2 = np.array(total_acc2)
print((np.sum(true_pred2 == 1)) / (len(true_pred2)))



print("model 2 decision tree pred")
true_pred22 = np.array(total_acc22)
print((np.sum(true_pred22 == 1)) / (len(true_pred22)))

true_pred33= np.array(total_acc22)
true_pred3= np.array(total_acc2)

#table for logistic1 versus decision tree model2
table1 = np.array([[sum((true_pred1 == 1) & (true_pred22 == 1)), sum((true_pred1 == 1) & (true_pred22 == 0))],
         [sum((true_pred1 == 0) & (true_pred22 == 1)), sum((true_pred1 == 0) & (true_pred22 == 0))]])

#table for logistic1 versus logistic2
table2 = np.array([[sum((true_pred1 == 1) & (true_pred2 == 1)), sum((true_pred1 == 1) & (true_pred2 == 0))],
         [sum((true_pred1 == 0) & (true_pred2 == 1)), sum((true_pred1 == 0) & (true_pred2 == 0))]])


#table for logistic1 versus decision tree model2
table3 = np.array([[sum((true_pred11 == 1) & (true_pred22 == 1)), sum((true_pred11 == 1) & (true_pred22 == 0))],
         [sum((true_pred11 == 0) & (true_pred22 == 1)), sum((true_pred11 == 0) & (true_pred22 == 0))]])

#table for logistic1 versus decision tree model2
table4 = np.array([[sum((true_pred11 == 1) & (true_pred2 == 1)), sum((true_pred11 == 1) & (true_pred2 == 0))],
         [sum((true_pred11 == 0) & (true_pred2 == 1)), sum((true_pred11 == 0) & (true_pred2 == 0))]])

table5 = np.array([[sum((true_pred3 == 1) & (true_pred33 == 1)), sum((true_pred3 == 1) & (true_pred33 == 0))],
         [sum((true_pred3 == 0) & (true_pred33 == 1)), sum((true_pred3 == 0) & (true_pred33 == 0))]])

print(table3)
print(table4)

result1 = mcnemar(table1, exact=False)
result2 = mcnemar(table2, exact=False)
result3 = mcnemar(table3, exact=False)
result4 = mcnemar(table4, exact=False)
result5 = mcnemar(table5, exact=False)
print('p-value for Mcnemar test between logistic and tree model :', result1.pvalue)

print('p-value for Mcnemar test between logistic and logistic:', result2.pvalue)

print('p-value for Mcnemar test between tree and tree model :', result3.pvalue)

print('p-value for Mcnemar test between tree and logistic model :', result4.pvalue)


print('p-value for Mcnemar test between baseline and baseline model :', result5.pvalue)


