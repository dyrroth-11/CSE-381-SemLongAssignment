
# **Importing Libraries**


import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# **Loading Dataset**"""

dataset = pd.read_csv('dataset_processed.csv')
dataset = dataset.dropna()

#Features & Output Split
X = dataset.iloc[:, 0: 10].values
y = dataset.iloc[:, 10: 16].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)      

for train, test in kfold.split(X, y[:, 0]):
  X_train = X[train]
  y_train = y[train, 0]
  X_test = X[test]
  y_test = y[test, 0]


#Normalization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)            
        
# Begin Classification
y_current_train = y_train
y_current_test =  y_test

"""# **1 - linear regression**"""

linear_classifier = LogisticRegression(random_state = random_state)
linear_classifier.fit(X_train, y_current_train)
cm_linear = pd.crosstab(y_current_test, linear_classifier.predict(X_test))
accuracy_linear = pd.DataFrame(classification_report(y_current_test, linear_classifier.predict(X_test), output_dict = True)).transpose() 
l1, l2, l3, l4 = score(y_current_test, linear_classifier.predict(X_test))

"""# **2 - KNN**"""

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_current_train)
cm_knn = pd.crosstab(y_current_test, knn_classifier.predict(X_test))
accuracy_knn = pd.DataFrame(classification_report(y_current_test, knn_classifier.predict(X_test), output_dict = True)).transpose() 
k1, k2, k3, k4 = score(y_current_test, knn_classifier.predict(X_test))

"""# **3 - SVM**"""

svm_classifier = SVC(kernel = 'linear', random_state = random_state)
svm_classifier.fit(X_train, y_current_train)
cm_svm = pd.crosstab(y_current_test, svm_classifier.predict(X_test))
accuracy_svm = pd.DataFrame(classification_report(y_current_test, svm_classifier.predict(X_test), output_dict = True)).transpose() 
s1, s2, s3, s4 = score(y_current_test, svm_classifier.predict(X_test))

"""# **4 - Kernel SVM**"""

kernel_svm_classifier = SVC(kernel = 'rbf', random_state = random_state)
kernel_svm_classifier.fit(X_train, y_current_train)
cm_kernel_svm = pd.crosstab(y_current_test, kernel_svm_classifier.predict(X_test))
accuracy_kernel_svm = pd.DataFrame(classification_report(y_current_test, kernel_svm_classifier.predict(X_test), output_dict = True)).transpose() 
sv1, sv2, sv3, sv4 = score(y_current_test, kernel_svm_classifier.predict(X_test))

"""# **5 - Naive Bayes**"""

naive_classifier = GaussianNB()
naive_classifier.fit(X_train, y_current_train)
cm_naive = pd.crosstab(y_current_test, naive_classifier.predict(X_test))
accuracy_naive =pd.DataFrame(classification_report(y_current_test, naive_classifier.predict(X_test), output_dict = True)).transpose() 
n1, n2, n3, n4 = score(y_current_test, naive_classifier.predict(X_test))

"""# **6 - Decision Tree**"""

decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = random_state)
decision_tree_classifier.fit(X_train, y_current_train)
cm_decision_tree = pd.crosstab(y_current_test, decision_tree_classifier.predict(X_test))
accuracy_decision_tree = pd.DataFrame(classification_report(y_current_test, decision_tree_classifier.predict(X_test), output_dict = True)).transpose() 
d1, d2, d3, d4 = score(y_current_test, decision_tree_classifier.predict(X_test))

"""# **7 - Random Forest**"""

random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = random_state)
random_forest_classifier.fit(X_train, y_current_train)
cm_random_forest = pd.crosstab(y_current_test, random_forest_classifier.predict(X_test))
accuracy_random_forest = pd.DataFrame(classification_report(y_current_test, random_forest_classifier.predict(X_test), output_dict = True)).transpose() 
r1, r2, r3, r4 = score(y_current_test, random_forest_classifier.predict(X_test))

"""# **8 - Stacked Deep learning**

**Model-1**
"""

mlp1 = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp1.fit(X_train,y_train)
y1_pred = mlp1.predict(X_test)

# print(classification_report(y_current_test, y1_pred, output_dict = True))

"""**Model-2**"""

mlp2 = MLPClassifier(hidden_layer_sizes=(8,16,8), activation='relu', solver='adam', max_iter=500)
mlp2.fit(X_train,y_train)
y2_pred = mlp2.predict(X_test)

# print(classification_report(y_current_test, y2_pred, output_dict = True))

"""**Model-3**"""

mlp3 = MLPClassifier(hidden_layer_sizes=(8,8,12), activation='relu', solver='adam', max_iter=500)
mlp3.fit(X_train,y_train)
y3_pred = mlp3.predict(X_test)

# print(classification_report(y_current_test, y3_pred, output_dict = True))

"""**Model-4**"""

mlp4 = MLPClassifier(hidden_layer_sizes=(8,16,16), activation='relu', solver='adam', max_iter=500)
mlp4.fit(X_train,y_train)
y4_pred = mlp4.predict(X_test)

# print(classification_report(y_current_test, y4_pred, output_dict = True))

"""**Model-5**"""

mlp5 = MLPClassifier(hidden_layer_sizes=(16,8,8), activation='relu', solver='adam', max_iter=500)
mlp5.fit(X_train,y_train)
y5_pred = mlp5.predict(X_test)

# print(classification_report(y_current_test, y5_pred, output_dict = True))

d11, d12, d13, d14 = score(y_current_test, y1_pred)
d21, d22, d23, d24 = score(y_current_test, y2_pred)
d31, d32, d33, d34 = score(y_current_test, y3_pred)
d41, d42, d43, d44 = score(y_current_test, y4_pred)
d51, d52, d53, d54 = score(y_current_test, y5_pred)

dd1 = (d11 + d21 + d31 + d41 + d51)/5
dd2 = (d12 + d22 + d32 + d42 + d52)/5
dd3 = (d13 + d23 + d33 + d43 + d53)/5
dd4 = (d14 + d24 + d34 + d44 + d54)/5

accuracy_deep = []

accuracy_deep.append(classification_report(y_current_test, y1_pred, output_dict = True))
accuracy_deep.append(classification_report(y_current_test, y2_pred, output_dict = True))
accuracy_deep.append(classification_report(y_current_test, y3_pred, output_dict = True))
accuracy_deep.append(classification_report(y_current_test, y4_pred, output_dict = True))
accuracy_deep.append(classification_report(y_current_test, y5_pred, output_dict = True))

accuracy_average = accuracy_deep[0]

for i in range(1,5):
  for x, y in accuracy_deep[i].items():
    if isinstance(y, dict):
      for a, b in y.items():
        accuracy_average[x][a]+=b
    else:
      accuracy_average[x]+=y

            
for x, y in accuracy_average.items():
  if isinstance(y, dict):
    for a, b in y.items():
      accuracy_average[x][a]/=5
  else:
    accuracy_average[x]/=5

accuracy_stacked_deep = pd.DataFrame(accuracy_average).transpose()

with open('Accuracies.csv', 'a') as cv_file:
  cv_file.write('\n\n{}\n'.format('Linear'))
  accuracy_linear.to_csv(cv_file, header=True)
  cv_file.write('\n\n{}\n'.format('Naive'))
  accuracy_naive.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('KNN'))
  accuracy_knn.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('SVM'))
  accuracy_svm.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('Kern SVM'))
  accuracy_kernel_svm.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('DT'))
  accuracy_decision_tree.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('RF'))
  accuracy_random_forest.to_csv(cv_file, header=False)
  cv_file.write('\n\n{}\n'.format('Stacked Deep learning'))
  accuracy_stacked_deep.to_csv(cv_file, header=False)

with open('all_scores.csv', 'a') as cv_file:
  cv_file.write('\n\n\nLinear   , ' + 'percision {} \nrecall {}\n {}\nSupport {}'.format(l1, l2, l3, l4))
  cv_file.write('\n\n\nNaive    , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(n1, n2, n3, n4))
  cv_file.write('\n\n\nKNN      , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(k1, k2, k3, k4))
  cv_file.write('\n\n\nSVM      , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(s1, s2, s3, s4))
  cv_file.write('\n\n\nKern SVM , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(sv1, sv2, sv3, sv4))
  cv_file.write('\n\n\nDecision Trees  , ' + 'percision {} \nrecall {}\n {}\nSupport {}'.format(d1, d2, d3, d4))
  cv_file.write('\n\n\nRandom Forest , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(r1, r2, r3, r4))
  cv_file.write('\n\n\nStacked Deep Learning , ' +  'percision {} \nrecall {}\n {}\nSupport {}'.format(dd1, dd2, dd3, dd4))
