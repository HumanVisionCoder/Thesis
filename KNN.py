import pandas as pd
import time 
from imblearn.over_sampling import RandomOverSampler

time_start = time.perf_counter()

dataset = pd.read_csv('REKE10JANdataset.csv')
X = dataset.iloc[:, 4:].values
y = dataset.iloc[:, 3].values

ovs = RandomOverSampler(random_state=42)
x_res, y_res = ovs.fit_resample(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2, n_jobs = -1)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("base accuracy = ", accuracy)


from sklearn.model_selection import GridSearchCV

grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
g_res = gs.fit(X_train, y_train)

best_score = g_res.best_score_
best_params = g_res.best_params_

print(best_score)
print(best_params)

import numpy as np
# use the best hyperparameters
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance',algorithm = 'brute',metric = 'manhattan')
knn.fit(X_train, y_train)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = x_res, y = y_res, cv = 10, n_jobs = -1)
true_accuracy = accuracies.mean()
st_deviation = accuracies.std()

from sklearn.metrics import classification_report

# get a prediction
y_hat = knn.predict(X_train)
y_knn = knn.predict(X_test)
print(classification_report(y_test, y_knn))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv =5)
print(scores)
true_accuracy = np.mean(scores)
print('true accuracy: ',true_accuracy)
st_deviation = accuracies.std()
print("standard deviation", st_deviation)

time_elapsed = (time.perf_counter() - time_start)
print ("%5.1f secs" % (time_elapsed))
