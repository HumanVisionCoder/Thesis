import pandas as pd
import time 
from imblearn.over_sampling import RandomOverSampler

time_start = time.perf_counter()

dataset = pd.read_csv('SUJU11JULdataset.csv')
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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("base accuracy = ", accuracy)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

true_accuracy = accuracies.mean()
st_deviation = accuracies.std()

from sklearn.model_selection import GridSearchCV

hyper_params = [{'C': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]}]
        
grid_search = GridSearchCV(estimator = classifier, param_grid = hyper_params, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_
best_params = grid_search.best_params_

from sklearn.metrics import classification_report
creport_rfc = classification_report(y_pred, y_test)
print(creport_rfc)

print(true_accuracy)
print(st_deviation)
print(best_score)
print(best_params)

time_elapsed = (time.perf_counter() - time_start)
print ("%5.1f secs" % (time_elapsed))
