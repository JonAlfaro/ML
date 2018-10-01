import json

Log_parameters = [
    {'C': [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
]

Tree_parameters = [
    {'max_depth': [1, 2, 3, 4, 5, 10, 100]},
    {'min_samples_split': [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]}
]

Forest_parameters = [
    {'n_estimators': [1, 10, 100, 200]},
    {'min_samples_split': [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]}
]

SVM_parameters = [
    {'kernel': ['linear', 'polynomial', 'rbf']},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
]

KNN_parameters = [
    {'p': [1, 2], 'n_neighbors': [1, 2 ,3 ,4 ,5 ]}
]


with open('classifiers/sk/Logistic Regression/parameters.json', 'w') as fp:
    json.dump(Log_parameters, fp)

with open('classifiers/sk/Decision Tree/parameters.json', 'w') as fp:
    json.dump(Tree_parameters, fp)

with open('classifiers/sk/Random Forest/parameters.json', 'w') as fp:
    json.dump(Forest_parameters, fp)

with open('classifiers/sk/Kernel SVM/parameters.json', 'w') as fp:
    json.dump(SVM_parameters, fp)

with open('classifiers/sk/Nearest Neighbor/parameters.json', 'w') as fp:
    json.dump(KNN_parameters, fp)