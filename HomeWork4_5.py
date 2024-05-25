from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=18)


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2],
    'min_samples_split': range(2, 11)
}

tree_model = DecisionTreeClassifier()

grid_search = GridSearchCV(tree_model, param_grid, cv=5)

grid_search.fit(X_train, y_train)

# Получение лучшей модели и ее оценки accuracy
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_accuracy = accuracy_score(y_test, best_model.predict(X_test))

# Вывод результатов
print("Лучшие параметры:")
print(best_params)
print("Лучшая точность:")
print(best_accuracy)

'''
Лучшие параметры:
{'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 3}
Лучшая точность:
0.9666666666666667
'''
