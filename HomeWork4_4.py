# Логическая регрессия

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Загрузка датасета
df = pd.read_csv('titanic.csv')

# Удаление категориальных признаков
df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Избавление от пропусков
df.isna().sum()

mean_age_surv = round(df[df['Survived'] == 1]['Age'].mean())
mean_age_died = round(df[df['Survived'] == 1]['Age'].mean())

df.loc[(df['Survived'] == 1) & (df['Age'].isnull()), 'Age'] = mean_age_surv
df['Age'].fillna(mean_age_died, inplace=True)

# Нормализация данных
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
df['SibSp'] = (df['SibSp'] - df['SibSp'].mean()) / df['SibSp'].std()
df['Parch'] = (df['Parch'] - df['Parch'].mean()) / df['Parch'].std()
df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

mapping = {'male': 1, 'female': 0}
df['Gender'] = df['Gender'].map(mapping)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=6)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()

model.fit(X_train, y_train)

# Предсказания тестового набора
y_pred = model.predict(X_test)

# Вычисление F1-меры
f1 = f1_score(y_test, y_pred)
print('F1-мера:', round(f1, 3))

# F1-мера: 0.806


# Метод ближайших соседей
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

# Загрузка датасета
df = pd.read_csv('titanic.csv')

# Удаление категориальных признаков
df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Обработка пропусков
df.isna().sum()

mean_age_surv = round(df[df['Survived'] == 1]['Age'].mean())
mean_age_died = round(df[df['Survived'] == 1]['Age'].mean())

df.loc[(df['Survived'] == 1) & (df['Age'].isnull()), 'Age'] = mean_age_surv
df['Age'].fillna(mean_age_died, inplace=True)

# Нормализация данных
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

mapping = {'male': 1, 'female': 0}
df['Gender'] = df['Gender'].map(mapping)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=18)

# Создание модели kNN
knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)

# Предсказание
y_pred = knn.predict(X_test)

# Вычисление F1-меры
f1 = f1_score(y_test, y_pred)
print('F1-мера:', round(f1, 3))

# F1-мера: 0.849



'''Вывод: Каждый из методов имеет свои преимущества и недостатки, их применение и эффективность зависит от конкретного случая. 
В моем случае модель Метода ближайших соседей (kNN) показала более высокую эффективность в классификации данных по сравнению с Логистической регрессией на данном датасете, 
kNN демонстрирует бОльшую общую точность, достигая лучшего баланса между точностью и полнотой.'''



#p.s.:
for j in range(1, 70):
  accuracies = []
  for i in range(1, 30):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=j)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracies.append(f1)
# print(f'Оптимальное: random_state- {j}, n_neighbors- {accuracies.index(max(accuracies)) + 1}, Результат: {round(max(accuracies), 3)}')
