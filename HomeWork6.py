X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, c='b', marker='$*$', alpha=0.7, label= 'Train')
plt.scatter(X_test, y_test, c='g', marker='$*$', alpha=0.7, label= 'Test')
plt.plot(X_test, y_pred, color='r', label='Regression')
plt.legend()
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.grid(linestyle='--', alpha=0.7)
plt.show()
