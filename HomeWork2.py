x = [1, 3, 7]
y = [2, 6, 14]
lr = 0.01
w1 = 0
w0 = 0
for j in range(250):  
  for i in range(len(x)):
    pred = w1 * x[i] + w0
    w1 += 2 * lr * x[i] * (y[i] - pred)
    w0 += 2 * lr * (y[i] - pred)

w1= round(w1, 2)
w0= round(w0, 2)
print(f'w1 ={w1}, w0= {w0}')
plt.scatter(x, y, c= 'green', marker='p', linewidth=5)
plt.plot([1,7], [w1 * x[0] + w0, w1 * x[2] + w0],linewidth=5)
plt.plot(x, y, c='r', linestyle='-.')   # с "настоящими" весами
plt.grid(linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#Вывод: веса получичились отличные и полностью соответствуют "Настоящим"
