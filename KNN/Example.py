from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data 
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

kNN = KNN(1)
kNN.train(X_train, y_train)
res = kNN.predict(X_test)

print('Real--->Predicted')

for i, val in enumerate(y_test):
    print('  %d ---> %d' % (val, res[i]))

print('预测准确率：')
print(kNN.score(X_test, y_test))

