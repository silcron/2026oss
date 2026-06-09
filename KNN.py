from sklearn.neighbors import KNeighborsClassifier

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

prediction = model.predict([[4]])

print("예측 결과:", prediction[0])