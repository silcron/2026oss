from sklearn.ensemble import RandomForestClassifier

X = [[1, 50], [2, 60], [3, 70], [4, 80], [5, 90]]
y = [0, 0, 0, 1, 1]

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

prediction = model.predict([[4, 85]])

print("예측 결과:", prediction[0])