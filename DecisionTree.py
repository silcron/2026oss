from sklearn.tree import DecisionTreeClassifier

# 공부시간, 출석률
X = [[1, 50], [2, 60], [3, 70], [4, 80], [5, 90]]
y = [0, 0, 0, 1, 1]  # 합격 여부

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([[4, 85]])

print("예측 결과:", prediction[0])