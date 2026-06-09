# 공부학습시간에 따른 점수 예측 모델

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 데이터
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([50, 60, 70, 80, 90])

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측값
y_pred = model.predict(X)

# 그래프
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, label='Regression Line')
plt.xlabel('Study Time')
plt.ylabel('Score')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# 예측
print("6시간 공부 시 예상 점수:", model.predict([[6]])[0])
