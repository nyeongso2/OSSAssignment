from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 당뇨병 데이터셋 로드
diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

# train/test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 데이터 크기 출력
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
r2 = metrics.r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"R2 Score: {r2:.4f}")
