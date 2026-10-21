import numpy as np

np.random.seed(42)

X = np.sort(10 * np.random.rand(1000, 1), axis=0)

y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

print(f"Кількість згенерованих значень X: {X.shape[0]}")
print(f"Кількість згенерованих значень y: {y.shape[0]}")
print(f"Перші 5 значень X:\n{X[:5]}")
print(f"Перші 5 значень y:\n{y[:5]}")

from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

print(f"Мін/Макс X до нормалізації: {X.min():.4f} / {X.max():.4f}")
print(f"Мін/Макс X після нормалізації: {X_normalized.min():.1f} / {X_normalized.max():.1f}")
print(f"Мін/Макс y до нормалізації: {y.min():.4f} / {y.max():.4f}")
print(f"Мін/Макс y після нормалізації: {y_normalized.min():.1f} / {y_normalized.max():.1f}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_normalized, test_size=0.2, random_state=42
)

print(f"Розмір навчальної вибірки: {X_train.shape[0]} записів")
print(f"Розмір тестової вибірки: {X_test.shape[0]} записів")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

k_values = [1, 3, 5, 7, 10, 15, 20, 50, 100]

mse_scores = []
r2_scores = []

print("Результати навчання KNN-регресора з різними значеннями K:")
print("-" * 50)

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"K = {k:<3} | MSE (Похибка): {mse:.4f} | R² (Точність): {r2:.4f}")

print("-" * 50)

best_index = np.argmin(mse_scores)
best_k = k_values[best_index]

print(f"Найкраще значення K: {best_k}")
print(f"Мінімальна середньоквадратична похибка (MSE): {mse_scores[best_index]:.4f}")
print(f"Найкращий коефіцієнт детермінації (R²): {r2_scores[best_index]:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_values, mse_scores, marker='o', linestyle='-', color='b')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Найкраще K = {best_k}')
plt.title('Залежність похибки (MSE) від кількості сусідів (K)')
plt.xlabel('Значення K')
plt.ylabel('Середньоквадратична похибка (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

X_plot = np.linspace(0, 1, 500).reshape(-1, 1)

best_knn = KNeighborsRegressor(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_plot_pred = best_knn.predict(X_plot)

plt.scatter(X_test, y_test, color='darkorange', label='Тестові дані', s=20, alpha=0.7)
plt.plot(X_plot, y_plot_pred, color='navy', label=f'Прогноз (K={best_k})', linewidth=2)

plt.title('Результат k-NN регресії (найкраще K)')
plt.xlabel('Нормалізований X')
plt.ylabel('Нормалізований y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()