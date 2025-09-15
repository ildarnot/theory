import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Исходные данные
data = {
    'gap': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    'surface_energy': [34, 44, 44, 44, 34, 44, 41, 39, 43, 38, 44],
    'roughness': [3.2, 3.1, 3.1, 3.3, 0.1, 2.6, 0.2, 1.8, 1.2, 0.25, 1.5],
    'curing_time': [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
    'break_force': [0, 9.2, 9.1, 9.7, 0, 9.2, 5, 3, 7, 9.2, 9.2]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Разделяем на признаки (X) и целевую переменную (y)
X = df[['gap', 'surface_energy', 'roughness', 'curing_time']].values
y = df['break_force'].values

# Добавляем столбец единиц для intercept (свободного члена)
X = np.column_stack((np.ones(len(X)), X))

# Вычисляем коэффициенты по методу наименьших квадратов: β = (X^T * X)^(-1) * X^T * y
XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(X.T, y)
beta = np.dot(XtX_inv, Xty)

# Предсказанные значения
y_pred = np.dot(X, beta)

# Остатки
residuals = y - y_pred

# Оценка качества модели
# Сумма квадратов остатков (SSE)
SSE = np.sum(residuals**2)
# Общая сумма квадратов (SST)
SST = np.sum((y - np.mean(y))**2)
# Коэффициент детерминации R^2
R_squared = 1 - (SSE / SST)

# Вывод результатов
print("Результаты регрессионного анализа:")
print("----------------------------------")
print(f"Intercept (β0): {beta[0]:.4f}")
print(f"Коэффициент для gap (β1): {beta[1]:.4f}")
print(f"Коэффициент для surface_energy (β2): {beta[2]:.4f}")
print(f"Коэффициент для roughness (β3): {beta[3]:.4f}")
print(f"Коэффициент для curing_time (β4): {beta[4]:.4f}")
print("----------------------------------")
print(f"R² (коэффициент детерминации): {R_squared:.4f}")
print(f"Скорректированный R²: {1 - (1 - R_squared) * (len(y)-1)/(len(y)-X.shape[1]-1):.4f}")
print("----------------------------------")
print("Проверка условия:")
print(f"Сумма квадратов остатков (SSE): {SSE:.4f}")
print(f"Общая сумма квадратов (SST): {SST:.4f}")
print(f"SSE + SSR = {SSE + (SST - SSE):.4f} (должно быть равно SST = {SST:.4f})")

# Визуализация: предсказанные vs фактические значения
plt.figure(figsize=(10, 6))

# Предсказанные vs фактические
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Фактические значения break_force')
plt.ylabel('Предсказанные значения break_force')
plt.title('Предсказанные vs Фактические значения')
plt.grid(True)

# Остатки
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Диаграмма остатков')
plt.grid(True)

plt.tight_layout()
plt.show()

# Создаем DataFrame для удобного просмотра результатов
results_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred,
    'Residual': residuals
})
print("\nТаблица предсказаний и остатков:")
print(results_df.round(4))