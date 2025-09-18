# На данном этапе работает не очень


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Загрузка и подготовка данных
df = pd.read_excel('another_results.xlsx')

# # Преобразуем curing_time с помощью логарифма
# df['log_curing_time'] = np.log(df['curing_time'])

# Разделяем признаки и целевую переменную
X = df[['curing_time', 'surface_energy', 'roughness']]  # Используем логарифмированное время
y = df['break_force']

from sklearn.preprocessing import PolynomialFeatures

# Создаем полиномиальные признаки для curing_time
poly = PolynomialFeatures(degree=2, include_bias=False)
curing_poly = poly.fit_transform(df[['curing_time']])

# Создаем новую матрицу признаков
X_poly = np.column_stack((curing_poly, df[['surface_energy', 'roughness']].values))

# Создаем и обучаем модель
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Предсказание
y_pred_poly = model_poly.predict(X_poly)

# Оценка качества модели
r2_poly = r2_score(y, y_pred_poly)
print(f"R² для полиномиальной модели: {r2_poly:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(df['curing_time'], y, alpha=0.7, label='Фактические данные')
plt.scatter(df['curing_time'], y_pred_poly, alpha=0.7, label='Предсказания (полиномиальная модель)')

# Сортируем для плавной кривой
sorted_idx = np.argsort(df['curing_time'])
plt.plot(df['curing_time'].iloc[sorted_idx], y_pred_poly[sorted_idx], 'r-', label='Аппроксимация')

plt.xlabel('Curing Time')
plt.ylabel('Break Force')
plt.title('Полиномиальная регрессия: Зависимость Break Force от Curing Time')
plt.legend()
plt.grid(True)
plt.show()