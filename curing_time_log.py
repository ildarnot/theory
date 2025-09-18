import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Загрузка и подготовка данных
df = pd.read_excel('another_results.xlsx')

# Преобразуем curing_time с помощью логарифма
df['log_curing_time'] = np.log(df['curing_time'])

# Разделяем признаки и целевую переменную
X = df[['log_curing_time', 'surface_energy', 'roughness']]  # Используем логарифмированное время
y = df['break_force']

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Получаем веса и смещение
print("Результаты регрессии:")
print("Коэффициенты (log_curing_time, surface_energy, roughness):", model.coef_)
print("Смещение (b):", model.intercept_)
print(f'''
Наша предсказательная формула может быть представлена как:
break_force = {model.intercept_} + {model.coef_[0]}*log(curing_time) + {model.coef_[1]}*surface_energy + {model.coef_[2]}*roughness ''')

# Предсказание
y_pred = model.predict(X)

# Оценка качества модели
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"R² (коэффициент детерминации): {r2:.4f}")
print(f"MSE (среднеквадратичная ошибка): {mse:.4f}")

# Визуализация зависимости от curing_time
plt.figure(figsize=(12, 5))

# Исходные данные
plt.subplot(1, 2, 1)
plt.scatter(df['curing_time'], y, alpha=0.7, label='Фактические данные')
plt.scatter(df['curing_time'], y_pred, alpha=0.7, label='Предсказания')
plt.xlabel('Curing Time')
plt.ylabel('Break Force')
plt.title('Зависимость Break Force от Curing Time')
plt.legend()
plt.grid(True)

# В логарифмической шкале
plt.subplot(1, 2, 2)
plt.scatter(df['log_curing_time'], y, alpha=0.7, label='Фактические данные')
plt.scatter(df['log_curing_time'], y_pred, alpha=0.7, label='Предсказания')
plt.xlabel('log(Curing Time)')
plt.ylabel('Break Force')
plt.title('Зависимость Break Force от log(Curing Time)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

