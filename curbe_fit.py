# Помогает отследить влияние времени полимеризации на силу разррушения


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
from scipy.optimize import curve_fit

# Определяем нелинейную функцию (например, экспоненциальную)
def exponential_func(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

# Извлекаем данные только по curing_time для подбора кривой
curing_data = df[df['surface_energy'] == 42]  # Выбираем данные с surface_energy=42
x_data = curing_data['curing_time'].values
y_data = curing_data['break_force'].values

# Подбираем параметры нелинейной функции
popt, pcov = curve_fit(exponential_func, x_data, y_data, p0=[10, 0.1, 0])
a, b, c = popt

print(f"Параметры экспоненциальной модели: a={a:.3f}, b={b:.3f}, c={c:.3f}")

# Визуализация нелинейной аппроксимации
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = exponential_func(x_fit, a, b, c)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='Фактические данные (surface_energy=42)')
plt.plot(x_fit, y_fit, 'r-', label=f'Экспоненциальная аппроксимация: y = {a:.3f}(1 - e^(-{b:.3f}x)) + {c:.3f}')
plt.xlabel('Curing Time')
plt.ylabel('Break Force')
plt.title('Экспоненциальная аппроксимация зависимости Break Force от Curing Time')
plt.legend()
plt.grid(True)
plt.show()