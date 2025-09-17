import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Загрузите данные из Excel
df = pd.read_excel('another_results.xlsx')  # Замените 'your_file.xlsx' на путь к вашему файлу

# Посмотрим на данные
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())

# Создаем матрицу диаграмм рассеяния
sns.pairplot(df)
plt.show()

# Рассчитаем матрицу корреляций
correlation_matrix = df.corr()

# Визуализируем тепловую карту корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Тепловая карта корреляций')
plt.show()

# Разделяем признаки и целевую переменную
X = df[['surface_energy', 'roughness']]
y = df['break_force']

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(X, y)

# Получаем веса и смещение
print("\nРезультаты регрессии:")
print("Коэффициенты (w1, w2):", model.coef_)
print("Смещение (b):", model.intercept_)
print(f'''
Наша предсказательная формула может быть представлена как:
break_force = {model.intercept_} + {model.coef_[0]}'surface_energy' +{model.coef_[1]}'roughness' ''')

# Предсказание
y_pred = model.predict(X)
print("\nПредсказанные значения y (break_force):", y_pred)

# Создаем сетку для построения плоскости регрессии
x_surf, y_surf = np.meshgrid(
    np.linspace(X['surface_energy'].min(), X['surface_energy'].max(), 20),
    np.linspace(X['roughness'].min(), X['roughness'].max(), 20)
)
grid_points = pd.DataFrame({
    'surface_energy': x_surf.ravel(),
    'roughness': y_surf.ravel()
})
z_pred = model.predict(grid_points).reshape(x_surf.shape)

# Создаем 3D график
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Отображаем исходные данные
ax.scatter(
    X['surface_energy'], 
    X['roughness'], 
    y, 
    c='blue', 
    marker='o', 
    alpha=1, 
    label='Фактические данные'
)

# Отображаем предсказания модели
ax.scatter(
    X['surface_energy'], 
    X['roughness'], 
    y_pred, 
    c='red', 
    marker='^', 
    alpha=0.5, 
    label='Предсказания'
)

# Отображаем плоскость регрессии
ax.plot_surface(
    x_surf, 
    y_surf, 
    z_pred, 
    color='green', 
    alpha=0.3,
    label='Плоскость регрессии'
)

# Настраиваем график
ax.set_xlabel('Surface Energy')
ax.set_ylabel('Roughness')
ax.set_zlabel('Break Force')
ax.set_title('Линейная регрессия: Зависимость Break Force от Surface Energy и Roughness')
ax.legend()

plt.show()

# Сравниваем фактические и предсказанные значения на 2D графике
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, c='blue', marker='o', alpha=0.7, label='Фактические значения')
plt.scatter(range(len(y_pred)), y_pred, c='red', marker='s', alpha=0.7, label='Предсказанные значения')
plt.xlabel('Номер наблюдения')
plt.ylabel('Break Force')
plt.title('Сравнение фактических и предсказанных значений')
plt.legend()
plt.grid(True)
plt.show()

# График остатков (разница между фактическими и предсказанными значениями)
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.grid(True)
plt.show()

# Оценка качества модели
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nОценка качества модели:")
print(f"R² (коэффициент детерминации): {r2:.4f}")
print(f"MSE (среднеквадратичная ошибка): {mse:.4f}")