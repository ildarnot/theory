import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Загрузите данные из Excel
df = pd.read_excel('another_results.xlsx')

# Убедимся, что числовые данные правильно преобразованы (запятые в точки)
numeric_columns = ['curing_time', 'surface_energy', 'roughness', 'break_force']
for col in numeric_columns:
    if df[col].dtype == 'object':  # Если данные строковые (из-за запятых)
        df[col] = df[col].str.replace(',', '.').astype(float)

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
X = df[['curing_time', 'surface_energy', 'roughness']]  # Добавили curing_time
y = df['break_force']

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(X, y)

# Получаем веса и смещение
print("\nРезультаты регрессии:")
print("Коэффициенты (curing_time, surface_energy, roughness):", model.coef_)
print("Смещение (b):", model.intercept_)
print(f'''
Наша предсказательная формула может быть представлена как:
break_force = {model.intercept_} + {model.coef_[0]}*curing_time + {model.coef_[1]}*surface_energy + {model.coef_[2]}*roughness ''')

# Предсказание
y_pred = model.predict(X)
print("\nПредсказанные значения y (break_force):", y_pred)

# Для 3D визуализации выберем два наиболее значимых признака
# Создаем сетку для построения поверхности регрессии
# Выберем два признака для визуализации (например, curing_time и surface_energy)
x_surf, y_surf = np.meshgrid(
    np.linspace(X['curing_time'].min(), X['curing_time'].max(), 20),
    np.linspace(X['surface_energy'].min(), X['surface_energy'].max(), 20)
)

# Для третьего признака используем среднее значение
roughness_mean = X['roughness'].mean()
grid_points = pd.DataFrame({
    'curing_time': x_surf.ravel(),
    'surface_energy': y_surf.ravel(),
    'roughness': roughness_mean  # Фиксируем roughness на среднем уровне
})

z_pred = model.predict(grid_points).reshape(x_surf.shape)

# Создаем 3D график
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Отображаем исходные данные
ax.scatter(
    X['curing_time'], 
    X['surface_energy'], 
    y, 
    c='blue', 
    marker='o', 
    alpha=1, 
    label='Фактические данные'
)

# Отображаем предсказания модели (только для фиксированного значения roughness)
ax.scatter(
    X['curing_time'], 
    X['surface_energy'], 
    y_pred, 
    c='red', 
    marker='^', 
    alpha=0.5, 
    label='Предсказания'
)

# Отображаем поверхность регрессии
surf = ax.plot_surface(
    x_surf, 
    y_surf, 
    z_pred, 
    cmap='viridis',
    alpha=0.6,
    label='Поверхность регрессии'
)

# Добавляем цветовую шкалу
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Настраиваем график
ax.set_xlabel('Curing Time')
ax.set_ylabel('Surface Energy')
ax.set_zlabel('Break Force')
ax.set_title('Регрессия: Break Force от Curing Time, Surface Energy и Roughness\n(Roughness зафиксирован на среднем уровне)')
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

# Дополнительный анализ: важность признаков
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Коэффициент': model.coef_,
    'Абсолютное значение': np.abs(model.coef_)
}).sort_values('Абсолютное значение', ascending=False)

print("\nВажность признаков (по абсолютному значению коэффициента):")
print(feature_importance)