import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузите данные из Excel
df = pd.read_excel('another_results.xlsx')  # Замените 'your_file.xlsx' на путь к вашему файлу

# Посмотрим на данные
print(df.head())
print(df.info())
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

# import pandas as pd
from sklearn.linear_model import LinearRegression

# Загружаем данные (например, из CSV)
data = pd.read_excel('another_results.xlsx')  # файл должен содержать колонки x1, x2, x3, x4, y

# Разделяем признаки и целевую переменную
X = data[['surface_energy', 'roughness']]
y = data['break_force']

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(X, y)

# Получаем веса и смещение
print("Коэффициенты (w1, w2):", model.coef_)
print("Смещение (b):", model.intercept_)

# Предсказание
y_pred = model.predict(X)
print("Предсказанные значения y (break_force):", y_pred)