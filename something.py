import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из Excel
df = pd.read_excel('your_data.xlsx')  # Изменили на read_excel

# 1. Посмотреть на первые строки
print("Первые 5 строк данных:")
print(df.head())

# 2. Основные статистики
print("\nОсновные статистические характеристики данных:")
print(df.describe())

# 3. Построить матрицу корреляций (для численных переменных)
print("\nМатрица корреляций:")
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()

# 4. Построить диаграммы рассеяния для целевой переменной и каждого численного признака
print("\nДиаграммы рассеяния:")
sns.pairplot(df, y_vars=['break_force'], x_vars=['gap', 'roughness', 'curing_time'])
plt.suptitle('Диаграммы рассеяния: усилие разрыва vs параметры', y=1.02)
plt.show()