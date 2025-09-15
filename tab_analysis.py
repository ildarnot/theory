import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

# Загрузка данных из Excel
df = pd.read_excel('another_results.xlsx', sheet_name='another_results')

# Заменим запятые на точки в числовых столбцах и преобразуем к float
numeric_columns = ['gap', 'surface_energy', 'roughness', 'curing_time', 'break_force']
for col in numeric_columns:
    if df[col].dtype == 'object':  # Если данные строковые (из-за запятых)
        df[col] = df[col].str.replace(',', '.').astype(float)

# 1. Предварительный анализ данных
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nОсновные статистики:")
print(df.describe())

# 2. Визуальный анализ
# Матрица корреляций (только для числовых переменных)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Матрица корреляций числовых переменных')
plt.tight_layout()
plt.show()

# Диаграммы рассеяния
sns.pairplot(df, y_vars=['break_force'], x_vars=['gap', 'surface_energy', 'roughness', 'curing_time'])
plt.suptitle('Диаграммы рассеяния: усилие разрыва vs параметры', y=1.02)
plt.show()

# 3. Подготовка данных для регрессионного анализа
# Кодируем категориальную переменную Method
encoder = OneHotEncoder(drop='first', sparse_output=False)
method_encoded = encoder.fit_transform(df[['Method']])
method_encoded_df = pd.DataFrame(method_encoded, columns=encoder.get_feature_names_out(['Method']))

# Объединяем закодированные категориальные признаки с числовыми
X_numeric = df[['gap', 'surface_energy', 'roughness', 'curing_time']]
X = pd.concat([X_numeric, method_encoded_df], axis=1)
y = df['break_force']

# 4. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Строим и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Оцениваем модель
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR² (коэффициент детерминации) на тестовых данных: {r2:.3f}")
print(f"MSE (среднеквадратичная ошибка) на тестовых данных: {mse:.3f}")

# 7. Детальный анализ с помощью statsmodels (для p-значений и доверительных интервалов)
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()
print("\nДетальная статистика модели:")
print(ols_model.summary())

# 8. Визуализация предсказаний vs реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказанные vs Реальные значения усилия разрыва')
plt.show()

# 9. Важность признаков
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Коэффициент': model.coef_
}).sort_values('Коэффициент', key=abs, ascending=False)

print("\nВажность признаков (по абсолютному значению коэффициента):")
print(feature_importance)

# 10. Функция для предсказания на новых данных
def predict_break_force(gap, surface_energy, roughness, curing_time, method):
    """Функция для предсказания усилия разрыва на основе введенных параметров"""
    # Преобразуем метод в one-hot encoding
    method_df = pd.DataFrame([method], columns=['Method'])
    method_encoded = encoder.transform(method_df)
    method_encoded_df = pd.DataFrame(method_encoded, columns=encoder.get_feature_names_out(['Method']))
    
    # Создаем DataFrame с числовыми признаками
    numeric_data = pd.DataFrame([[gap, surface_energy, roughness, curing_time]], 
                               columns=['gap', 'surface_energy', 'roughness', 'curing_time'])
    
    # Объединяем все признаки
    input_data = pd.concat([numeric_data, method_encoded_df], axis=1)
    
    # Убедимся, что порядок столбцов совпадает с обучающими данными
    input_data = input_data[X.columns]
    
    # Предсказываем
    prediction = model.predict(input_data)
    return prediction[0]

# Пример использования функции
example_prediction = predict_break_force(
    gap=0.2,
    surface_energy=44,
    roughness=3.1,
    curing_time=24,
    method="СО2 Лазер + Полировка на диодном лазере (Пульс большинство 150 наносекунд) + Обезжириватель Efele"
)

print(f"\nПример предсказания: {example_prediction:.2f}")