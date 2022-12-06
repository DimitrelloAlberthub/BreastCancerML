В рамках данной лабораторной работы были реализованы следующие алгоритмы машинного обучения: 
+ Линейная регрессия;
+ Логистическая регрессия;
+ Метод опорных векторов;
+ К близжайших соседей;
+ Наивный байесовский классификатор;
+ Рандомный лес;
+ Дерево решений;


Для каждой модели были полученны оценки метрик:
+ Confusion Matrix, 
+ Accuracy, 
+ Recall, 
+ Precision, 
+ ROC_AUC curve;




```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
```


```python
# Считываем данные
data = pd.read_csv('./Data/breast_cancer.csv')
```


```python
# Выводим таблицу
data
```

Категориальные параметры:
+ diagnosis

Количественные параметры:

+ radius_mean
+ texture_mean 
+ perimeter_mean
+ area_mean	smoothness_mean
+ compactness_mean
+ concavity_mean
+ concave points_mean
+ radius_worst
+ texture_worst 
+ perimeter_worst
+ area_worst 
+ smoothness_worst
+ compactness_worst
+ concavity_worst
+ concave points_worst
+ symmetry_worst
+ fractal_dimension_worst


```python
# Количество столбцов и строк
data.shape
```


```python
# Выведем информацию по датафрейму включая тип и столбцы индекса, ненулевые значения и использование памяти.
data.info()
```


```python
# Подсчет уникальных значений
data.diagnosis.value_counts()
```


```python
# Вывод описательной статистики. все столбцы ввода включены в вывод.
data.describe(include='all').T
```


```python
data.hist(bins=50,figsize=(20,15))
# построение гитрограммы значений признаков
plt.show();
```


```python
# вывод минимального и максимального значений
print("radius_mean range:", data.radius_mean.min(), data.radius_mean.max())
print("Htexture_mean range:", data.texture_mean.min(),data.texture_mean.max())
print("perimeter_mean range:", data.perimeter_mean.min(),data.perimeter_mean.max())
print("area_mean range:", data.area_mean.min(), data.area_mean.max())
print("smoothness_mean range:", data.smoothness_mean.min(), data.smoothness_mean.max())
print("compactness_mean range:", data.compactness_mean.min(), data.compactness_mean.max())
```


```python
# замена
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data
```


```python
# Устанавливаем seed
np.random.seed(42)
# Убираем столбец
x = data.drop('diagnosis', axis=1)
y = data['diagnosis']
```


```python
# построение кривой
def plot_ROC_curve(y_true, y_predicted_probabilities):  
    from sklearn.metrics import roc_curve
    false_positive_rate, true_postitive_rate, _ = roc_curve(y_true,  y_predicted_probabilities)   
    plt.plot(false_positive_rate, true_postitive_rate)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
```

## Логистическая регрессия


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix, classification_report
# разбиваем данные
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
# обучаем модель
model.fit(x_train, y_train)
# оцениваем точность логистической регрессии
model.score(x_test, y_test)

```


```python
import pickle 
```


```python
# запись обученной модели в файл
filename = 'lr.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
# Прогнозируем на основе обученной модели
test_predictions = model.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm = confusion_matrix(y_test, test_predictions)
cm
```


```python
import seaborn as sn
# построим график матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


```python
# оцениваем точность линейной регрессии
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## Дерево решений


```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# обучаем модель
model.fit(x_train, y_train)

```


```python
# запись обученной модели в файл
filename = 'Td.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
# Прогнозируем на основе обученной модели
test_predictions = model.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm1 = confusion_matrix(y_test, test_predictions)
cm1

```


```python
# построим график матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm1, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


```python
# оцениваем точность дерева решений
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## К близжайших соседей


```python
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=9)
# обучаем модель
knn.fit(x_train, y_train)
```


```python
# запись обученной модели в файл
filename = 'knn.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
# Прогнозируем на основе обученной модели
train_predictions = knn.predict(x_train)
test_predictions = knn.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm2 = confusion_matrix(y_test, test_predictions)
cm2

```


```python
# Построение графика матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm2, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

```


```python
# оцениваем точность knn
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## Метод опорных векторов


```python
from sklearn import svm
from sklearn.svm import SVC

```


```python
svm = SVC()
```


```python
## обучаем модель
svm.fit(x_train, y_train)
```


```python
# запись обученной модели в файл
filename = 'svc.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
# Прогнозируем на основе обученной модели
test_predictions = svm.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm3= confusion_matrix(y_test, test_predictions)
cm3

```


```python
# построим график матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm3, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


```python
# оцениваем точность метода опорных векторов
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## Рандомный лес


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
# установка классификатора рандомного леса
rf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=99)
# обучаем модель
rf.fit(x_train, y_train)
```


```python
# Прогнозируем на основе обученной модели
test_predictions = rf.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm4= confusion_matrix(y_test, test_predictions)
cm4
```


```python
# Построение графика матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm4, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


```python
# оцениваем точность модели
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## Наивный байесовский классификатор


```python
from sklearn.naive_bayes import GaussianNB
```


```python
model = GaussianNB()
# обучение модели
model.fit(x_train, y_train)
```


```python
# запись обученной модели в файл
filename = 'gauss.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
# Прогнозируем на основе обученной модели
test_predictions = model.predict(x_test)
```


```python
# построение кривой
plot_ROC_curve(y_test,test_predictions)
```


```python
# получим матрицу ошибок
cm5= confusion_matrix(y_test, test_predictions)
cm5
```


```python
# Построение графика матрицы ошибок
plt.figure(figsize=(5,3))
sn.heatmap(cm5, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


```python
# оценим точность
accuracy_score(y_test,test_predictions)
```


```python
# отобразим основные метрики классификации
print(classification_report(y_test,test_predictions))
```

## Точность моделей

Логистическая регрессия = 0.6228070175438597

Дерево решений = 0.9473684210526315

Рандомный лес = 0.9649122807017544

KNN = 0.7192982456140351

Метод опорных векторов = 0.6228070175438597

Наивный байесовский классификатор = 0.6140350877192983


```python

```
