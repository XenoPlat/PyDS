
### Практическое задание NumPy
1. Импортируйте библиотеку Numpy и дайте ей псевдоним np.

```python
import numpy as np
```
Создать одномерный массив Numpy под названием a из 12 последовательных целых чисел чисел от 12 до 24 невключительно

```python
a = [i for i in range (12,24)]
a
```




    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]




```python
a = np.arange(12,24)
a
```




    array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])


Создать 5 двумерных массивов разной формы из массива a. Не использовать в аргументах метода reshape число -1

```python
b = a.reshape(3,4)
b
```




    array([[12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])




```python
b = a.reshape(2,6)
b
```




    array([[12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])




```python
b = a.reshape(4,3)
b
```




    array([[12, 13, 14],
           [15, 16, 17],
           [18, 19, 20],
           [21, 22, 23]])




```python
b = a.reshape(6,2)
b
```




    array([[12, 13],
           [14, 15],
           [16, 17],
           [18, 19],
           [20, 21],
           [22, 23]])




```python
b = a.reshape(3,4)
c = b[1:, 2:]
c
```




    array([[18, 19],
           [22, 23]])




```python
b = a.reshape(12,1)
b
```




    array([[12],
           [13],
           [14],
           [15],
           [16],
           [17],
           [18],
           [19],
           [20],
           [21],
           [22],
           [23]])


Создать 5 двумерных массивов разной формы из массива a. Использовать в аргументах метода reshape число -1 (в трех примерах - для обозначения числа столбцов, в двух - для строк).

```python
b = a.reshape(2, -1)
b
```




    array([[12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])




```python
b = a.reshape(3, -1)
b
```




    array([[12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])




```python
b = a.reshape(4, -1)
b
```




    array([[12, 13, 14],
           [15, 16, 17],
           [18, 19, 20],
           [21, 22, 23]])




```python
b = a.reshape(-1, 2)
b
```




    array([[12, 13],
           [14, 15],
           [16, 17],
           [18, 19],
           [20, 21],
           [22, 23]])




```python
b = a.reshape(-1, 1)
b
```




    array([[12],
           [13],
           [14],
           [15],
           [16],
           [17],
           [18],
           [19],
           [20],
           [21],
           [22],
           [23]])


Можно ли массив Numpy, состоящий из одного столбца и 12 строк, назвать одномерным?

```python
print('Да')
```

    Да
    
Создать массив из 3 строк и 4 столбцов, состоящий из случайных чисел с плавающей запятой из 
нормального распределения  со средним, равным 0 и среднеквадратичным отклонением, равным 1.0.

```python
a = np.random.randn(3, 4)
a
```




    array([[ 2.27345508,  0.6929144 ,  0.0926329 ,  0.15036675],
           [-0.07775819,  1.11493749, -1.8953378 , -0.26788536],
           [ 0.70531711,  0.83433111, -1.01561848, -0.27700114]])




```python
Получить из этого массива одномерный массив с таким же атрибутом size, как и исходный массив.
```


```python
b = a.reshape(1, a.size)
b
```




    array([[ 2.27345508,  0.6929144 ,  0.0926329 ,  0.15036675, -0.07775819,
             1.11493749, -1.8953378 , -0.26788536,  0.70531711,  0.83433111,
            -1.01561848, -0.27700114]])


3. Создать массив a, состоящий из целых чисел, убывающих от 20 до 0 невключительно с интервалом 2.

```python
a = np.arange(20,0,-2)
a
```




    array([20, 18, 16, 14, 12, 10,  8,  6,  4,  2])


Создать массив b, состоящий из 1 строки и 10 столбцов: целых чисел, убывающих от 20 до 1 невключительно с интервалом 2. 

```python
b = np.arange(20,1,-2)
b
```




    array([20, 18, 16, 14, 12, 10,  8,  6,  4,  2])


В чем разница между массивами a и b?

```python
print('Отсутствует')
```

    Отсутствует
    
4. Вертикально соединить массивы a и b. a - двумерный массив из нулей, число строк которого больше 1 и на 1 меньше, чем число строк двумерного массива b, состоящего из единиц. Итоговый массив v должен иметь атрибут size, равный 10.

```python
a = np.zeros((2, 2))
b = np.ones((3, 2))
v = np.concatenate((a, b), axis=0).size
v
```




    10


5. Создать одномерный массив а, состоящий из последовательности целых чисел от 0 до 12

```python
a = np.arange(12)
a
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])


Поменять форму этого массива, чтобы получилась матрица A (двумерный массив Numpy), состоящая из 4 строк и 3 столбцов

```python
A = a.reshape(3,4)
A
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])


Получить матрицу At путем транспонирования матрицы A

```python
At = A.T
At
```




    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])


Получить матрицу B, умножив матрицу A на матрицу At с помощью матричного умножения

```python
B = At.dot(A)
B
```




    array([[ 80,  92, 104, 116],
           [ 92, 107, 122, 137],
           [104, 122, 140, 158],
           [116, 137, 158, 179]])


Какой размер имеет матрица B? Получится ли вычислить обратную матрицу для матрицы B и почему?

```python
B.size
```




    16




```python
d = np.linalg.det(B)
d
print('Нет. Определитель = 0')
```

    Нет. Определитель = 0
    
6. Инициализируйте генератор случайных числе с помощью объекта seed, равного 42

```python
np.random.seed(42)
```
Создайте одномерный массив c, составленный из последовательности 16-ти случайных равномерно распределенных целых чисел от 0 до 16 невключительно

```python
c = np.random.randint(0, 16, 16)
c
```




    array([ 7,  6, 10,  4,  2, 10, 11,  6, 14, 11,  6,  8,  4,  9,  4,  8])


Поменяйте его форму так, чтобы получилась квадратная матрица C

```python
C = c.reshape(4, 4)
C
```




    array([[ 7,  6, 10,  4],
           [ 2, 10, 11,  6],
           [14, 11,  6,  8],
           [ 4,  9,  4,  8]])


Получите матрицу D, поэлементно прибавив матрицу B из предыдущего вопроса к матрице C, умноженной на 10

```python
D = B + C*10
D
```




    array([[150, 152, 204, 156],
           [112, 207, 232, 197],
           [244, 232, 200, 238],
           [156, 227, 198, 259]])


Вычислите определитель, ранг и обратную матрицу D_inv для D

```python
print(np.linalg.det(D))
print(np.linalg.matrix_rank(D))
D_inv = np.linalg.inv(D)
D_inv
```

    -65045000.000000134
    4
    




    array([[ 0.00511738, -0.0067996 ,  0.00663664, -0.00400892],
           [-0.03310016,  0.02993312,  0.02040895, -0.02158506],
           [ 0.01438266, -0.00089876, -0.00806734, -0.00056607],
           [ 0.01493305, -0.02145223, -0.01571743,  0.02562657]])


7. Приравняйте к нулю отрицательные числа в матрице D_inv, а положительные - к единице. Убедитесь, что в матрице D_inv остались только нули и единицы

```python
D_inv[np.where(D_inv < 0)] = 0
D_inv[np.where(D_inv > 0)] = 1
D_inv
```




    array([[1., 0., 1., 0.],
           [0., 1., 1., 0.],
           [1., 0., 0., 0.],
           [1., 0., 0., 1.]])


С помощью функции numpy.where, используя матрицу D_inv в качестве маски, а матрицы B и C - в качестве источников данных, получите матрицу E размером 4x4. Элементы матрицы E, для которых соответствующий элемент матрицы D_inv равен 1, должны быть равны соответствующему элементу матрицы B, а элементы матрицы E, для которых соответствующий элемент матрицы D_inv равен 0, должны быть равны соответствующему элементу матрицы C

```python
E = np.where(D_inv, B, C)
E
```




    array([[ 80,   6, 104,   4],
           [  2, 107, 122,   6],
           [104,  11,   6,   8],
           [116,   9,   4, 179]])


Задание 2Создайте массив Numpy под названием a размером 5x2, то есть состоящий из 5 строк и 2 столбцов.
Первый столбец должен содержать числа 1, 2, 3, 3, 1, а второй - числа 6, 8, 11, 10, 7.
Будем считать, что каждый столбец - это признак, а строка - наблюдение.
Затем найдите среднее значение по каждому признаку, используя метод mean массива Numpy.
Результат запишите в массив mean_a, в нем должно быть 2 элемента

```python
a = np.array([[1, 2, 3, 3, 1], [6, 8, 11, 10, 7]])
a = a.transpose()
print(a)
mean_a = np.mean(a, axis = 0)
print()
print(mean_a)
```

    [[ 1  6]
     [ 2  8]
     [ 3 11]
     [ 3 10]
     [ 1  7]]
    
    [2.  8.4]
    
Задание 3Вычислите массив a_centered, отняв от значений массива а средние значения соответствующих признаков, содержащиеся в массиве mean_a. Вычисление должно производиться в одно действие. Получившийся массив должен иметь размер 5x2

```python
a_centered = a - mean_a
a_centered
```




    array([[-1. , -2.4],
           [ 0. , -0.4],
           [ 1. ,  2.6],
           [ 1. ,  1.6],
           [-1. , -1.4]])


Задание 4Найдите скалярное произведение столбцов массива a_centered. В результате должна получиться величина a_centered_sp. Затем поделите a_centered_sp на N-1, где N - число наблюдений

```python
x = a_centered[:,0]
y = a_centered[:,1]
a_centered_sp = np.dot(x, y)
a_centered_sp = a_centered_sp/4
a_centered_sp
```




    2.0


Задание 5Число, которое мы получили в конце задания 3 является ковариацией двух признаков, содержащихся в массиве а. В задании 4 мы делили сумму произведений центрированных признаков на N-1, а не на N, поэтому полученная нами величина является несмещенной оценкой ковариации. В этом задании проверьте получившееся число, вычислив ковариацию еще одним способом - с помощью функции np.cov. В качестве аргумента m функция np.cov должна принимать транспонированный массив a. В получившейся ковариационной матрице (массив Numpy размером 2x2) искомое значение
### Практическое задание Pandas
Задание 1
A. Импортируйте библиотеку Pandas и дайте ей псевдоним pd.


```python
import pandas as pd
```
B. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].

```python
authors = pd.DataFrame({'author_id': [1, 2, 3], 'author_name': ['Тургенев', 'Чехов', 'Островский']}, 
                       columns = ['author_id', 'author_name'])
authors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>author_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Тургенев</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Чехов</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Островский</td>
    </tr>
  </tbody>
</table>
</div>


Затем создайте датафрейм books cо столбцами author_id, book_title и price,в которых соответственно содержатся данные: [1, 1, 1, 2, 2, 3, 3], ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'], [450, 300, 350, 500, 450, 370, 290]

```python
books = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                     'price': [450, 300, 350, 500, 450, 370, 290]},
                   columns = ['author_id', 'book_title', 'price'])
books
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>book_title</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Отцы и дети</td>
      <td>450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Рудин</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Дворянское гнездо</td>
      <td>350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Толстый и тонкий</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Дама с собачкой</td>
      <td>450</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>Гроза</td>
      <td>370</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>Таланты и поклонники</td>
      <td>290</td>
    </tr>
  </tbody>
</table>
</div>


Задание 2Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id

```python
authors_price = pd.merge(authors, books, on = 'author_id', how = 'inner')
authors_price
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>author_name</th>
      <th>book_title</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Тургенев</td>
      <td>Отцы и дети</td>
      <td>450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Тургенев</td>
      <td>Рудин</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Тургенев</td>
      <td>Дворянское гнездо</td>
      <td>350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Чехов</td>
      <td>Толстый и тонкий</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Чехов</td>
      <td>Дама с собачкой</td>
      <td>450</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>Островский</td>
      <td>Гроза</td>
      <td>370</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>Островский</td>
      <td>Таланты и поклонники</td>
      <td>290</td>
    </tr>
  </tbody>
</table>
</div>


Задание 3Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами

```python
top5 = authors_price.nlargest(5, 'price')
top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>author_name</th>
      <th>book_title</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Чехов</td>
      <td>Толстый и тонкий</td>
      <td>500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Тургенев</td>
      <td>Отцы и дети</td>
      <td>450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Чехов</td>
      <td>Дама с собачкой</td>
      <td>450</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>Островский</td>
      <td>Гроза</td>
      <td>370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Тургенев</td>
      <td>Дворянское гнездо</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>


Задание 4A. Создайте датафрейм authors_stat на основе информации из authors_priceB. В датафрейме authors_stat должны быть четыре столбца: author_name, min_price, max_price и mean_price, в которых должны содержаться соответственно имя автора, минимальная, максимальная и средняя цена на книги этого автора

```python
author_stat = pd.DataFrame({'min_price': authors_price.groupby('author_name')['price'].min(),
                                                'max_price': authors_price.groupby('author_name')['price'].max(),
                                                'mean_price': authors_price.groupby('author_name')['price'].mean()})
author_stat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min_price</th>
      <th>max_price</th>
      <th>mean_price</th>
    </tr>
    <tr>
      <th>author_name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Островский</th>
      <td>290</td>
      <td>370</td>
      <td>330.000000</td>
    </tr>
    <tr>
      <th>Тургенев</th>
      <td>300</td>
      <td>450</td>
      <td>366.666667</td>
    </tr>
    <tr>
      <th>Чехов</th>
      <td>450</td>
      <td>500</td>
      <td>475.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
