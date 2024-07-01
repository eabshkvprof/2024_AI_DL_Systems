# Модуль 7. TensorFlow
## Лекція 02. Приклад 2. Лінійна алгебра

Операції лінійної алгебри надає модуль [**tf.linalg**](https://www.tensorflow.org/api_docs/python/tf/linalg)



---



### Вирішення систем лінійних алгебраїчних рівняннь
[tf.linalg.solve](https://www.tensorflow.org/api_docs/python/tf/linalg/solve)


#### Функція

tf.linalg.solve(
    **matrix**: Annotated[Any, TV_MatrixSolve_T],
    **rhs**: Annotated[Any, TV_MatrixSolve_T],
    **adjoint**: bool = False,
    **name=None**
) -> Annotated[Any, TV_MatrixSolve_T]

Matrix — це тензор форми [..., M, M], внутрішні 2 виміри якого утворюють квадратні матриці. Rhs — тензор форми [..., M, K]. Результатом є тензорна форма [..., M, K]. Якщо adjoint має значення False, тоді кожна вихідна матриця задовольняє

 **matrix**[..., :, :] X **output**[..., :, :] = **rhs**[..., :, :].

 Якщо adjoint має значення True, тоді кожна вихідна матриця задовольняє

adjoint(**matrix**[..., :, :]) X **output**[..., :, :] = **rhs**[..., :, :].

Args
- **matrix**	A Tensor. Must be one of the following types: float64, float32, half, complex64, complex128. Shape is [..., M, M].

- **rhs**	A Tensor. Must have the same type as matrix. Shape is [..., M, K].

- **adjoint**	An optional bool. Defaults to False. Boolean indicating whether to solve with matrix or its (block-wise) adjoint.

- **name**	A name for the operation (optional).
Returns
- A Tensor. Has the same type as matrix.

Визначення версії TF


```python
import tensorflow as tf # імпорт tensofflow
import numpy as np # імпорт numpy
import pprint as pprint # імпорт пакету посиленого друку
print(tf.__version__) # версія TF
```

    2.15.0
    

### Пакет квадратних матриць


```python
# A = ... # shape 3 x 2 x 2
A = tf.constant([[[1,-2],[-3,4]],[[5,-6],[7,8]],[[9,-10],[-11,12]]], dtype=tf.float32)
print(tf.shape(A))
print(A.numpy())
```

    tf.Tensor([3 2 2], shape=(3,), dtype=int32)
    [[[  1.  -2.]
      [ -3.   4.]]
    
     [[  5.  -6.]
      [  7.   8.]]
    
     [[  9. -10.]
      [-11.  12.]]]
    

### Пакет векторів правих частин


```python
#B = ... # shape 3 x 2 x 1
B = tf.constant([[[1],[-2]],[[-5],[6]],[[-9],[-10]]], dtype=tf.float32)
print(tf.shape(B))
print(B.numpy())
```

    tf.Tensor([3 2 1], shape=(3,), dtype=int32)
    [[[  1.]
      [ -2.]]
    
     [[ -5.]
      [  6.]]
    
     [[ -9.]
      [-10.]]]
    


```python
# Вирішення 3-х лінійних систем розміром 2x2:
X = tf.linalg.solve(A, B)

```


```python
print(tf.shape(X))
print(X.numpy())
```

    tf.Tensor([3 2 1], shape=(3,), dtype=int32)
    [[[-0.0000000e+00]
      [-5.0000000e-01]]
    
     [[-4.8780512e-02]
      [ 7.9268295e-01]]
    
     [[ 1.0400010e+02]
      [ 9.4500092e+01]]]
    

### Приклад 1

Маємо 100 матриць  4x4 (тензор A1) та 100 4-вимірних векторів (тензор B1).  Тобто необхідно вирішити  100 різних лінійних систем . Необхідно використати формат [100,4,4] для матриці та [100,4,1] для пакету векторів правих частин.


```python
A1 = tf.random.normal([100, 4 ,4], 0, 100, tf.float32)
print('Object A1 -->', tf.shape(A1).numpy())
```

    Object A1 --> [100   4   4]
    


```python
B1 = tf.random.normal([100, 4 , 1], 0, 20, tf.float32)
print('Object B1 -->', tf.shape(B1).numpy())
```

    Object B1 --> [100   4   1]
    


```python
X1 = tf.linalg.solve(A1, B1)
```


```python
print(tf.shape(X1))
print(X1[0].numpy())

```

    tf.Tensor([100   4   1], shape=(3,), dtype=int32)
    [[-0.35240462]
     [ 0.58189464]
     [-0.45182198]
     [ 0.06966337]]
    

### Приклад 2

Маємо одну 4x4-матрицю A2 та 100 4-вимірних векторів B2.  Тобто необхідно вирішити  100 лінійних систем з однаковою матрицею. Необхідно використати формат [1,4,4] для матриці та [1,4,100] для пакету векторів правих частин. Або, спрощено, [4,4] та [4,100] відповідно.


```python
A2 = tf.random.normal([4 ,4], 0, 200, tf.float32)
print('Object A2 -->', tf.shape(A2).numpy())
print(A2.numpy())
```

    Object A2 --> [4 4]
    [[   7.2443404  101.91894    -45.502487     7.0334845]
     [  50.031918   197.67627     65.50794   -481.22183  ]
     [ 123.122116   -47.91781   -268.49185   -105.97589  ]
     [ 161.8969      -3.3252578  647.87555    197.0063   ]]
    


```python
B2 = tf.random.normal([4, 100], 0, 20, tf.float32)
print('Object B2 -->', tf.shape(B2).numpy())
print(B2[:,0].numpy())
```

    Object B2 --> [  4 100]
    [-6.6021185 -9.577134  46.866814   6.110441 ]
    


```python
X2 = tf.linalg.solve(A2, B2)
```


```python
print(tf.shape(X2))
print(X2[:,0].numpy())

```

    tf.Tensor([  4 100], shape=(2,), dtype=int32)
    [ 0.23259854 -0.1023673  -0.04785533 -0.00448027]
    

---------
