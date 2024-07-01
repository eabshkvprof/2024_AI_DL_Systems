# Модуль 7. TensorFlow
## Лекція 03. Приклад 1. Обчислювальний граф

##### Введення в градієнти та автоматичне діференцювання
##### [Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff)




```python
import tensorflow as tf
from pprint import pprint
print(tf.__version__)
```

    2.15.0
    


```python
print(tf.executing_eagerly())
```

    True
    

### Використання [tf.gradients](https://www.tensorflow.org/api_docs/python/tf/gradients)


```python
a = tf.Variable(20.0,name="A")
b = tf.Variable(30.0,name="B")
c = tf.Variable(15.0,name="C")
d = tf.Variable(tf.add(a, b),name="D")
e = tf.Variable(tf.subtract(b, c),name="E")
y = tf.Variable(tf.multiply(d,e),name="Y")
```


```python
print('A=', a.numpy())
print('B=', b.numpy())
print('C=', c.numpy())
print('D=', d.numpy())
print('E=', e.numpy())
print('Y=', y.numpy())
```

    A= 20.0
    B= 30.0
    C= 15.0
    D= 50.0
    E= 15.0
    Y= 750.0
    


```python
@tf.function
def example_1():
  d = a + b
  e = b - c
  cost = d * e
  g = tf.gradients(cost, [a, b, c])
  dCost_da, dCost_db, dCost_dc = g
  # Повертаємо компоненти градієнту
  return dCost_da, dCost_db, dCost_dc
ga,gb,gc = example_1()
print(ga, gb, gc)
print(ga.numpy(), gb.numpy(),gc.numpy())


```

    tf.Tensor(15.0, shape=(), dtype=float32) tf.Tensor(65.0, shape=(), dtype=float32) tf.Tensor(-50.0, shape=(), dtype=float32)
    15.0 65.0 -50.0
    


```python
@tf.function
def example_2():
  Ws = tf.constant(1.)
  bs = 2 * Ws
  cost = Ws + bs  # Це тільки приклад
  g = tf.gradients(cost, [Ws, bs])
  dCost_dW, dCost_db = g
  return dCost_dW, dCost_db
example_2()
```




    (<tf.Tensor: shape=(), dtype=float32, numpy=3.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=1.0>)



Використання [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape)

Аналог першого прикладу


```python
# Стрічка
with tf.GradientTape(watch_accessed_variables=True) as tape:
  d = a + b
  e = b - c
  cost = d * e

print('D ', d.numpy(), 'E ',e.numpy())
print('COST ', cost.numpy())

ga , gb, gc  = tape.gradient(target=cost, sources=[a, b, c])
print(ga, gb, gc)
print(ga.numpy(), gb.numpy(),gc.numpy())


```

    D  [20.1 20.2 20.3] E  [-14.9 -14.8 -14.7]
    COST  [-299.49    -298.96002 -298.40997]
    tf.Tensor(-44.4, shape=(), dtype=float32) tf.Tensor([5.200001  5.4000006 5.5999994], shape=(3,), dtype=float32) tf.Tensor(-60.600002, shape=(), dtype=float32)
    -44.4 [5.200001  5.4000006 5.5999994] -60.600002
    

Градієнти від двох змінних векторів


```python
avec = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
bvec = tf.Variable([0.1, 0.2, 0.3, 0.4,], dtype=tf.float32)
print('A', avec.numpy())
print('B', bvec.numpy())
def example_3 (a, b, power = 2, d = 3):
  return tf.pow(a, power) + d * b

# Стрічка
with tf.GradientTape(watch_accessed_variables=True) as tape:
  cvec = example_3(avec, bvec)
print('C', cvec.numpy())

ga , gb = tape.gradient(target=cvec, sources=[avec, bvec])
print( ga, gb )
print('Gradient on A', ga.numpy())
print('Gradient on B', gb.numpy())
```

    A [4. 5. 6. 7.]
    B [0.1 0.2 0.3 0.4]
    C [16.3 25.6 36.9 50.2]
    tf.Tensor([ 8. 10. 12. 14.], shape=(4,), dtype=float32) tf.Tensor([3. 3. 3. 3.], shape=(4,), dtype=float32)
    Gradient on A [ 8. 10. 12. 14.]
    Gradient on B [3. 3. 3. 3.]
    

Градієнт скаляра від двох змінних веторів


```python
avect = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
bvect = tf.Variable([0.1, 0.2, 0.3, 0.4,], dtype=tf.float32)
print('A', avect.numpy())
print('B', bvect.numpy())
def example_4 (a, b, power = 2, d = 3):
  return tf.reduce_sum(tf.pow(a, power) + d * b)

# Стрічка
with tf.GradientTape(watch_accessed_variables=True) as tape:
  cscal = example_4(avect, bvect)
print('C', cscal.numpy())

ga , gb = tape.gradient(target=cscal, sources=[avect, bvect])
print( ga, gb )
print('Gradient on A', ga.numpy())
print('Gradient on B', gb.numpy())
```

    A [4. 5. 6. 7.]
    B [0.1 0.2 0.3 0.4]
    C 129.0
    tf.Tensor([ 8. 10. 12. 14.], shape=(4,), dtype=float32) tf.Tensor([3. 3. 3. 3.], shape=(4,), dtype=float32)
    Gradient on A [ 8. 10. 12. 14.]
    Gradient on B [3. 3. 3. 3.]
    

Градіент іфд змінних вектору та матриці


```python
x  = tf.Variable([4, 5, 6, 7], dtype=tf.float32) #vector
print('x', x.numpy())
b = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float32)  #vector
print('b', b.numpy())
W = tf.Variable(tf.random.normal([3 ,4], 0, 1, tf.float32))  #matrix
print('W', W.numpy())
```

    x [4. 5. 6. 7.]
    b [0.1 0.2 0.3]
    W [[-0.5234383  -1.0841007  -2.2260287  -1.0235329 ]
     [-0.46446875  0.76562446  0.7562393   1.2991254 ]
     [-1.7583      0.59467864  0.45476842 -0.01941086]]
    


```python
# Стрічка
with tf.GradientTape(watch_accessed_variables=True) as tape:
  ydot = tf.tensordot(W, x, axes=1)
  out = ydot + b

gb , gW = tape.gradient(target=out, sources=[b, W])

print(gW)
print(gb)
print('Gradient on W', gW.numpy())
print('Gradient on b', gb.numpy())
```

    tf.Tensor(
    [[4. 5. 6. 7.]
     [4. 5. 6. 7.]
     [4. 5. 6. 7.]], shape=(3, 4), dtype=float32)
    tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)
    Gradient on W [[4. 5. 6. 7.]
     [4. 5. 6. 7.]
     [4. 5. 6. 7.]]
    Gradient on b [1. 1. 1.]
    

Сума квадратів


```python
# Стрічка
with tf.GradientTape(watch_accessed_variables=True) as tape:
  ydot = tf.tensordot(W, x, axes=1)
  # print('YDOT',ydot.numpy())
  outvec = ydot + b
  # print('OUT VEC',outvec.numpy())
  loss = tf.reduce_sum(tf.pow(outvec,2))
  # print('LOSS',loss.numpy())

gb , gW = tape.gradient(target=loss, sources=[b, W])

print('Gradient on W', gW.numpy())
print('Gradient on b', gb.numpy())
```

    Gradient on W [[-223.48128  -279.3516   -335.22192  -391.09222 ]
     [ 126.41248   158.01561   189.61873   221.22185 ]
     [  -9.336578  -11.670723  -14.004868  -16.339012]]
    Gradient on b [-55.87032    31.60312    -2.3341446]
    


```python
# Перевірка: Похідна від Loss до W[0,0]
dL_on_W00 = 2*outvec[0]*x[0]
print (dL_on_W00.numpy())
```

    -223.48128
    


```python

```
