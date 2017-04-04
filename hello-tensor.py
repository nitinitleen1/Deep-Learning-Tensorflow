import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)
#c = a + b is also a way to define the sum of the terms

session = tf.Session()

result = session.run(c)
print(result)

session.close()

with tf.Session() as session:
    result = session.run(c)
    print(result)
