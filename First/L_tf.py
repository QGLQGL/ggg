import tensorflow as tf

a = tf.constant(1.0)
b = tf.constant(2.0)
c = a+b
print(a,b,c)

with tf.Session() as sess:
    print(c.eval())
    print(sess.run(c))