################### Tensorflow boilerplate code ########################
import tensorflow as tf

a = tf.constant(4.0,dtype=tf.float32)
b = tf.constant(2.0)
total = a + b
print(total)
c = tf.fill(dims=[1],value=4.0)
d = tf.fill(dims=[1],value=2.0)
total1 = c + d
print(total1)
vec = tf.random_uniform(shape=(3,))
print('vec',vec+1)
with tf.Session() as sess:
    print(sess.run((vec+1,vec+2))) #has to be a tuple or dictionary inside sess.run()
    print(sess.run(total))
    print(sess.run({'ab':(a,b),'total':total})) #has to be a tuple or dictionary inside sess.run()

                    #########################
x = tf.placeholder(shape=(3,None),dtype=tf.float32)#None is placeholder for any number of cols. 
                                                   #cannot work with tensors of shape=(3,)
y = tf.placeholder(shape=(3,None),dtype=tf.float32) 
z = x + y
with tf.Session() as sess2:
    #print(sess2.run(z, feed_dict={x:[3,1,2], y:[4,3,5]}))
    print(sess2.run(z, feed_dict={x:[[3],[1],[2]], y:[[4],[3],[5]]}))
    print(sess2.run(z, feed_dict={x:[[3,2],[1,3],[2,3]], y:[[4,3],[3,5],[5,4]]}))

                    #########################
a = tf.linspace(-3.0,3.0,100)
b = tf.linspace(-3.0,3.0,100)
print(a,b)
with tf.Session() as sess3:
    print(sess3.run(a))
g = tf.get_default_graph()
print([op.name for op in g.get_operations()])
print(g.get_tensor_by_name('LinSpace:0'))

                    #########################
g1=tf.Graph()#rarely useful
var = tf.Variable(tf.random_uniform(shape=(3,4), dtype = tf.float32))
print(var)
with tf.Session(graph=g) as sess4:
    sess4.run(tf.global_variables_initializer())
    print(sess4.run(var))

                ############### Linear Regression ##################
# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
print(loss,optimizer,train)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, feed_dict={x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
