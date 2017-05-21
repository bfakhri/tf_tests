import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


## Constants
MAX_TRAIN_STEPS = 1000

## Setup the graph
# Input placeholder
X = tf.placeholder(tf.float32, [None, 28*28], name="Image")
# Groundtruth placeholder
Y_gt = tf.placeholder(tf.float32, [None, 10])
# Setup network 
W1 = tf.Variable(tf.random_normal([28*28, 10], stddev=0.1))
b1 = tf.Variable(tf.random_normal([10], stddev=0.1))
# Output of the network 
Y_pred = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 28*28]), W1) + b1)
# Loss function
cross_entropy_loss = - tf.reduce_sum(Y_gt * tf.log(Y_pred))

# Metrics
correct_vector = tf.equal(tf.argmax(Y_gt, 1), tf.argmax(Y_pred, 1))
percent_correct = tf.reduce_mean(tf.cast(correct_vector, tf.float32))

# Setup Optimizer and calculate training step 
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy_loss)


# Init the variables and the tf session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training Loop
for step in range(MAX_TRAIN_STEPS):
	# Get batch 
	batch_X, batch_Y = mnist.train.next_batch(100)
	train_data = {X: batch_X, Y_gt: batch_Y}

	# Run the graph to calculate the training step
	sess.run(train_step, feed_dict=train_data)

	# Calculate the accuracy and loss
	acc, loss = sess.run([percent_correct, cross_entropy_loss], feed_dict=train_data)

	# Get test data metrics
	test_data = {X: mnist.test.images, Y_gt: mnist.test.labels}
	acc, loss = sess.run([percent_correct, cross_entropy_loss], feed_dict=test_data)

	# Print out relevant information
	print("Step: " + str(step) + "\tAcc: " + str(acc) + "\tLoss: " + str(loss))
	
