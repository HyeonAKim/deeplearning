# 1
# Load library and load data
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.shape)
# ( 55000, 784 ) 784 사이즈의 55000건의 데이터가 있음을 말함

# 2
# Define Model and Solver Class
# Object-Oriented-Programming allow to define multiple model easily.
# Why do we separeate model and solver classes?
# We can just swap out the model class in the solver class when we need a different network architecture.
# Usually we need one solver class.

class Model:
    """Network Model class

    Note that class has only the constructor.
    The actual model is defined inside the constructor.

    Attributes
    ----------
    X : tf.float32
        This is a tensorflow_tutorial placholder for MNIST images
        Expected shape is (None, 784)

    y : tf.float32
        This is a tensorflow_tutorial placeholder for MNIST labels
        Expected shape is (None, 10)

    mode : tf.bool
        This is used for the batch normalization.
        It's 'True' at training time and 'False' at test time.

    loss : tf.float32
        The loss function is a softmax cross entropy.

    train_op
        This is simply the training op that minimizes the loss.

    accuracy : tf.float32
        The accuracy operation

    """
    def __init__(self, name, input_dim, output_dim, hidden_dims=[32,32],
                 use_batchnorm=True, activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer, lr= 0.01):
        """ Constructor

        :param name: str
            The name of this network
            The entire network will be created under 'tf.variable_scope(name)'

        :param input_dim: int
            The input dimension
            In this example, 784

        :param output_dim: int
            The number of output labels
            There are 10 labels

        :param hidden_dims: List ( default:[32, 32])
            len(hidden_dims) = number of layers
            each element is the number of hidden units

        :param use_batchnorm: bool ( default : True )
            If true, it will create the batchmormaliztion Layer

        :param activation_fn: TF Functions ( default: tf.nn.relu )
            Activation Function

        :param optimizer: TF optimizer ( default : tf.train.AdamOptimizer)
            Optimizer Function

        :param lr: float (default : 0.01)
            Learning rate
        """
        with tf.variable_scope(name):
            # placedholders are defined
            self.X = tf.placeholder(tf.float32, [None, input_dim], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_dim], name='y')
            self.mode = tf.placeholder(tf.bool, name='train_mode')

            # Loop over hidden layers
            net = self.X
            for i, h_dim in enumerate(hidden_dims):
                with tf.variable_scope('layer{}'.format(i)):
                    net = tf.layers.dense(net, h_dim)

                    if use_batchnorm:
                        net = tf.layers.batch_normalization(net, training=self.mode)

                    net = activation_fn(net)

            # Attach fully connected layers
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, output_dim)

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=self.y)
            self.loss = tf.reduce_mean(self.loss, name='loss')

            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer(lr).minimize(self.loss)

            # Accuracy etc
            softtmax = tf.nn.softmax(net, name='softmax')
            self.accuracy = tf.equal(tf.argmax(softtmax, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

class Solver:
    """Solver class

    This class  will contain the model class and session

    Attrivutes
    ---------
    model : Model class
    sess : TF session

    Methods
    -------
    train(X,y)
        Run the train_op and Returns the loss

    evaluate(X, y, batch_size=None)
        Returns "Loss" and "Accuracy"
        If batch_size is given, it's computed using batch_size
        becaues most GPU memorices cannot handle entire training data at once


    Example
    -------
    >>> sess = tf.InteractiveSession()
    >>> model = Model("BatchNorm", 32, 10)
    >>> solver = Solver(sess, model)

    # Train
    >>> solver.train(X, y)

    # Evaluate
    >>> solver.evaluate(X, y)
    """

    def __init__(self, sess, model):
        self.model = model
        self.sess = sess

    def train(self, X, y):
        feed = {
            self.model.X: X,
            self.model.y: y,
            self.model.mode: True
        }
        train_op = self.model.train_op
        loss = self.model.loss

        return self.sess.run([train_op, loss], feed_dict=feed)

    def evaluate(self, X, y, batch_size=None):
        if batch_size:
            N = X.shape[0]

            total_loss = 0
            total_acc = 0

            for i in range(0, N, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                feed = {
                    self.model.X: X_batch,
                    self.model.y: y_batch,
                    self.model.mode: False
                }

                loss = self.model.loss
                accuracy = self.model.accuracy

                step_loss, step_acc = self.sess.run([loss, accuracy],
                                                    feed_dict=feed)
                total_loss += step_loss * X_batch.shape[0]
                total_acc += step_acc * X_batch.shape[0]

            total_loss /= N
            total_acc /= N

            return total_loss, total_acc

        else:
            feed = {
                self.model.X : X,
                self.model.y : y,
                self.model.mode: False
            }
            loss = self.model.loss
            accuracy = self.model.accuracy

            return self.sess.run([loss, accuracy], feed_dict=feed)

# 3
# Instantiate Model/Solver classes

input_dim = 784
output_dim = 10
N = 55000

tf.reset_default_graph()
sess = tf.InteractiveSession()

# We create two models : one with the batch norm and other withour
bn = Model('batchnorm', input_dim, output_dim, use_batchnorm=True)
nn = Model('no_norm', input_dim, output_dim, use_batchnorm=False)

# We create two solvers: to train both models at the same time for comparison
# Usuaaly we only need one solver class
bn_solver = Solver(sess, bn)
nn_solver = Solver(sess, nn)

epoch_n = 10
batch_size = 32

# Save Lossess and Accuracies every epoch
# We are going to plot them later
train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

# 4
# Run the train step
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(epoch_n):
    for _ in range(int(N/batch_size)):
        X_batch, y_batch = mnist.train.next_batch(batch_size)

        _, bn_loss = bn_solver.train(X_batch, y_batch)
        _, nn_loss = nn_solver.train(X_batch, y_batch)

    b_loss, b_acc = bn_solver.evaluate(mnist.train.images, mnist.train.labels, batch_size)
    n_loss, n_acc = nn_solver.evaluate(mnist.train.images, mnist.train.labels, batch_size)

    # Save train losses / acc
    train_losses.append([b_loss, n_loss])
    train_accs.append([b_acc, n_acc])
    print(f'[Epoch {epoch}-TRAIN] Batchnorm Loss(Acc): {b_loss:.5f}({b_acc:.2%}) vs No Batchnorm Loss(Acc): {n_loss:.5f}({n_acc:.2%})')

    b_loss, b_acc = bn_solver.evaluate(mnist.validation.images, mnist.validation.labels)
    n_loss, n_ass = nn_solver.evaluate(mnist.validation.images, mnist.validation.labels)

    # Save valid lossed/acc
    valid_losses.append([b_loss, n_loss])
    valid_accs.append([b_acc, n_acc])
    print(f'[Epoch {epoch}-VALID] Batchnorm Loss(Acc): {b_loss:.5f}({b_acc:.2%}) vs No Batchnorm Loss(Acc): {n_loss:.5f}({n_acc:.2%})')
