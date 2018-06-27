import tensorflow as tf

class Model:

    n_frequency = 64
    n_frame = 64        # 48000[Hz] * 0.1[s] / 128 * 2 ~= 64
    n_frequency_pooled = n_frequency // 16
    n_frame_pooled = n_frame // 16
    border = 0.5

    n_channel_conv1 = 8
    n_channel_conv2 = 4
    n_channel_conv2_flat = n_frequency_pooled * n_frame_pooled * n_channel_conv2
    n_channel_fullconnect1 = 16

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, Model.n_frame, Model.n_frequency, 1])
        self.y_ = tf.placeholder(tf.float32, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.y = self.create_model1()
        self.sess = tf.InteractiveSession()

    def initialize_variables(self):
        tf.global_variables_initializer().run()

    def setup_training(self):
    #    cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y) + (1 - self.y_) * tf.log(1 - self.y))
    #    self.train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        least_square = tf.reduce_sum(tf.square(self.y_ - self.y))
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(least_square)

    def setup_accuracy(self):
        diff = tf.abs(self.y_ - self.y)
        inference = tf.less(diff, Model.border)
        self.accuracy = tf.reduce_mean(tf.cast(inference, tf.float32))

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "fingersnap_model")

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "fingersnap_model")

    def run(self, batch_x):
        return self.sess.run(self.y, feed_dict={self.x: batch_x, self.keep_prob:1.0})

    def exec_training(self, batch_x, batch_y, keep_prob = 0.5):
        self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob:keep_prob})

    def calc_accuracy(self, batch_x, batch_y):
        return self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y_:batch_y, self.keep_prob:1.0})

    def create_model1(self):
        W_conv1 = self.weight_variable([5, 5, 1, Model.n_channel_conv1])
        b_conv1 = self.bias_variable([Model.n_channel_conv1])
        h_conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1) + b_conv1)

        h_pool1 = self.max_pool_4x4(h_conv1)

        W_conv2 = self.weight_variable([5, 5, Model.n_channel_conv1, Model.n_channel_conv2])
        b_conv2 = self.bias_variable([Model.n_channel_conv2])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = self.max_pool_4x4(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, Model.n_channel_conv2_flat])

        W_fullconnect1 = self.weight_variable([Model.n_channel_conv2_flat, Model.n_channel_fullconnect1])
        b_fullconnect1 = self.bias_variable([Model.n_channel_fullconnect1])
        h_fullconnect1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fullconnect1) + b_fullconnect1)

        h_fullconnect1_drop = tf.nn.dropout(h_fullconnect1, self.keep_prob)

        W_fullconnect2 = self.weight_variable([Model.n_channel_fullconnect1, 1])
        b_fullconnect2 = self.bias_variable([1])
        output = tf.nn.sigmoid(tf.matmul(h_fullconnect1_drop, W_fullconnect2) + b_fullconnect2)

        return tf.reshape(output, [-1])

    def create_model2(self):
        n_pool_flat = Model.n_frame * Model.n_frequency // 16
        n_fc1 = 64

        h_pool = self.max_pool_4x4(self.x)
        h_pool_flat = tf.reshape(h_pool, [-1, n_pool_flat])

        W_fc1 = self.weight_variable([n_pool_flat, n_fc1])
        b_fc1 = self.bias_variable([n_fc1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([n_fc1, 1])
        b_fc2 = self.bias_variable([1])

        return tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    def create_model3(self):
        n_x_flat = Model.n_frame * Model.n_frequency
        n_fc1 = 32

        x_flat = tf.reshape(self.x, [-1, n_x_flat])

        W_fc1 = self.weight_variable([n_x_flat, n_fc1])
        b_fc1 = self.bias_variable([n_fc1])
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([n_fc1, 1])
        b_fc2 = self.bias_variable([1])

        return tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_4x4(self, x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
