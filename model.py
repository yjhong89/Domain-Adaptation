import tensorflow as tf
import numpy as np
import utils

def Model:
    def __init__(self, args, sess, name=None):
        self.args = args
        self.sess = sess
        if name is not None:
            self.name = name
        
        if self.args.model_domain == 'MNIST':
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            

        # Process MNIST and MNIST-M
        self.flip_gradient = utils.FlipGradient()
        self.build_model()

    def build_model(self):
        # Image placeholder
        self.x = tf.placeholder(tf.float32, [None, 28,28,3])
        # Label placeholder(one-hot)
        self.label = tf.placeholder(tf.float32, [None, 10])
        # Domain placeholder(source, target)
        self.domain = tf.placeholder(tf.int32, [None, 2])
        self.is_training = tf.placeholder(tf.bool, [])

        '''
            Model is composed of 3 components:
                1. Feature extractor 2. Label classifier 3. Domain discriminator
        '''
        with tf.variable_scope('Feature_Extractor'):
            conv1 = utils.conv2d(self.x, output_dim=32, filter_len=5, stride=2, name='conv1', activation=tf.nn.relu)  
            conv2 = utils.conv2d(conv1, output_dim=48, filter_len=5, stride=2, name='conv2', activation=tf.nn.relu)
            flatten = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            self.feature = utils.fc(flatten, output_dim=self.args.final_dim, name='fc', activation=None)

        with tf.variable_scope('Label_Classify'):
            '''
                Only source data when training, only target data when test
            '''
            # tf.slice(input_, begin, size) : 'begin' shape is same as input_
            source_features = lambda: tf.slice(self.feature, [0,0], [self.args.batch_size / 2, -1]) 
            classify_input = tf.cond(self.is_training, source_features, lambda: self.feature)
            source_label = lambda: tf.slice(self.label, [0, 0], [self.args.batch_size / 2, -1])
            self.classify_label = tf.cond(self.is_training, source_label, lambda: self.label)
            fc1 = utils.fc(classify_input, output_dim=100, name='fc1', activation=tf.nn.relu)
            fc2 = utils.fc(fc1, output_dim=50, name='fc2', activation=tf.nn.relu)
            logits = utils.fc(fc2, output_dim=self.args.num_classes, name='fc3', activation=None)
        
            # Probability for correctly predicting label
            prob = tf.nn.softmax(logits)
            # one-hot encoded label
            cross_ent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=source_label)

        with tf.variable_scope('Domain_Discriminator'):
            # Flip the gradient when back-propagating
            flip_op_feature = self.flip_gradient(self.feature)

            fc1_d = utils.fc(flip_op_features, output_dim=100, name='fc1', activation=tf.nn.relu)
            fc2_d = utils.fc(fc1_d, output_dim=50, name='fc2', activation=tf.nn.relu)
            logits_d = utils.fc(fc2_d, output_dim=2, name='fc3', activation=None)

            prob_d = tf.nn.softmax(logits_d)
            cross_ent_loss_d = tf.nn.softmax_cross_entropy_with_logits(logits=logits_d, labels=self.domain)

        # Normal classify loss
        self.classify_loss = tf.reduce_mean(cross_ent_loss)
        # Optional regularizer
        self.domain_loss = tf.reduce_mean(cross_ent_loss_d)
        self.total_loss = self.classify_loss + self.domain_loss

        self.regular_train_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.classify_loss)
        self.dann_train_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.total_loss)

        # tf.equal(x,y) returns boolean, true if x==y
        self.label_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.classify_label, axis=1), tf.argmax(prob, axis=1))), tf.float32)
        self.domain_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.domain, axis=1), tf.argmax(prob_d, axis=1))), tf.float32)

        self.saver = tf.Saver()

    def train(self):
        if self.load():
            print('Checkpoint loaded')
        else:
            print('No ckpt')

        source_batch = utils.batch_generator()
        target_batch = utils.batch_generator()
        # np.tile(array, reps) -> source : [1,0], target : [0,1]
        domain_label = np.vstack([np.tile([1,0], [self.args.batch_size/2, 1]), np.tile([0,1], [self.args.batch_size/2, 1])])
                    
        for epoch in range(self.args.num_epoch):
            s_x, s_label = source_batch.next()
            t_x, t_label = target_batch.next()

            data_x = np.vstack([s_x, t_x])
            data_label = np.vstack([s_label, t_label])

            feed_dict = {self.x:data_x, self.label:data_label, self.domain:domain_label, self.is_training:True})
            _, loss_, label_acc, domain_acc = self.sess.run([self.dann_train_op, self.total_loss, self.label_accuracy, self.domain_accuracy], feed_dict=feed_dict)

            if (epoch+1) % self.args.eval_interval == 0:
                self.evaluate()
            

    def evaluate(self):
        source_acc = self.sess.run(self.label_accuracy, feed_dict={self.x:, self.label:, self.is_training:False})
        target_acc = self.sess.run(self.label_accuracy, feed_dict={self.x:, self.label:, seef.is_training:False})
        print('Source domain accuracy: %3.4f, Target domain accuracy: %3.4f' % (source_acc, target_acc))

