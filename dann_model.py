import tensorflow as tf
import numpy as np
import utils
import os
import pickle

class DANN_Model():
    def __init__(self, args, sess, name=None):
        self.args = args
        self.sess = sess
        if name is not None:
            self.name = name
        
        if self.args.model_domain == 'MNIST':
            '''
                Loading MNIST 
            '''
            from tensorflow.examples.tutorials.mnist import input_data
            # mnist is 28*28 grayscale images having values between 0~1
            self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            # images > 0 returns boolean, True if element is larger than 0
            # Make 0 or 255, shape of (55000,28,28,1)
            mnist_train = (self.mnist.train.images>0).reshape(55000,28,28,1).astype(np.float32)*255
            # Make RGB 3 channel, white will be (255,255,255) and black will be (0,0,0), therefore concatentate through axis=3
            self.mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], axis=3)
            mnist_test = (self.mnist.test.images>0).reshape(10000,28,28,1).astype(np.float32)*255
            self.mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], axis=3)
            '''
                Loading MNIST-M which images have different background
            '''
            mnistm = pickle.load(open('mnistm_data.pkl'))
            self.mnistm_train = mnistm['train']
            self.mnistm_test = mnistm['test']
            # Normalize, calculate mean value along each channels
            self.pixel_mean = np.vstack([self.mnist_train, self.mnistm_train]).mean((0,1,2))
            

        # Process MNIST and MNIST-M
        self.flip_gradient = utils.FlipGradient()
        self.build_dann_model()

    def build_dann_model(self):
        print('Build DANN Model')	
        # Image placeholder
        self.x = tf.placeholder(tf.float32, [None, 28,28,3])
        # Label placeholder(one-hot)
        self.label = tf.placeholder(tf.float32, [None, self.args.num_classes])
        self.mag = tf.placeholder(tf.float32, [])
        # Domain placeholder(source, target)
        self.domain = tf.placeholder(tf.int32, [None, 2])
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])

        '''
            Model is composed of 3 components:
                1. Feature extractor 2. Label classifier 3. Domain discriminator
        '''
        with tf.variable_scope('Feature_Extractor'):
            conv1 = utils.conv2d(self.x, output_dim=32, filter_len=5, stride=2, name='conv1', activation=tf.nn.relu)  
            conv2 = utils.conv2d(conv1, output_dim=48, filter_len=5, stride=2, name='conv2', activation=tf.nn.relu)
            conv3 = utils.conv2d(conv2, output_dim=48, filter_len=3, stride=1, name='conv3', activation=tf.nn.relu)
            flatten = tf.reshape(conv3, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            self.feature = utils.fc(flatten, output_dim=self.args.final_dim, name='fc', activation=tf.nn.relu)

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
            cross_ent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_label)

        with tf.variable_scope('Domain_Discriminator'):
            # Flip the gradient when back-propagating
            flip_op_feature = self.flip_gradient(self.feature, self.mag)

            fc1_d = utils.fc(flip_op_feature, output_dim=100, name='fc1', activation=tf.nn.relu)
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
        self.dann_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)
        #self.regular_train_op = tf.train.MomentumOptimizer(self.args.learning_rate, 0.9).minimize(self.classify_loss)
        #self.dann_train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.total_loss)
        # tf.equal(x,y) returns boolean, true if x==y
        self.label_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.classify_label, axis=1), tf.argmax(prob, axis=1)), tf.float32))
        self.domain_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.domain, axis=1), tf.argmax(prob_d, axis=1)), tf.float32))

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print('Checkpoint loaded')
        else:
            print('No ckpt')

        source_batch = utils.batch_generator([self.mnist_train, self.mnist.train.labels], self.args.batch_size/2)
        target_batch = utils.batch_generator([self.mnistm_train, self.mnist.train.labels], self.args.batch_size/2)
        source_only_batch = utils.batch_generator([self.mnist_train, self.mnist.train.labels], self.args.batch_size)
        # np.tile(array, reps) -> source : [1,0], target : [0,1]
        domain_label = np.vstack([np.tile([1,0], [self.args.batch_size/2, 1]), np.tile([0,1], [self.args.batch_size/2, 1])])
                    
        for epoch in range(self.args.num_epoch):
            # Source only, domain data is not necessary
            if self.args.model_mode == 'SO':
                src_x, src_label = source_only_batch.next()
                src_x = (src_x - self.pixel_mean)/255
                feed_dict = {self.x:src_x, self.label:src_label, self.is_training:False}
                _, loss_, label_acc = self.sess.run([self.regular_train_op, self.classify_loss, self.label_accuracy], feed_dict=feed_dict)
            elif self.args.model_mode == 'DA':
                # Flip gradient magnitude, learning rate according to paper
                flip_mag = 2/(1+np.exp(-10*(float(epoch)/self.args.num_epoch)))-1
                learning_r = 0.01 / (1+10*(float(epoch)/self.args.num_epoch))**0.75
                s_x, s_label = source_batch.next()
                t_x, t_label = target_batch.next()
    
                data_x = np.vstack([s_x, t_x])
                data_x = (data_x - self.pixel_mean)/255
                data_label = np.vstack([s_label, t_label])
                feed_dict = {self.x:data_x, self.label:data_label, self.domain:domain_label, self.is_training:True, self.mag:flip_mag, self.lr:learning_r}
                _, loss_, label_acc, domain_acc = self.sess.run([self.dann_train_op, self.total_loss, self.label_accuracy, self.domain_accuracy], feed_dict=feed_dict)
            else:
                raise Exception('Not supported mode')
          
            print('Epoch %d, label accuracy: %3.4f, loss: %3.4f' % (epoch+1, label_acc, loss_))             

            if (epoch+1) % self.args.eval_interval == 0:
                self.evaluate()
            

    def evaluate(self):
        m_test = (self.mnist_test - self.pixel_mean) / 255
        mm_test = (self.mnistm_test - self.pixel_mean) / 255
        source_acc = self.sess.run(self.label_accuracy, feed_dict={self.x:m_test, self.label:self.mnist.test.labels, self.is_training:False})
        target_acc = self.sess.run(self.label_accuracy, feed_dict={self.x:mm_test, self.label:self.mnist.test.labels, self.is_training:False})
        print('Source domain accuracy: %3.4f, Target domain accuracy: %3.4f' % (source_acc, target_acc))


    def load(self):
        checkpoint_path = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
            return True
        else:
            return False 
  
    @property
    def model_dir(self):
        return '{}_{}batch'.format(self.args.model_type, self.args.batch_size) 

