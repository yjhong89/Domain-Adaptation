import numpy as np
import os
import tensorflow as tf
import utils
import pickle

class WGAN_Model():
    def __init__(self, args, sess, name=None, mnist=None):
        self.args = args
        self.sess = sess
        if name is not None:
            self.name = name
        if mnist is not None:
            self.mnist = mnist
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

        self.build_model()

    def build_model(self):
        print('Build WGAN Model')
        self.source_x = tf.placeholder(tf.float32, [None, 28,28,3], name='source_input')
        self.target_x = tf.placeholder(tf.float32, [None, 28,28,3], name='target_input')
        self.label = tf.placeholder(tf.float32, [None, self.args.num_classes])
        self.is_train = tf.placeholder(tf.bool, [])

        self.source_features = self.generator(self.source_x, self.args.use_bn, reuse=False, training=self.is_train)
        self.target_features = self.generator(self.target_x, self.args.use_bn, reuse=True, training=self.is_train)

        self.discriminator_real = self.discriminator(self.source_features, reuse=False, training=self.is_train)
        self.discriminator_fake = self.discriminator(self.target_features, reuse=True, training=self.is_train)

        self.classifier_logits = tf.cond(self.is_train, lambda: self.classifier(self.source_features, reuse=False), lambda: self.classifier(self.target_features, reuse=True))
        self.classify_prob = tf.nn.softmax(self.classifier_logits)

        self.d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.c_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

        if self.args.gp:
            self.epsilon = np.random.uniform(0,1, [self.args.batch_size, 1])
            self.penalty_point = self.real_images * self.epsilon + self.g * (1 - self.epsilon) # [batch, 64, 64, 3]
            self.gradient = tf.gradients(tf.reduce_mean(self.discriminator(self.penalty_point, reuse=True)), self.d_param) # Returns a list of gradients in self.d_param, mean gradient of examples in batch
            self.gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(self.gradient)))
            self.discriminator_loss = tf.reduce_mean(self.discriminator_fake - self.discriminator_real) + self.args.penalty_hyperparam * (self.gradient_norm - 1)

        else:
            self.discriminator_loss = tf.reduce_mean(self.discriminator_fake - self.discriminator_real)

        self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.classifier_logits, labels=self.label))
        self.total_loss = self.classify_loss - self.args.weight_gan*self.discriminator_loss 

        self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)

        d_grads = self.optimizer.compute_gradients(self.discriminator_loss, var_list=self.d_param)
        gc_grads = self.optimizer.compute_gradients(self.total_loss, var_list=[self.g_param, self.c_param])
        # Discriminator gradient
        self.d_optimizer = self.optimizer.apply_gradients(d_grads)
        # Generator gradient
        self.gc_optimizer = self.optimizer.apply_gradients(gc_grads)
        self.label_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, axis=1), tf.argmax(self.classify_prob, axis=1)), tf.float32))
        self.clipping_op = [disc_param.assign(tf.clip_by_value(disc_param, -self.args.clip, self.args.clip)) for disc_param in self.d_param]

        self.saver = tf.train.Saver(max_to_keep=5)

    def generator(self, inp_, use_batchnorm, reuse, training=True):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            if use_batchnorm:
                de1 = deconv2d(z_reshaped, output_shape=[self.args.batch_size, self.args.target_size // 8, self.args.target_size // 8, self.args.final_dim * 4], name='gen_batch1')
                de1_result = relu_with_batch(de1, training, name='relu_batch1')
                de2 = deconv2d(de1_result, output_shape=[self.args.batch_size, self.args.target_size // 4, self.args.target_size // 4, self.args.final_dim *2], name='gen_batch2')
                de2_result = relu_with_batch(de2, training, name='relu_batch2')
                de3 = deconv2d(de2_result, output_shape=[self.args.batch_size, self.args.target_size // 2, self.args.target_size // 2, self.args.final_dim], name='gen_batch3')
                de3_result = relu_with_batch(de3, training, name='relu_batch3')
                de4 = deconv2d(de3_result, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channels], name='gen_batch4')

                return tf.nn.tanh(de4, name='generator_result')
            else:
                conv1 = utils.conv2d(inp_, output_dim=32, filter_len=5, stride=2, name='gen_wo_batch1', activation=tf.nn.relu)
                conv2 = utils.conv2d(conv1, output_dim=48, filter_len=5, stride=2, name='gen_wo_batch2', activation=tf.nn.relu)
                conv3 = utils.conv2d(conv2, output_dim=64, filter_len=3, stride=1, name='gen_wo_batch3', activation=tf.nn.relu)
                flatten = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
                features = utils.fc(flatten, output_dim=self.args.final_dim, name='fc', activation=tf.nn.tanh)

                return features

    # In WGAN, discriminator or critic does not output probability anymore(That`s why it is called as critic) So just output logits
    def discriminator(self, features, reuse, training=True):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            fc1_d = utils.fc(features, output_dim=100, name='fc1', activation=tf.nn.relu)
            fc2_d = utils.fc(fc1_d, output_dim=50, name='fc2', activation=tf.nn.relu)
            fc3_d = utils.fc(fc2_d, output_dim=10, name='fc3', activation=tf.nn.relu)
            logits = utils.fc(fc3_d, output_dim=1, name='fc4', activation=None)

            return logits 

    def classifier(self, features, reuse):
        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            fc1 = utils.fc(features, output_dim=100, name='fc1', activation=tf.nn.relu)
            fc2 = utils.fc(fc1, output_dim=100, name='fc2', activation=tf.nn.relu)
            logits = utils.fc(fc2, output_dim=self.args.num_classes, name='fc3', activation=None)

            return logits


    def train(self):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print('Checkpoint loaded')
        else:
            print('No ckpt')

        source_batch = utils.batch_generator([self.mnist_train, self.mnist.train.labels], self.args.batch_size)
        target_batch = utils.batch_generator([self.mnistm_train, self.mnist.train.labels], self.args.batch_size)

        for epoch in range(self.args.num_epoch):
            for train_critic_step in range(self.args.train_critic):
                s_x, s_label = source_batch.next()
                t_x, t_label = target_batch.next()
                s_x = (s_x - self.pixel_mean) / 255
                t_x = (t_x - self.pixel_mean) / 255
                feed_dict = {self.source_x:s_x, self.target_x:t_x} 
                disc_loss, _ = self.sess.run([self.discriminator_loss, self.d_optimizer], feed_dict=feed_dict)
                if not self.args.gp:
                     self.sess.run(self.clipping_op)
                print('[%d/%d] loss : %3.4f' % (train_critic_step+1, epoch+1, disc_loss))
            feed_dict = {self.source_x:s_x, self.target_x:t_x, self.label:s_label, self.is_train:True} 
            total_loss_, label_acc, _ = self.sess.run([self.total_loss, self.label_accuracy, self.gc_optimizer], feed_dict=feed_dict)
 
            if np.mod(epoch+1, self.args.eval_interval) == 0:
                s_acc, t_acc = self.evaluate()
                if t_acc > self.args.aim:
                    break

            print('Epoch %d, total_loss: %3.4f, label_accuracy: %3.4f' % (epoch+1, total_loss_, label_acc))

    def evaluate(self):
        m_test = (self.mnist_test - self.pixel_mean) / 255
        mm_test = (self.mnistm_test - self.pixel_mean) / 255
        source_acc = self.sess.run(self.label_accuracy, feed_dict={self.source_x:m_test, self.target_x:mm_test, self.label:self.mnist.test.labels, self.is_train:True})
        target_acc = self.sess.run(self.label_accuracy, feed_dict={self.source_x:m_test, self.target_x:mm_test, self.label:self.mnist.test.labels, self.is_train:False})
        print('Source domain accuracy: %3.4f, Target domain accuracy: %3.4f' % (source_acc, target_acc))
        return source_acc, target_acc
 
    @property
    def model_dir(self):
        if not self.args.gp:
            return '{}_{}batch'.format(self.args.model_type, self.args.batch_size)
        if self.args.improved:
            return '{}_{}batch_gp'.format(self.args.model_type, self.args.batch_size)

    def save(self, global_step):
        model_name='WGAN'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Checkpoint saved at %d steps' % global_step)

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
