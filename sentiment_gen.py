import tensorflow as tf
import numpy as np
from random import randint
from data_helpers import batch_iter, load_data, string_to_int
import os
import time
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import pickle
from tqdm import tqdm

class Sentiment() :
    vocabulary = None
    FLAGS      = None
    LOG_FILE   = None

    def log(self, *string, **kwargs):
        output = ' '.join(string)
        if kwargs.pop('verbose', True):
            print output
        #self.LOG_FILE.write(''.join(['\n', output]))


    def weight_variable(self,shape, name):
        """
        Creates a new Tf weight variable with the given shape and name.
        Returns the new variable.
        """
        var = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(var, name=name)


    def bias_variable(self,shape, name):
        """
        Creates a new Tf bias variable with the given shape and name.
        Returns the new variable.
        """
        var = tf.constant(0.1, shape=shape)
        return tf.Variable(var, name=name)


    def human_readable_output(a_batch):
        """
        Feeds a batch to the network and prints in a human readable format a
        comparison between the batch's labels and the network output.
        Outputs comparison to stdout.
        """
        log('Network output on random data...')
        sentences = zip(*a_batch)[0]
        word_sentence = []
        network_result = self.sess.run(tf.argmax(network_out, 1),
                                  feed_dict={self.data_in: zip(*a_batch)[0],
                                             self.dropout_keep_prob: 1.0})
        actual_result = self.sess.run(tf.argmax(data_out, 1),
                                 feed_dict={data_out: zip(*a_batch)[1]})
        # Translate the string to ASCII (remove <PAD/> symbols)
        for s in sentences:
            output = ''
            for w in s:
                output += vocabulary_inv[w.astype(np.int)][0] + ' '
            output = output.translate(None, '<PAD/>')
            word_sentence.append(output)
        # Output the network result
        for idx, item in enumerate(network_result, start=0):
            network_sentiment = 'POS' if item == 1 else 'NEG'
            actual_sentiment = 'POS' if actual_result[idx] == 1 else 'NEG'

            if item == actual_result[idx]:
                status = '\033[92mCORRECT\033[0m'
            else:
                status = '\033[91mWRONG\033[0m'

            log('\n%s\nLABEL: %s - OUTPUT %s | %s' %
                (word_sentence[idx], actual_sentiment, network_sentiment, status))




    def init_hyperparameters(self):

        # Hyperparameters
        tf.flags.DEFINE_boolean('train', False,
                                'Train the network (default: False)')
        tf.flags.DEFINE_boolean('save', False,
                                'Save session checkpoints (default: False)')
        tf.flags.DEFINE_boolean('save_protobuf', False,
                                'save model as binary protobuf (default: False)')
        tf.flags.DEFINE_boolean('evaluate_batch', False,
                                'Evaluate the network on a held-out batch from the '
                                'dataset and print the results (for '
                                'debugging/educational purposes)')
        tf.flags.DEFINE_string('load', None,
                               'Restore a model from the given path.')
        tf.flags.DEFINE_string('device', 'cpu',
                               'Device to use (can be either \'cpu\' or \'gpu\').')
        tf.flags.DEFINE_string('custom_input', '',
                               'Evaluate the model on the given string.')
        tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                               'Comma-separated filter sizes for the convolution layer '
                               '(default: \'3,4,5\')')
        tf.flags.DEFINE_integer('embedding_size', 128,
                                'Size of the word embeddings (default: 128)')
        tf.flags.DEFINE_integer('num_filters', 128,
                                'Number of filters per filter size (default: 128)')
        tf.flags.DEFINE_integer('batch_size', 128, 'Batch size (default: 128)')
        tf.flags.DEFINE_integer('epochs', 3, 'Number of training epochs (default: 3)')
        tf.flags.DEFINE_integer('valid_freq', 1,
                                'Check model accuracy on validation set '
                                '[VALIDATION_FREQ] times per epoch (default: 1)')
        tf.flags.DEFINE_integer('checkpoint_freq', 1,
                                'Save model [CHECKPOINT_FREQ] times per epoch '
                                '(default: 1)')
        tf.flags.DEFINE_float('dataset_fraction', 1.0,
                              'Fraction of the dataset to load in memory, to reduce '
                              'memory usage (default: 1.0; uses all dataset)')
        tf.flags.DEFINE_float('test_data_ratio', 0.1,
                              'Fraction of the dataset to use for validation (default: '
                              '0.1)')
        self.FLAGS = tf.flags.FLAGS

    def init_path(self):

        # File paths
        OUT_DIR = os.path.abspath(os.path.join(os.path.curdir, 'output'))
        RUN_ID = time.strftime('run%Y%m%d-%H%M%S')
        self.RUN_DIR = os.path.abspath(os.path.join(OUT_DIR, RUN_ID))
        LOG_FILE_PATH = os.path.abspath(os.path.join(self.RUN_DIR, 'log.log'))

        if self.FLAGS.load is not None:
            CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(self.FLAGS.load, 'ckpt.ckpt'))
        else:
            CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(self.RUN_DIR, 'ckpt.ckpt'))
        #os.mkdir(self.RUN_DIR)
        self.SUMMARY_DIR = os.path.join(self.RUN_DIR, 'summaries')
        #self.LOG_FILE = open(LOG_FILE_PATH, 'a', 0)

    def init(self):

        self.init_hyperparameters()
        self.init_path()

        self.log('======================= Loading data and model')
        # Load data
        #self.x, y, self.vocabulary, vocabulary_inv = load_data(self.FLAGS.dataset_fraction)
        #x, y, vocabulary, vocabulary_inv = load_data(self.FLAGS.dataset_fraction)


        pickle_in = open("pickle/x.pickle","rb")
        self.x = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("pickle/y.pickle","rb")
        y = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("pickle/vocabulary.pickle","rb")
        self.vocabulary = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("pickle/vocabulary_inv.pickle","rb")
        vocabulary_inv = pickle.load(pickle_in)
        pickle_in.close()

        print('\n*********** Pickle files loaded \n')


        # Randomly shuffle data
        np.random.seed(123)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = self.x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        text_percent = self.FLAGS.test_data_ratio
        test_index = int(len(self.x) * text_percent)
        x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
        y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

        # Parameters
        self.sequence_length = x_train.shape[1]
        self.num_classes = y_train.shape[1]
        self.vocab_size = len(self.vocabulary)
        self.filter_sizes = map(int, self.FLAGS.filter_sizes.split(','))
        self.validate_every = len(y_train) / (self.FLAGS.batch_size * self.FLAGS.valid_freq)
        self.checkpoint_every = len(y_train) / (self.FLAGS.batch_size * self.FLAGS.checkpoint_freq)

        # Set computation device
        if self.FLAGS.device == 'gpu':
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

        # Log run data
        self.log('\nFlags:')
        #for attr, value in sorted(self.FLAGS.__flags.iteritems()):
        #    self.log('\t%s = %s' % (attr, value._value))
        self.log('\nDataset:')
        self.log('\tTrain set size = %d\n'
            '\tTest set size = %d\n'
            '\tVocabulary size = %d\n'
            '\tInput layer size = %d\n'
            '\tNumber of classes = %d' %
            (len(y_train), len(y_test), len(self.vocabulary), self.sequence_length, self.num_classes))
        self.log('\nOutput folder:', self.RUN_DIR)


    def init_run(self):

        # Session
        self.sess = tf.InteractiveSession()

        # Network
        with tf.device(self.device):
            # Placeholders
            self.data_in = tf.placeholder(tf.int32, [None, self.sequence_length], name='data_in')
            data_out = tf.placeholder(tf.float32, [None, self.num_classes], name='data_out')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            # Stores the accuracy of the model for each batch of the validation testing
            valid_accuracies = tf.placeholder(tf.float32)
            # Stores the loss of the model for each batch of the validation testing
            valid_losses = tf.placeholder(tf.float32)

            # Embedding layer
            with tf.name_scope('embedding'):
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.FLAGS.embedding_size],
                                                  -1.0, 1.0),
                                name='embedding_matrix')
                embedded_chars = tf.nn.embedding_lookup(W, self.data_in)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            # Convolution + ReLU + Pooling layer
            pooled_outputs = []
            for i, self.filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % self.filter_size):
                    # Convolution Layer
                    self.filter_shape = [self.filter_size,
                                    self.FLAGS.embedding_size,
                                    1,
                                    self.FLAGS.num_filters]
                    W = self.weight_variable(self.filter_shape, name='W_conv')
                    b = self.bias_variable([self.FLAGS.num_filters], name='b_conv')
                    conv = tf.nn.conv2d(embedded_chars_expanded,
                                        W,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='conv')
                    # Activation function
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # Maxpooling layer
                    ksize = [1,
                             self.sequence_length - self.filter_size + 1,
                             1,
                             1]
                    pooled = tf.nn.max_pool(h,
                                            ksize=ksize,
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name='pool')
                pooled_outputs.append(pooled)

            # Combine the pooled feature tensors
            num_filters_total = self.FLAGS.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Dropout
            with tf.name_scope('dropout'):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

            # Output layer
            with tf.name_scope('output'):
                W_out = self.weight_variable([num_filters_total, self.num_classes], name='W_out')
                b_out = self.bias_variable([self.num_classes], name='b_out')
                self.network_out = tf.nn.softmax(tf.matmul(h_drop, W_out) + b_out)

            # Loss function
            cross_entropy = -tf.reduce_sum(data_out * tf.log(self.network_out))

            # Training algorithm
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            # Testing operations
            correct_prediction = tf.equal(tf.argmax(self.network_out, 1),
                                          tf.argmax(data_out, 1))
            # Accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # Validation ops
            valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
            valid_mean_loss = tf.reduce_mean(valid_losses)


        self.log('Data processing OK, creating network...')
        self.sess.run(tf.global_variables_initializer())

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('Training loss', cross_entropy)
        valid_loss_summary = tf.summary.scalar('Validation loss', valid_mean_loss)
        valid_accuracy_summary = tf.summary.scalar('Validation accuracy',
                                                   valid_mean_accuracy)
        summary_writer = tf.summary.FileWriter(self.SUMMARY_DIR, self.sess.graph)
        tf.summary.merge_all()


    def evaluate(self, input_string):

        # Evaluate custom input
        self.log('Evaluating custom input:', input_string)
        self.evaluate_sentence(input_string)


    def evaluate_sentence( self, sentence):
        """
        Translates a string to its equivalent in the integer vocabulary and feeds it
        to the network.
        Outputs result to stdout.
        """

        #x_to_eval = string_to_int(sentence, self.vocabulary, max(len(_) for _ in self.x))
        #x_to_eval = string_to_int(sentence, self.vocabulary, max(len(_) for _ in self.x))
        x_to_eval = string_to_int(sentence, self.vocabulary, max(len(_) for _ in self.x))
        result = self.sess.run(tf.argmax(self.network_out, 1),
                          feed_dict={self.data_in: x_to_eval,
                                     self.dropout_keep_prob: 1.0})
        unnorm_result = self.sess.run(self.network_out, feed_dict={self.data_in: x_to_eval,
                                                         self.dropout_keep_prob: 1.0})
        network_sentiment = 'POS' if result == 1 else 'NEG'
        self.log('Custom input evaluation:', network_sentiment)
        self.log('Actual output:', str(unnorm_result[0]))

        result = network_sentiment
        return result
# end class

if __name__ == '__main__' :
        sentiment = Sentiment()
        sentiment.init()
        sentiment.init_run()

        sentiment.evaluate_sentence( "Ai is awesome cool state of the art superb excellent superior best world class   ")


'''

        pickle_out = open("sentiment.pickle","wb")
        pickle.dump(sentiment, pickle_out)
        pickle_out.close()

        pickle_in = open("sentiment.pickle","rb")
        sentiment2 = pickle.load(pickle_in)

        pickle_out = open("x.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        pickle_out = open("vocabulary.pickle","wb")
        pickle.dump(vocabulary, pickle_out)
        pickle_out.close()

        pickle_out = open("vocabulary_inv.pickle","wb")
        pickle.dump(vocabulary_inv, pickle_out)
        pickle_out.close()
'''
