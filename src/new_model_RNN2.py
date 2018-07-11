import pandas as pd

import os

os.environ['PYTHONHASHSEED'] = '123'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda, Embedding, Concatenate
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import Sequence, to_categorical
from keras.losses import categorical_crossentropy
from keras import initializers
import random as rn
import numpy as np

import tensorflow as tf
# tf.set_random_seed(123)
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer, InputSpec
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process
# import tensorflow as tf

import re
import keras.callbacks
import sys
import pickle as pkl
# from numpy.random import seed
# from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from numpy import around
from keras.utils import plot_model
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
import itertools, tqdm
# rn.seed(123)
# seed(123)
# set_random_seed(123)





# def get_session():
#     gpu_options = tf.GPUOptions(allow_growth=True)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


verbose = 0

# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


class Evaluation(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if verbose == 1:
            print('epoch [{0:3d}]:  loss: {1:.4f} - sup_loss: {2:.4f} - unsup_loss: {3:.4f} - val_loss: {4:.4f} - val_sup_loss: {5:.4f} - val_unsup_loss: {6:.4f} - sup_accuracy: {7:.4f} val_sup_accuracy: {8:.4f}'.format(epoch, logs['loss'], logs['classification_loss'],
                    logs['clustering_loss'],logs['val_loss'], logs['val_classification_loss'],
                    logs['val_clustering_loss'], logs['classification_m_accuracy'],
                     logs['val_classification_m_accuracy']))
        elif verbose == 2:
            print('epoch [{0:3d}]:  loss: {1:.4f} - val_loss: {2:.4f} - accuracy: {3:.4f} val_accuracy: {4:.4f}'.format(epoch, logs['loss'],
                    logs['val_loss'], logs['classification_m_accuracy'],
                     logs['val_classification_m_accuracy']))



def _cost(true_y,pred_y):
    # mask = K.sum(true_y, -1) > 0
    # true_label = K.boolean_mask(true_y, mask)
    # pred_label = K.boolean_mask(pred_y, mask)

    treshold = 0
    mask = Lambda(lambda x:K.greater_equal(x, treshold))(true_y)
    mask = Lambda(lambda x:K.cast(x, 'float32'))(mask)
    pred_label = Lambda(lambda x: x * mask)(pred_y)
    true_label = Lambda(lambda x: x * mask)(true_y)
    return K.mean(K.categorical_crossentropy( pred_label, true_label))

def m_kl(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values.
    '''
    K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1))

def m_accuracy(true_y,pred_y):
    treshold = 0
    mask = Lambda(lambda x:K.greater_equal(x, treshold))(true_y)
    mask = Lambda(lambda x:K.cast(x, 'float32'))(mask)
    pred_label = Lambda(lambda x: x * mask)(pred_y)
    true_label = Lambda(lambda x: x * mask)(true_y)
    return K.mean(K.equal(K.argmax(true_label, axis=-1), K.argmax(pred_label, axis=-1)))


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, m_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T




def G2V(data, checkpoint, num_walks, walk_length, _dataset,train_mask,val_mask,u_mask,test_mask,l_dim,n_clusters,X, y,features):


    maxlen = walk_length
    max_sentences = num_walks
    update_interval = 1

    X_train, y_train = X[train_mask, :], y[train_mask, :]
    X_test, y_test = X[test_mask, :], y[test_mask, :]
    X_ul, y_ul = X[u_mask, :], y[u_mask, :]
    X_v, y_v = X[val_mask, :], y[val_mask, :]
    X_tu, y_tu = np.vstack((X_train, X_ul)), np.vstack((y_train, -1*np.ones_like(y_ul)))

    filter_length = [3, 2, 2]
    nb_filter = [64, 6, 64]
    pool_length = 2

    # l_dim_arr = [64]#[64,128,32,256]
    # document input
    search_iter = 1
    best = [-1]
    best_config = []
    n_labels = y_train.shape[1]
    # n_clusters_arr = [n_labels]#[n_labels, 10]
    # configs = list(itertools.product(l_dim_arr, n_clusters_arr))

    r_alpha = 0.9
    file_name = _dataset + 'g2v'

    # old_state = rn.getstate()
    # st0=np.random.get_state()
    # K.clear_session()
    rn.seed(123)
    np.random.seed(123)
    tf.reset_default_graph()
    # rn.seed(123)
    # seed(123)
    tf.set_random_seed(123)
    # rn.setstate(old_state)
    # np.random.set_state(st0)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    with sess.as_default():
        document = Input(shape=(max_sentences, maxlen), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(maxlen,), dtype='int64')
        embedded = Embedding(len(data),
                             features.shape[1],
                             weights=[features],
                             # input_length=MAX_SEQUENCE_LENGTH,
                             trainable=False)(in_sentence)
        # embedded: encodes sentence
        # for i in range(len(nb_filter)):
        #     embedded = Conv1D(filters=nb_filter[i],
        #                       kernel_size=filter_length[i],
        #                       padding='valid',
        #                       activation='relu',
        #                       kernel_initializer='glorot_normal',
        #                       strides=1)(embedded)
        #
        #     embedded = Dropout(0.15)(embedded)
        #     embedded = MaxPooling1D(pool_size=pool_length)(embedded)

        bi_lstm_sent = \
            Bidirectional(LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
                embedded)

        # sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
        sent_encode = Dropout(0.25)(bi_lstm_sent)
        # sentence encoder
        encoder = Model(inputs=in_sentence, outputs=sent_encode)
        # encoder.summary()
        # plot_model(encoder, to_file='Images/enoder2.png')
        # exit()

        encoded = TimeDistributed(encoder)(document)
        # encoded: sentences to bi-lstm for document encoding
        b_lstm_doc = \
            Bidirectional(LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
                encoded)

        output_lstm = Dropout(0.25)(b_lstm_doc)



        output_dense1 = Dense(l_dim, activation='relu')(output_lstm)
        output_dense1 = Dropout(0.1)(output_dense1)

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(output_dense1)
        clustering_layer = Dropout(0.1)(clustering_layer)
        # onehot_layer = Lambda(lambda x:  K.one_hot(K.argmax(x,-1),n_clusters))(clustering_layer)

        output = Concatenate(axis=-1)([output_dense1, clustering_layer])



        output = Dense(y.shape[1], activation='softmax', name='classification')(output)

        model = Model(inputs=document, outputs=[output, clustering_layer])



        class DataSequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y, self.p = x_set, y_set, target_distribution(model.predict(x_set, batch_size=128)[1])# np.zeros((x_set.shape[0], n_clusters))
                self.epoch = 0
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_p = self.p[idx * self.batch_size:(idx + 1) * self.batch_size]
                # batch_p _  = self.model.predict(x, verbose=0)

                return batch_x, [batch_y, batch_p]

            def on_epoch_begin(self):
                if self.epoch % update_interval == 0:
                    _, q = model.predict(X_tu, batch_size=128)
                    self.p = target_distribution(q)
                self.epoch += 1


        if checkpoint:
            model.load_weights(checkpoint)

        check_cb = keras.callbacks.ModelCheckpoint('../checkpoint/' + file_name + '.hdf5',
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, mode='min')
        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=verbose, mode='auto')
        history = LossHistory()


        epoch_callback = Evaluation()
        optimizer = keras.optimizers.Adam(lr=0.001)
        # optimizer = keras.optimizers.Adagrad(lr=0.01)

        data_sequence_train = DataSequence(X_tu, y_tu, 128)
        data_sequence_val = DataSequence(X_v, y_v, 128)

        model.compile(loss=[_cost, m_kl], loss_weights=[1,500], optimizer=optimizer, metrics=[m_accuracy])

        model2 = Model(inputs=document, outputs=[output_dense1])

        # Initialize cluster centers using k-means.
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=123)
        kmeans.fit(model2.predict(X_tu,batch_size=128))

        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])




        model.fit_generator(data_sequence_train, validation_data=data_sequence_val, verbose=0,
                  epochs=200, shuffle=False, callbacks=[earlystop_cb, check_cb, epoch_callback])
        # model.fit(X_train, y_train, validation_split=0.1, batch_size=100,
        #           epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb])



        model.load_weights('../checkpoint/' + file_name + '.hdf5')
        # data_sequence_test = DataSequence(X_test, y_test, 128)


        # a = target_distribution(model.predict_generator(data_sequence_test)[1])
        # from scipy.sparse import csr_matrix
        # b = csr_matrix((np.ones(a.shape[0]), (y_test.argmax(-1), a.argmax(-1)))).toarray()
        # print(b)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(a[:,0], a[:,1], a[:,2],c=y_test.argmax(-1))
        # ax.set_title(file_name)
        # plt.show()
        predY = model.predict(X_test, batch_size=128)[0]
        # predY = model.predict_generator(data_sequence_test)[0]
        acc = accuracy_score(y_test.argmax(-1), predY.argmax(-1))
        print (acc)

        # exit()
        # if acc > best[0]:
        #     best = [acc]
        best_config = [[l_dim, n_clusters]]
        # elif acc == best[0]:
        #     best_config.append([l_dim, n_clusters])

        # print(best, best_config)
        # exit()

    return [acc], best_config

    # print(model.history.keys())
    # summarize history for accuracy
    # plt.figure(1)
    # plt.plot(tr_acc/n_split)
    # plt.plot(val_acc/n_split)
    # plt.title('{} model accuracy'.format(str(sys.argv[1]).split("_all_")[-1]))
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # # plt.show()
    # # summarize history for loss
    # plt.figure(2)
    # plt.plot(tr_loss/n_split)
    # plt.plot(val_loss/n_split)
    # plt.title('{} model loss'.format(str(sys.argv[1]).split("_all_")[-1]))
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # print (str(sys.argv[1]))

    ''' my_df = pd.DataFrame(history.accuracies)
    my_df.to_csv('accuracies.csv', index=False, header=False)

    my_df = pd.DataFrame(history.losses)
    my_df.to_csv('losses.csv', index=False, header=False)'''

    '''plt.plot(history.accuracies)
    plt.ylabel('Batch_per_epoch')
    plt.ylabel('Accuraccy')
    plt.show()
    plt.plot(history.losses)
    plt.ylabel('Batch_per_epoch')
    plt.ylabel('Loss')
    plt.show()'''

    # just showing access to the history object
    # print metrics.f1s
    # print history.losses
    # print history.accuracies
