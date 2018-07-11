import pandas as pd

import os

os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda, Embedding, Concatenate, Masking, Add, Multiply, Subtract
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import Sequence, to_categorical
from keras.utils import plot_model
# from keras.losses import categorical_crossentropy
from keras import initializers
import random as rn
import numpy as np

import tensorflow as tf
# tf.set_random_seed(2018)
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
# rn.seed(2018)
# seed(2018)
# set_random_seed(2018)

verbose = 0


# def generate_layer_sizes(w_min, w_max):
#     '''generate random neural network layer size'''
#
#     return int(np.random.uniform(w_min, w_max))


def mcat_crossentropy(y_true, y_pred):
    loss =  K.categorical_crossentropy(y_true, y_pred)

    condition = K.greater(K.sum(y_true), 0)
    return K.switch(condition, loss, K.zeros_like(loss))


def m_kl(y_true, y_pred):

    true_y = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = K.sum(true_y * K.log(true_y / y_pred), axis=-1)
    return  loss
    # condition = K.greater(K.sum(y_true), 0)
    # return K.switch(condition, loss, K.zeros_like(loss))


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


# def load_preprocessed(dataset):
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_cite_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    u_mask = ~(train_mask | test_mask | val_mask)

    features = load_npz('../features/{}.npz'.format(dataset_str)).toarray()

    return (features, labels, np.where(train_mask == True)[0] ,np.where(val_mask == True)[0], np.where(test_mask == True)[0], np.where(u_mask == True)[0])


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))



# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def G2V(file_name, num_walks, walk_length, _dataset,train_mask,val_mask,u_mask,test_mask,l_dim,n_clusters,X, y,features, is_supervised=False):


    maxlen = walk_length
    max_sentences = num_walks

    X_train, y_train = X[train_mask, :], y[train_mask, :]
    X_test, y_test = X[test_mask, :], y[test_mask, :]
    X_ul, y_ul = X[u_mask, :], y[u_mask, :]
    X_v, y_v = X[val_mask, :], y[val_mask, :]
    # X_tu, y_tu = np.vstack((X_train, X_ul)), np.vstack((y_train, -1*np.ones_like(y_ul)))

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

    # old_state = rn.getstate()
    # st0=np.random.get_state()
    # K.clear_session()
    rn.seed(2018)
    np.random.seed(2018)
    tf.reset_default_graph()
    # rn.seed(2018)
    # seed(2018)
    tf.set_random_seed(2018)
    # rn.setstate(old_state)
    # np.random.set_state(st0)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    with sess.as_default():
        document = Input(shape=(max_sentences, maxlen), dtype='int64')

        # sentence input
        in_sentence = Input(shape=(maxlen,), dtype='int64')
        # char indices to one hot matrix, 1D sequence to 2D
        # embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
        embedded = Embedding(len(X),
                             features.shape[1],
                             weights=[features],
                             # input_length=MAX_SEQUENCE_LENGTH,
                             trainable=False)(in_sentence)

        bi_lstm_sent = \
            Bidirectional(LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
                embedded)

        sent_encode = Dropout(0.25)(bi_lstm_sent)
        # sentence encoder
        encoder = Model(inputs=in_sentence, outputs=sent_encode)
        encoded = TimeDistributed(encoder)(document)
        # encoded: sentences to bi-lstm for document encoding
        b_lstm_doc = \
            Bidirectional(LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
                encoded)

        output_lstm = Dropout(0.25)(b_lstm_doc)
        output_dense = Dense(l_dim, activation='relu')(output_lstm)
        # output_dense = Dropout(0.1)(output_dense)

        if is_supervised == False:
            clustering_layer = ClusteringLayer(n_clusters, name='ss')(output_dense)
            output = Concatenate(axis=-1)([output_dense, clustering_layer])
            output = Dense(y.shape[1], activation='softmax', name='su')(output)
            output =  Concatenate(axis=-1)([output, clustering_layer])
            model = Model(inputs=document, outputs=output)
        else:
            output = output_dense
            output = Dense(y.shape[1], activation='softmax', name='su')(output)
            model = Model(inputs=document, outputs=output)

        class DataSequence(Sequence):

            def __init__(self, x_l, x_ul, y_l, batch_size, _type="train"):
                self._type = _type
                if self._type == "train":
                    if is_supervised == False:
                        self.xl, self.xu, self.y, self.pl, self.pu = x_l, x_ul, y_l, target_distribution(
                            model.predict(x_l, batch_size=128)[:,-n_clusters:]), target_distribution(
                            model.predict(x_ul, batch_size=128)[:,-n_clusters:])
                    else:
                        self.xl, self.xu, self.y = x_l, x_ul, y_l
                else:
                    if is_supervised == False:
                        self.xl, self.y, self.p = x_l, y_l, target_distribution(
                            model.predict(x_l, batch_size=128)[:,-n_clusters:])
                    else:
                        self.xl, self.y = x_l, y_l
                self.epoch = 0
                self.labeled_size = self.xl.shape[0]
                self.batch_size = batch_size
                self.steps_l = int(np.ceil(len(self.xl) / float(self.batch_size)))
                self.update_interval = 1

            def __len__(self):
                if self._type == "train":
                    return int(
                        np.ceil(len(self.xl) / float(self.batch_size)) + np.ceil(len(self.xu) / float(self.batch_size)))
                else:
                    return int(
                        np.ceil(len(self.xl) / float(self.batch_size)))

            def __getitem__(self, idx):
                if self._type == "train":
                    if idx < self.steps_l:
                        batch_x = self.xl[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                        if is_supervised == False:
                            batch_y = np.hstack((batch_y,self.pl[idx * self.batch_size:(idx + 1) * self.batch_size]))
                    else:
                        batch_x = self.xu[(idx - self.steps_l) * self.batch_size:((idx - self.steps_l) + 1) * self.batch_size]
                        batch_y = -1 * np.ones((batch_x.shape[0], self.y.shape[1]))
                        if is_supervised == False:
                            batch_y = np.hstack((batch_y,self.pu[
                                  (idx - self.steps_l) * self.batch_size:((idx - self.steps_l) + 1) * self.batch_size]))
                else:
                    batch_x = self.xl[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    if is_supervised == False:
                        batch_y = np.hstack((batch_y,self.p[idx * self.batch_size:(idx + 1) * self.batch_size]))

                return batch_x, batch_y

            def on_epoch_begin(self):
                if self.epoch % self.update_interval == 0:
                    ql = model.predict(self.xl, batch_size=128)[:,-n_clusters:]
                    self.pl = target_distribution(ql)
                    qu = model.predict(self.xu, batch_size=128)[:,-n_clusters:]
                    self.pu = target_distribution(qu)
                self.epoch += 1

        data_sequence_train = DataSequence(X_train, X_ul, y_train, 128, "train")
        data_sequence_val = DataSequence(X_v, None,  y_v, 128, "val")
        optimizer = keras.optimizers.Adam(lr=0.0005)

        def m_loss(y_true, y_pred):
            alpha = 0.99
            if is_supervised:
                sup_loss = mcat_crossentropy(y_true, y_pred)
                return sup_loss
            else:
                sup_loss = mcat_crossentropy(y_true[:, :-n_clusters], y_pred[:, :-n_clusters])
                cl_loss = m_kl(y_true[:,-n_clusters:], clustering_layer)
                return (alpha*sup_loss) + (1-alpha)*cl_loss

        def m_accuracy(y_true, y_pred):
            treshold = 0
            if is_supervised == False:
                y_true, y_pred = y_true[:, :-n_clusters], y_pred[:, :-n_clusters]
            mask = Lambda(lambda x: K.greater_equal(x, treshold))(y_true)
            mask = Lambda(lambda x: K.cast(x, 'float32'))(mask)
            pred_label = Lambda(lambda x: x * mask)(y_pred)
            true_label = Lambda(lambda x: x * mask)(y_true)
            return K.mean(K.equal(K.argmax(true_label, axis=-1), K.argmax(pred_label, axis=-1)))

        model.compile(loss=m_loss, optimizer=optimizer, metrics=[m_accuracy])

        if is_supervised == False:
            model2 = Model(inputs=document, outputs=[output_dense])
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=2018)
            kmeans.fit(model2.predict(np.vstack((X_train, X_v)), batch_size=128))

            model.get_layer(name='ss').set_weights([kmeans.cluster_centers_])

        # plot_model(model, to_file='G2Vmodel.png')
        # exit()
        # model.summary()
        # plot_model(model, to_file='Images/model.png')
        # exit()

        # total_train_size = data_sequence_train.__len__() * 128
        # ltrain_size = data_sequence_train.steps_l * 128

        # if checkpoint:
        #     model.load_weights(checkpoint)
        #
        # file_name = _dataset + 'g2v'
        check_cb = keras.callbacks.ModelCheckpoint('../checkpoint/' + file_name + '.hdf5',
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, mode='min')
        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=verbose, mode='auto')
        history = LossHistory()
        # optimizer = keras.optimizers.Adam(lr=0.001)
        # optimizer = keras.optimizers.Adagrad(lr=0.001)

        model.fit_generator(data_sequence_train, validation_data=data_sequence_val, verbose=verbose,
                  epochs=200, shuffle=False, callbacks=[earlystop_cb, check_cb])
        # model.fit(X_train, y_train, validation_split=0.1, batch_size=100,
        #           epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb])

        model.load_weights('../checkpoint/' + file_name + '.hdf5')
        # clust = Model(inputs=[document], outputs=z_mu)
        to_print = False

        if is_supervised == False and to_print == True:
            pred = model.predict(X_test,batch_size=128)
            a = target_distribution(pred[:,-n_clusters:])
            from scipy.sparse import csr_matrix
            b = csr_matrix((np.ones(a.shape[0]), (y_test.argmax(-1), a.argmax(-1)))).toarray()
            print(b)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(a[:,0], a[:,1], a[:,2],c=y_test.argmax(-1))
            ax.set_title(file_name)

            # plt.figure(1)
            # plt.scatter(a[:, 0], a[:, 1], c=y_test.argmax(-1))
            plt.show()

            predY = pred[:,:-n_clusters]
        elif is_supervised == False:
            predY =  model.predict(X_test,batch_size=128)[:,:-n_clusters]
        else:
            predY = model.predict(X_test, batch_size=128)

        acc = accuracy_score(y_test.argmax(-1), predY.argmax(-1))
        # print(acc)
        # exit()


        if acc > best[0]:
            best = [acc]
            best_config = [[l_dim,n_clusters]]
        elif acc == best[0]:
            best_config.append([l_dim,n_clusters] )

    return best, best_config

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
