import pandas as pd

import os

os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda, Embedding, Concatenate
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import Sequence
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer, InputSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import pickle as pkl
from numpy.random import seed
from tensorflow import set_random_seed
import random as rn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from numpy import around
from keras.utils import plot_model
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
import itertools,tqdm


verbose = 0


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


class Evaluation(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if verbose == 1:
            print(
            'epoch [{0:3d}]:  loss: {1:.4f} - sup_loss: {2:.4f} - unsup_loss: {3:.4f} - val_loss: {4:.4f} - val_sup_loss: {5:.4f} - val_unsup_loss: {6:.4f} - sup_accuracy: {7:.4f} val_sup_accuracy: {8:.4f}'.format(
                epoch, logs['loss'], logs['classification_loss'],
                logs['clustering_loss'], logs['val_loss'], logs['val_classification_loss'],
                logs['val_clustering_loss'], logs['classification_m_accuracy'],
                logs['val_classification_m_accuracy']))
        elif verbose == 2:
            print(
            'epoch [{0:3d}]:  loss: {1:.4f} - val_loss: {2:.4f} - accuracy: {3:.4f} val_accuracy: {4:.4f}'.format(epoch,
                                                                                                                  logs[
                                                                                                                      'loss'],
                                                                                                                  logs[
                                                                                                                      'val_loss'],
                                                                                                                  logs[
                                                                                                                      'classification_m_accuracy'],
                                                                                                                  logs[
                                                                                                                      'val_classification_m_accuracy']))


class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)


def _cost(true_y, pred_y):
    # mask = K.sum(true_y, -1) > 0
    # true_label = K.boolean_mask(true_y, mask)
    # pred_label = K.boolean_mask(pred_y, mask)

    treshold = 0
    mask = Lambda(lambda x: K.greater_equal(x, treshold))(true_y)
    mask = Lambda(lambda x: K.cast(x, 'float32'))(mask)
    pred_label = Lambda(lambda x: x * mask)(pred_y)
    true_label = Lambda(lambda x: x * mask)(true_y)
    # return K.mean(K.binary_crossentropy(pred_label, true_label))
    return WeightedBinaryCrossEntropy(0.5)(true_label, pred_label)


def m_kl(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values.
    '''
    K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1))


def m_accuracy(true_y, pred_y):
    treshold = 0
    mask = Lambda(lambda x: K.greater_equal(x, treshold))(true_y)
    mask = Lambda(lambda x: K.cast(x, 'float32'))(mask)
    pred_label = Lambda(lambda x: x * mask)(pred_y)
    true_label = Lambda(lambda x: x * mask)(true_y)
    return K.mean(K.all(K.equal(true_label, K.round(pred_label)), axis=-1))


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
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
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


def G2V(data, checkpoint, num_walks, walk_length, _dataset, X, cv_splits, features, y, blacklist_samples,l_dim,n_clusters ):
    checkpoint = None

    loss = 0
    index = 0
    maxiter = 8000
    update_interval = 1
    # index_array = np.arange(x.shape[0])
    tol = 0.001  # tolerance threshold to stop training
    fold = 4
    rng = np.random.RandomState(seed=2018)

    # txt = ''
    # docs = []
    # sentences = []
    # labels = []
    # # features, Y, train_mask, val_mask, test_mask, u_mask, tu_mask = load_cite_data(_dataset)
    #
    # for key in data:
    #     try:
    #         docs.append(data[key])
    #         labels.append(key)
    #     except:
    #         pass
    #
    # cv_splits, features, y, blacklist_samples = iterative_sampling(labels, fold, rng, _dataset)

    maxlen = walk_length
    max_sentences = num_walks

    # X = np.array(docs)
    # y = np.array(labels)
    # l_dim_arr = [64]#[64, 128, 256]
    # document input
    search_iter = 1
    best = [-1]
    best_config = []
    n_labels = y.shape[1]
    # n_clusters_arr = [10]#[n_labels, 10]
    # configs = list(itertools.product(l_dim_arr, n_clusters_arr))
    old_state = rn.getstate()
    st0 = np.random.get_state()

    # for l_dim, n_clusters in tqdm.tqdm(configs):

    acc, f1_micro, precision_micro, recall_micro = 0, 0, 0, 0

    f1_macro, precision_macro, recall_macro = 0, 0, 0
    for i in range(fold):
        u_mask = []
        test_mask = []
        for j in range(len(cv_splits)):
            if j == i:
                train_mask = cv_splits[j]
            elif j == ((i + 1) % fold):
                val_mask = cv_splits[j]
            elif j == ((i + 2) % fold):
                test_mask = cv_splits[j]
            else:
                u_mask += cv_splits[j]


        rng.shuffle(train_mask)
        rng.shuffle(val_mask)
        rng.shuffle(u_mask)
        rng.shuffle(test_mask)

        # print(len(train_mask), len(val_mask),len(test_mask), len(u_mask))
        # exit()
        tu_mask = train_mask + u_mask
        X_train, y_train = X[train_mask, :], y[train_mask, :]
        X_test, y_test = X[test_mask, :], y[test_mask, :]
        X_ul, y_ul = X[u_mask, :], y[u_mask, :]
        X_v, y_v = X[val_mask, :], y[val_mask, :]
        X_tu, y_tu = X[tu_mask, :], np.vstack((y[train_mask, :], -1 * np.ones_like(y_ul)))

        filter_length = [3, 2, 2]
        nb_filter = [64, 6, 64]
        pool_length = 2

        r_alpha = 0.9

        K.clear_session()
        tf.reset_default_graph()
        # rn.seed(2018)
        # seed(2018)
        set_random_seed(2018)
        rn.setstate(old_state)
        np.random.set_state(st0)
        document = Input(shape=(max_sentences, maxlen), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(maxlen,), dtype='int64')
        embedded = Embedding(X.shape[0],
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
            Bidirectional(
                LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
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
            Bidirectional(
                LSTM(l_dim, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(
                encoded)

        output_lstm = Dropout(0.25)(b_lstm_doc)

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(output_lstm)

        output = Concatenate(axis=-1)([output_lstm, clustering_layer])

        output = Dense(l_dim, activation='relu')(output)
        output = Dropout(0.1)(output)

        output = Dense(y.shape[1], activation='sigmoid', name='classification')(output)

        model = Model(inputs=document, outputs=[output, clustering_layer])

        class DataSequence(Sequence):

            def __init__(self, x_set, y_set, p_set, batch_size):
                self.x, self.y, self.p = x_set, y_set, p_set
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
                    _, self.p = model.predict(X_tu)
                self.epoch += 1

        if checkpoint:
            model.load_weights(checkpoint)

        file_name = _dataset + 'g2v1'
        check_cb = keras.callbacks.ModelCheckpoint('../checkpoint/' + file_name + '.hdf5',
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, mode='min')
        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=verbose, mode='auto')
        history = LossHistory()

        epoch_callback = Evaluation()
        optimizer = keras.optimizers.Adam(lr=0.001)

        data_sequence_train = DataSequence(X_tu, y_tu, model.predict(X_tu,batch_size=128)[1], 128)
        data_sequence_val = DataSequence(X_v, y_v, model.predict(X_v,batch_size=128)[1], 1000)


        model.compile(loss=[_cost, m_kl], loss_weights=[1, 100], optimizer=optimizer, metrics=[m_accuracy])

        model2 = Model(inputs=document, outputs=[output_lstm])
        # model2.compile()

        # Initialize cluster centers using k-means.
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=2018)
        kmeans.fit(model2.predict(X_tu, batch_size=128))

        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        model.fit_generator(data_sequence_train, validation_data=data_sequence_val, verbose=0,
                            epochs=200, shuffle=False, callbacks=[earlystop_cb, check_cb, epoch_callback])
        # model.fit(X_train, y_train, validation_split=0.1, batch_size=100,
        #           epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb])

        model.load_weights('../checkpoint/' + file_name + '.hdf5')

        # a = model.predict(X_test)[1]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(a[:,0], a[:,1], a[:,2],c=y_test.argmax(-1))
        # plt.show()
        # exit()
        # predY = model.predict(X_test)[0]
        data_sequence_test = DataSequence(X_test, y_test, model.predict(X_test,batch_size=128)[1], 1000)
        predY = model.predict_generator(data_sequence_test)[0]

        # print(predY)
        # print(y_test)
        # print(accuracy_score(y_test, np.round(predY)))
        acc += accuracy_score(y_test, np.round(predY))
        print(acc)

        f1_micro += f1_score(y_test, np.round(predY), average='micro')
        precision_micro += precision_score(y_test, np.round(predY), average='micro')
        recall_micro += recall_score(y_test, np.round(predY), average='micro')

        f1_macro += f1_score(y_test, np.round(predY), average='macro')
        precision_macro += precision_score(y_test, np.round(predY), average='macro')
        recall_macro += recall_score(y_test, np.round(predY), average='macro')

    acc /= fold

    f1_micro /= fold
    precision_micro /= fold
    recall_micro /= fold

    f1_macro /= fold
    precision_macro /= fold
    recall_macro /= fold

    if f1_macro > best[0]:
        best = [f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, acc]
        best_config = [[l_dim, n_clusters]]
    elif f1_macro == best[0]:
        best_config.append([l_dim, n_clusters])

        # print(best, best_config)
        # exit()
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
