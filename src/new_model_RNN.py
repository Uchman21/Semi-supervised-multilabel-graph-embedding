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

verbose = 1


# def generate_layer_sizes(w_min, w_max):
#     '''generate random neural network layer size'''
#
#     return int(np.random.uniform(w_min, w_max))


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


def G2V(data, checkpoint, num_walks, walk_length, _dataset,train_mask,val_mask,u_mask,test_mask,l_dim,n_clusters,X, y,features):


    maxlen = walk_length
    max_sentences = num_walks

    X_train, y_train = X[train_mask, :], y[train_mask, :]
    X_test, y_test = X[test_mask, :], y[test_mask, :]
    # X_ul, y_ul = X[u_mask, :], y[u_mask, :]
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
        # char indices to one hot matrix, 1D sequence to 2D
        # embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
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

        output = Dropout(0.25)(b_lstm_doc)
        output_l1 = Dense(l_dim, activation='relu', name="dense1")(output)
        output = Dropout(0.1)(output_l1)
        output = Dense(y.shape[1], activation='softmax')(output)

        model = Model(inputs=document, outputs=output)
        # layer_name = 'dense1'
        # Dlayer_model = Model(inputs=model.input,
        #                                  outputs=model.get_layer(layer_name).output)


        # model.summary()
        # plot_model(model, to_file='Images/model.png')
        # exit()

        if checkpoint:
            model.load_weights(checkpoint)

        file_name = _dataset + 'g2v'
        check_cb = keras.callbacks.ModelCheckpoint('../checkpoint/' + file_name + '.hdf5',
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, mode='min')
        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=verbose, mode='auto')
        history = LossHistory()
        optimizer = keras.optimizers.Adam(lr=0.001)
        # optimizer = keras.optimizers.Adagrad(lr=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(X_train, y_train, validation_data=(X_v, y_v), batch_size=128,verbose=verbose,
                  epochs=200, shuffle=False, callbacks=[earlystop_cb, check_cb])
        # model.fit(X_train, y_train, validation_split=0.1, batch_size=100,
        #           epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb])

        model.load_weights('../checkpoint/' + file_name + '.hdf5')

        predY = model.predict(X_test,batch_size=128)

        acc = accuracy_score(y_test.argmax(-1), predY.argmax(-1))
        print(acc)
        # exit()


        if acc > best[0]:
            best = [acc]
            best_config = [[l_dim,0]]
        elif acc == best[0]:
            best_config.append([l_dim,0] )

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
