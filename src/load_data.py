'''
LICENSE BSD 2-Clause
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
from scipy.sparse import load_npz,hstack,lil_matrix, vstack, diags, coo_matrix,csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pkl
import networkx as nx
import sys, traceback

from skmultilearn import dataset as dataset_load

import math

import pandas as pd

import os

os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from scipy.sparse import load_npz



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
	ul_mask = (train_mask | u_mask)

	features = load_npz('../features/{}_2.npz'.format(dataset_str)).toarray()

	return (features, labels, np.where(train_mask == True)[0] ,np.where(val_mask == True)[0], np.where(test_mask == True)[0], np.where(u_mask == True)[0])


def iterative_sampling(labels,labeled_idx, fold, rng, dataset_str):
	ratio_per_fold = 1 / fold
	# indecies = np.arange(np.shape(Y)[0])
	if dataset_str == "dblp":
		Y = load_npz('../ml_labels/{}.npz'.format(dataset_str))[labels].toarray()
		index = np.arange(np.shape(Y)[0])
		unlabeled = np.where(Y.sum(-1) < 0)[0]
		labeled_idx = np.setdiff1d(index, unlabeled)
	else:
		Y = labels
	# print(Y.shape)
	# exit()


	folds = [[] for i in range(fold)]
	number_of_examples_per_fold = np.array([(1 / fold) * np.shape(Y[labeled_idx, :])[0] for i in range(fold)])

	blacklist_samples = np.array([])
	number_of_examples_per_label = np.sum(Y[labeled_idx, :], 0)
	blacklist_labels = np.where(number_of_examples_per_label < fold)[0]
	print(blacklist_labels)
	desired_examples_per_label = number_of_examples_per_label * ratio_per_fold

	subset_label_desire = np.array([desired_examples_per_label for i in range(fold)])
	total_index = np.sum(labeled_idx)
	max_label_occurance = np.max(number_of_examples_per_label) + 1
	sel_labels = np.setdiff1d(range(Y.shape[1]), blacklist_labels)

	while total_index > 0:
		try:
			min_label_index = np.where(number_of_examples_per_label == np.min(number_of_examples_per_label))[0]
			for index in labeled_idx:
				if (Y[index, min_label_index[0]] == 1 and index != -1) and (min_label_index[0] not in blacklist_labels):
					m = np.where(
						subset_label_desire[:, min_label_index[0]] == subset_label_desire[:, min_label_index[0]].max())[
						0]
					if len(m) == 1:
						folds[m[0]].append(index - (blacklist_samples < index).sum())
						subset_label_desire[m[0], Y[index, :]] -= 1
						labeled_idx[np.where(labeled_idx == index)] = -1
						number_of_examples_per_fold[m[0]] -= 1
						total_index = total_index - index
					else:
						m2 = np.where(number_of_examples_per_fold[m] == np.max(number_of_examples_per_fold[m]))[0]
						if len(m2) > 1:
							m = m[rng.choice(m2, 1)[0]]
							folds[m].append(index - (blacklist_samples < index).sum())
							subset_label_desire[m, Y[index, :]] -= 1
							labeled_idx[np.where(labeled_idx == index)] = -1
							number_of_examples_per_fold[m] -= 1
							total_index = total_index - index
						else:
							m = m[m2[0]]
							folds[m].append(index - (blacklist_samples < index).sum())
							subset_label_desire[m, Y[index, :]] -= 1
							labeled_idx[np.where(labeled_idx == index)] = -1
							number_of_examples_per_fold[m] -= 1
							total_index = total_index - index
				elif (Y[index, min_label_index[0]] == 1 and index != -1):
					if (min_label_index[0] in blacklist_labels) and np.any(Y[index, sel_labels]) == False:
						np.append(blacklist_samples, index)

						# subset_label_desire[m,Y[index,:]] -= 1
						labeled_idx[np.where(labeled_idx == index)] = -1
						# number_of_examples_per_fold[m] -= 1
						total_index = total_index - index

			number_of_examples_per_label[min_label_index[0]] = max_label_occurance
		except:
			traceback.print_exc(file=sys.stdout)
			exit()

	Y = Y[:, sel_labels]

	if dataset_str == "dblp":
		features = load_npz('../features/{}.npz'.format(dataset_str))[labels].toarray()
	else:
		features = None
	return folds, features, Y, blacklist_samples

def load_ml_data(fold, data, _dataset):

	rng = np.random.RandomState(seed=2018)

	txt = ''
	docs = []
	sentences = []
	labels = []
	# features, Y, train_mask, val_mask, test_mask, u_mask, tu_mask = load_cite_data(_dataset)

	for key in data:
		try:
			docs.append(data[key])
			labels.append(int(key))
		except:
			pass

	cv_splits, features, y, blacklist_samples = iterative_sampling(labels, fold, rng, _dataset)

	return np.array(docs), cv_splits, features, y, blacklist_samples

def preprocess_data(X, Y):
	filter_labels = np.where(Y.sum(0) > 100)[1]
	Y = Y[:,filter_labels]
	filter_samples = np.where(Y.sum(-1) > 0)[0]

	return X[filter_samples,:], Y[filter_samples,:]

def load_txt_data(dataset_name, walks, folds, rng):
	svd = TruncatedSVD(n_components=300, n_iter=10, random_state=0)
	std_scaler = Normalizer()
	tfidf = TfidfTransformer()
	if dataset_name == "tmc2007":
		labelcount = open("../multi_label_dataset/tmc2007/tmc2007.xml").read().count("class")
		features, Y = dataset_load.load_from_arff("../multi_label_dataset/tmc2007/tmc2007.arff", labelcount=labelcount,
									  endian="little", load_sparse=True)
		features, Y = preprocess_data(features, Y)
		features = tfidf.fit_transform(features)
		features = svd.fit_transform(features)
		features = std_scaler.fit_transform(features)

	elif "yahoo" in dataset_name:
		labelcount = open("../multi_label_dataset/yahoo/{}.xml".format(dataset_name.split("-")[1])).read().count(
			"Label")
		features, Y = dataset_load.load_from_arff("../multi_label_dataset/yahoo/{}.arff".format(dataset_name.split("-")[1].title()),
									  labelcount=labelcount, endian="little", load_sparse=True)
		features, Y = preprocess_data(features, Y)
		features = tfidf.fit_transform(features)
		features = svd.fit_transform(features)
		features = std_scaler.fit_transform(features)
	else:
		print ("unknown dataset")
		exit()

	X = []
	y=[]
	Y = Y.toarray()
	for I in range(len(walks)):
		X.append(walks[str(I)])
		y.append(Y[I])

	X = np.array(X)  # np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
	y = np.array(y)

	# X = scaler.fit_transform(pca.fit_transform(X.toarray()))
	# X = select_norm_count(X, mindf=0.03, maxdf=0.8, normalise = True)
	# elif dataset_name == "delve":
	#	 X, Y = load_delve(rng, comb_type=comb_type, add_validation_set=add_validation_set, add_Text=add_Text,
	#					   is_cv=is_cv, d_size=d_size)


	index = np.arange(np.shape(X)[0])
	unlabeled = np.where(Y.sum(-1) < 0)[0]
	labeled_idx = np.setdiff1d(index, unlabeled)
	print("Features shape = {}".format(X.shape))
	print("Label shape = {}".format(Y.shape))
	dataset = dataset_name.split("-")[1]
	folds, _, y, blacklist_samples = iterative_sampling(y, labeled_idx, folds, rng, dataset)
	sel_samples = np.setdiff1d(index, blacklist_samples)
	print(sel_samples.shape)
	return  ( X, y, folds, features)