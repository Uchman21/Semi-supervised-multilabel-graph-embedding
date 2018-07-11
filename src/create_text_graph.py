'''
LICENSE BSD 2-Clause
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import networkx as nx

from skmultilearn import dataset as dataset_load
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import load_npz
import sys, traceback


import math

import pandas as pd

import os

def writelabels(Y, selection,dataset_name):
	outfile = open("../ml_labels/{}.labels".format(dataset_name),"w")
	outfile.write("Total number of labels is #{}\n".format(Y.shape[1]))
	for sample in selection:
		outfile.write("{} {}\n".format(sample, ','.join(np.where(Y[sample,:] > 0)[0].astype(np.str).tolist())))

def iterative_sampling(labels,labeled_idx, fold, dataset_str):
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
							m = m[np.random.choice(m2, 1)[0]]
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

	return folds[0], Y, blacklist_samples


def preprocess_data(X, Y):
	filter_labels = np.where(Y.sum(0) > 100)[1]
	Y = Y[:,filter_labels]
	filter_samples = np.where(Y.sum(-1) > 0)[0]

	return X[filter_samples,:], Y[filter_samples,:]


tfidf = TfidfTransformer()
datasets = ['yahoo-' + data.split('.')[0] for data in os.listdir('../multi_label_dataset/yahoo') if
				'.xml' in data]
datasets = [data.split('.')[0] for data in os.listdir('../multi_label_dataset/tmc2007') if '.xml' in data]

for dataset_name in datasets:
	if dataset_name == "tmc2007":
		labelcount = open("../multi_label_dataset/tmc2007/tmc2007.xml").read().count("class")
		X, Y = dataset_load.load_from_arff("../multi_label_dataset/tmc2007/tmc2007.arff", labelcount=labelcount,
										   endian="little", load_sparse=True)
		
		X = tfidf.fit_transform(X)
		X, Y = preprocess_data(X, Y)
		labeled_idx = np.arange(Y.shape[0])
		selection, Y, blacklist_samples = iterative_sampling(Y.toarray(),labeled_idx, 5, dataset_name)
		writelabels(Y, selection, dataset_name)
		X = X.toarray()
		X[X <= 0.1] = 0
	
		S = np.zeros((X.shape[0], X.shape[0]))
		for k in range(X.shape[1]):
			connect_doc = np.where(X[:, k] > 0)[0]
			S[connect_doc, :] += X[:, k]
	
		np.fill_diagonal(S, 0.001)
	
		for k in range(S.shape[0]):
			ind = np.argpartition(S[k, :], -100)[:-100]
			S[k, ind] = 0

	elif "yahoo" in dataset_name:
		labelcount = open("../multi_label_dataset/yahoo/{}.xml".format(dataset_name.split("-")[1])).read().count(
			"Label")
		X, Y = dataset_load.load_from_arff(
			"../multi_label_dataset/yahoo/{}.arff".format(dataset_name.split("-")[1].title()),
			labelcount=labelcount, endian="little", load_sparse=True)
		
		X = tfidf.fit_transform(X)
		X, Y = preprocess_data(X, Y)
		labeled_idx = np.arange(Y.shape[0])
		selection, Y, blacklist_samples = iterative_sampling(Y.toarray(), labeled_idx, 5, dataset_name)
		writelabels(Y, selection, dataset_name)
		X = X.toarray()
		X[X <= 0.1] = 0

		S = np.zeros((X.shape[0], X.shape[0]))
		for k in range(X.shape[1]):
			connect_doc = np.where(X[:, k] > 0)[0]
			S[connect_doc, :] += X[:, k]
	
		np.fill_diagonal(S, 0.001)
	
		for k in range(S.shape[0]):
			ind = np.argpartition(S[k, :], -100)[:-100]
			S[k, ind] = 0
		# S = csr_matrix(S)

	nx_G = nx.from_numpy_matrix(S, create_using=nx.DiGraph())

	if dataset_name == "tmc2007":
		nx.write_weighted_edgelist(nx_G, "../graph/{}.edgelist".format(dataset_name))
	else:
		nx.write_weighted_edgelist(nx_G, "../graph/{}.edgelist".format(dataset_name.split("-")[1]))

