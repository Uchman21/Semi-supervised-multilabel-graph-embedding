'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import os
os.environ['PYTHONHASHSEED'] = '2018'
import argparse
import numpy as np
import networkx as nx
import node2vec, time
# from gensim.models import Word2Vec
from numpy.random import seed
# from tensorflow import set_random_seed
import random as rn
import itertools, tqdm
from load_data import load_cite_data, load_ml_data, load_txt_data
import matplotlib.pyplot as plt
import json
rn.seed(2018)
seed(2018)
import subprocess
# set_random_seed(2018)


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
						help='Embeddings path')

	parser.add_argument('--dataset', type=int, default=0,
						help='Dataset index [cora, citeseer, pubmed, dblp]')

	parser.add_argument('--l_type', nargs='?', default='cluster',
					help='cluster or vae')


	parser.add_argument('--su', dest='su', action='store_true',
					help='Choose between supervised (su) and semisupervised (ss) . Default is ss')
	parser.add_argument('--ss', dest='ss', action='store_false')
	parser.set_defaults(su=False)

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=20,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
					  help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')

	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph(input):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted or args.dataset > 2:
		G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)

	return



def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	if args.l_type == "cluster":
		import new_model_RNN4 as rm
		# import new_model_RNN_mla as rml
		import new_model_RNN6 as rml
	else:
		import new_model_RNN3 as rm
		import new_model_RNN_mlc as rml

	old_state = rn.getstate()
	st0=np.random.get_state()
	walk_length = [args.walk_length]
	num_walks = [args.num_walks]
	p_prob = [1]#[0.5, 1,2,4]
	q_prob = [1]#[0.5, 1,2,4]
	folds = 4
	configs = list(itertools.product(walk_length, num_walks, p_prob, q_prob))

	global_best = [-1]
	gb_config = []
	datasets = ['cora', 'citeseer', 'pubmed','dblp']
	datasets += ['yahoo-' + data.split('.')[0] for data in os.listdir('../multi_label_dataset/yahoo') if
					'.xml' in data]
	datasets += [data.split('.')[0] for data in os.listdir('../multi_label_dataset/tmc2007') if '.xml' in data]

	filename = "{}_{}_{}_{}_{}_{}".format(datasets[args.dataset], args.l_type, args.num_walks, args.walk_length, args.su, time.time())

	output_file = open("outputs/"+filename+".txt","w")
	# print(datasets)
	# exit()

	if "-" in datasets[args.dataset]:
		_dataset = datasets[args.dataset].split("-")[1]
	else:
		_dataset = datasets[args.dataset]

	print(_dataset)
	multi_learn = False

	for wlength, nwalks, p, q in configs:
		# restoreContext()
		# set_random_seed(2018)
		# fp = open("../emb/cora.walk")
		walk_file = "{}_{}_{}_{}_{}".format(_dataset, nwalks, wlength, str(p).replace(".","_"), str(q).replace(".","_"))
		walk_files_avail = set([data.split('.')[0] for data in os.listdir('../emb/') if
		 '.walk' in data])
		if walk_file not in walk_files_avail:
			if multi_learn == True and args.l_type == "cluster" and args.su == True:
				command = "./node2vec -i:../graph/{}.edgelist -o:../emb/{}.walk -l:{} -r:{} -p:{} -q:{}".format(_dataset,walk_file,wlength, nwalks,p,q)
				if args.dataset > 2:
					command += " -w"
				subprocess.check_call(command)
			elif multi_learn == True:
				while True:
					time.sleep(30)
					walk_files_avail = set([data.split('.')[0] for data in os.listdir('../emb/') if
											'.walk' in data])
					if walk_file in walk_files_avail:
						break
			else:
				command = "./node2vec -i:../graph/{}.edgelist -o:../emb/{}.walk -l:{} -r:{} -p:{} -q:{}".format(
					_dataset,walk_file, wlength, nwalks, p, q)
				if args.dataset > 2:
					command += " -w"
				subprocess.check_call(command, shell=True)

		fp = open("../emb/{}.walk".format(walk_file))
		walks = json.load(fp)



		rn.setstate(old_state)
		np.random.set_state(st0)
		# walks = json.load(fp)
		# G = node2vec.Graph(nx_G, args.directed, p, q)
		# print(total_size(nx_G))
		# G.preprocess_transition_probs(10)
		# walks = G.simulate_walks(nwalks, wlength, 1)
		# del G
		# output_file.write(walks[0])
		# exit()

		rng = np.random.RandomState(seed=2018)
		docs = []
		labels = []


		if args.dataset < 3:
			features, Y, train_mask, val_mask, test_mask, u_mask = load_cite_data(_dataset)

			for I in range(len(walks)):
				docs.append(walks[str(I)])
				labels.append(Y[I])




			X = np.array(docs)  # np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
			y = np.array(labels)

			rng.shuffle(train_mask)
			rng.shuffle(val_mask)
			rng.shuffle(u_mask)
			rng.shuffle(test_mask)

			del walks
			del docs

			l_dim_arr = [64,128,256]
			n_labels = y.shape[1]

			if args.su:
				n_clusters_arr = [0]
			else:
				n_clusters_arr = [n_labels, 3, 10] if n_labels > 3 else [n_labels, 6, 10]
			configs_2 = list(itertools.product(l_dim_arr, n_clusters_arr))
			for l_dim, n_clusters in tqdm.tqdm(configs_2):
				if args.su:
					best, best_config = rm.G2V( filename,nwalks,wlength, _dataset,train_mask,val_mask,u_mask,test_mask,l_dim,n_clusters, X,y, features, True)
				else:
					best, best_config = rm.G2V(filename, nwalks, wlength,_dataset, train_mask, val_mask, u_mask,
											   test_mask, l_dim, n_clusters, X, y, features, False)
					# best, best_config = [0], [[1,2,3]]
				# time.sleep(60)
				if best[0] > global_best[0]:
					global_best = best
					gb_config = best_config
					gb_config.append([p, q, nwalks, wlength])
					output_file.write ("---------------------best so far --------------------\n")
					output_file.write("best accuracy: {}\n".format(global_best))
				elif best[0] == global_best[0]:
					gb_config.append(best_config)
					gb_config.append([p, q, nwalks, wlength])
		elif args.dataset == 3:

			l_dim_arr = [64,128,256]
			if args.su:
				n_clusters_arr = [10]
			else:
				n_clusters_arr = [1, 4, 8]
			configs_2 = list(itertools.product(l_dim_arr, n_clusters_arr))

			for l_dim, n_clusters in tqdm.tqdm(configs_2):
				X, cv_splits, features, y, blacklist_samples = load_ml_data(folds,walks,_dataset)
				n_clusters_arr.append(y.shape[1])
				if args.su:
					best, best_config = rml.G2V( filename, nwalks, wlength, _dataset, X, cv_splits, features, y,l_dim,n_clusters, True)
				else:
					best, best_config = rml.G2V(filename, nwalks, wlength, _dataset, X, cv_splits, features, y, l_dim,
												n_clusters, False)
				if best[0] > global_best[0]:
					global_best = best
					gb_config = best_config
					gb_config.append([p, q, nwalks, wlength])
					output_file.write ("---------------------best so far --------------------\n")
					output_file.write("best accuracy: {}\n".format(global_best))
				elif best[0] == global_best[0]:
					gb_config.append(best_config)
					gb_config.append([p, q, nwalks, wlength])

		else:
			l_dim_arr = [64, 128, 256]
			if args.su:
				n_clusters_arr = [10]
			else:
				n_clusters_arr = [1, 5, 10]
			configs_2 = list(itertools.product(l_dim_arr, n_clusters_arr))

			X, y, cv_splits, features = load_txt_data(datasets[args.dataset], walks, folds, rng)
			for l_dim, n_clusters in tqdm.tqdm(configs_2):
				n_clusters_arr.append(y.shape[1])
				if args.su:
					best, best_config = rml.G2V(filename, nwalks, wlength,_dataset, X, cv_splits,
												features, y, l_dim, n_clusters, True)
				else:
					best, best_config = rml.G2V(filename, nwalks, wlength, _dataset, X, cv_splits,
												features, y, l_dim,
												n_clusters, False)
				if best[0] > global_best[0]:
					global_best = best
					gb_config = best_config
					gb_config.append([p, q, nwalks, wlength])
					output_file.write("---------------------best so far --------------------\n")
					output_file.write("best accuracy: {}\n".format(global_best))
				elif best[0] == global_best[0]:
					gb_config.append(best_config)
					gb_config.append([p, q, nwalks, wlength])



		output_file.write ("---------------------best general --------------------\n")
		output_file.write("best accuracy: {}\n".format(global_best))
		output_file.write("best configurations: h_dim:{}, n_clusters:{},  p: {}, q: {}, n_walks: {}, walk_length: {}".format(
			gb_config[0][0], gb_config[0][1],
			gb_config[-1][0], gb_config[-1][1], gb_config[-1][2], gb_config[-1][3]))
		output_file.write("{}".format(gb_config))


	# learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
