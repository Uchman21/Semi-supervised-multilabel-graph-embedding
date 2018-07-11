from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
os.environ['PYTHONHASHSEED'] = '123'

import numpy as np
import networkx as nx
import random, time
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager
from functools import partial
from tqdm import tqdm
from scipy.sparse import csr_matrix




class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def pre_n2vwalk(self, nodes, walk_length, walk_iter):
		random.seed(walk_iter)
		random.shuffle(nodes)
		walks = []
		for node in nodes:
			# walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
			# if node in walks:
			# 	walks[node].append(self.node2vec_walk(walk_length=walk_length, start_node=node))
			# else:
			walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def pre_process_transition(self, stage, isdirected, node_edge):
		if stage == 1:
			alias_nodes = {}
			unnormalized_probs = [self.G[node_edge][nbr]['weight'] for nbr in sorted(self.G.neighbors(node_edge))]
			norm_const = sum(unnormalized_probs)
			normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
			alias_nodes[node_edge] = alias_setup(normalized_probs)
			return alias_nodes
		elif stage == 2:
			alias_edges = {}
			if isdirected:
				alias_edges[node_edge] = self.get_alias_edge(node_edge[0], node_edge[1])
			else:
				alias_edges[node_edge] = self.get_alias_edge(node_edge[0], node_edge[1])
				alias_edges[(node_edge[1], node_edge[0])] = self.get_alias_edge(node_edge[1], node_edge[0])
			return alias_edges


	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		# G = self.G
		# alias_nodes = self.alias_nodes
		# alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(self.G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(self.alias_edges[(prev, cur)][0],
											   self.alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk


	def simulate_walks(self, num_walks, walk_length, workers=1):
		'''
		Repeatedly simulate random walks from each node.
		'''
		# G = self.G
		walks = {}
		nodes = list(self.G.nodes())
		print ('Walk iteration:')
		start_t = time.time()

		# for i in range(num_walks):
		# 	print(i, num_walks)
		# 	walks.update(pre_n2vwalk(self,nodes,walk_length,i))

		# BaseManager.register('n2v_class', Graph)
		# manager = BaseManager()
		# manager.start()
		# inst = manager.n2v_class()
		# inst.G = self.G
		# walks =  manager.dict()
		# l = manager.list(range(10))
		worker_pool = Pool(workers)
		partial_walk = partial(self.pre_n2vwalk, nodes, walk_length)

		# walks_pool = worker_pool.map_async(partial_walk, range(num_walks), 1)

		prev = 0
		# pbar = tqdm(total=100)
		# print(walks_pool._number_left)

		for results in tqdm(worker_pool.imap_unordered(partial_walk, range(num_walks)), total=num_walks):
			# walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
			for result in results:
				if result[0] in walks:
					walks[result[0]].append(result)
				else:
					walks[result[0]] = [result]



		# while not walks_pool.ready():
		# 	print(walks_pool._number_left)
		# 	current_level =  int(((num_walks - walks_pool._number_left)/num_walks)*100)
		# 	if current_level != prev:
		# 		pbar.update(current_level - prev)
		# 		prev = current_level
		# current_level = int(((num_walks - walks_pool._number_left) / num_walks) * 100)
		# pbar.update(current_level - prev)
        #
		# pbar.close()
        #
        #
        #
		# walks_gather = walks_pool.get()
		# worker_pool.close()
		# worker_pool.join()
		# for walk_iter in walks_gather:
		# 	for node in walk_iter:
		# 		if node in walks:
		# 			walks[node].append(walk_iter[node])
		# 		else:
		# 			walks[node] = [walk_iter[node]]
		cost_t = int(time.time() - start_t)
		print('took {}s to sample'.format(cost_t ))
		print(len(walks))

		# for walk_iter in range(num_walks):
		# 	print str(walk_iter+1), '/', str(num_walks)

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		# G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(self.G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(self.G[dst][dst_nbr]['weight']/p)
			elif self.G.has_edge(dst_nbr, src):
				unnormalized_probs.append(self.G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(self.G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		if norm_const == 0:
			norm_const += 0.00001
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self, workers=1):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		# G = self.G
		is_directed = self.is_directed
		chunksize =10000

		alias_nodes = {}
		total_nodes = len(self.G.nodes())
		optimal = int(np.ceil(total_nodes / chunksize))

		print('PreWalk iteration:')
		n_workers = min(workers, optimal)

		worker_pool = Pool(n_workers)

		partial_walk = partial(self.pre_process_transition, 1, is_directed)
		# walks_pool = worker_pool.map_async(partial_walk, self.G.nodes())
		for results in tqdm(worker_pool.imap_unordered(partial_walk, self.G.nodes(), chunksize=chunksize), total=total_nodes):
        #
			alias_nodes.update(results)
		# prev = 0
		# pbar = tqdm(total=100)
		# while not walks_pool.ready():
		# 	pass
		# # 	current_level = int(((total_nodes - walks_pool._number_left) / total_nodes) * 100)
		# # 	if current_level != prev:
		# # 		pbar.update(current_level - prev)
		# # 		prev = current_level
		# # current_level = int(((total_nodes - walks_pool._number_left) / total_nodes) * 100)
		# pbar.update(50)

		# pbar.close()

		# walks_gather = walks_pool.get()
		worker_pool.close()
		worker_pool.join()

		# for walk_iter in walks_gather:
			# for node in walk_iter:
				# alias_nodes[node] = walk_iter[node]
			# alias_nodes.update(walk_iter)
		# for node in G.nodes():
		# 	unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
		# 	norm_const = sum(unnormalized_probs)
		# 	normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		# 	alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		total_edges = len(self.G.edges())
		optimal = int(np.ceil(total_edges / chunksize))

		n_workers = min(workers, optimal)

		worker_pool = Pool(n_workers)

		partial_walk = partial(self.pre_process_transition, 2, is_directed)
		# walks_pool = worker_pool.map_async(partial_walk, self.G.edges())
		for results in tqdm(worker_pool.imap_unordered(partial_walk, self.G.edges(), chunksize=chunksize), total=total_edges):
			#
			alias_edges.update(results)
		# prev = 0
		# pbar = tqdm(total=100)
		# while not walks_pool.ready():
		# 	pass
		# 	current_level = int(((total_edges - walks_pool._number_left) / total_edges) * 100)
		# 	if current_level != prev:
		# 		pbar.update(current_level - prev)
		# 		prev = current_level
		# current_level = int(((total_edges - walks_pool._number_left) / total_edges) * 100)
		# pbar.update(100)
        #
		# pbar.close()
        #
		# walks_gather = walks_pool.get()
		worker_pool.close()
		worker_pool.join()

		# for walk_iter in walks_gather:
			# for edge in walk_iter:
			# 	alias_edges[edge] = walk_iter[edge]
			# alias_edges.update(walk_iter)


		# if is_directed:
		# 	for edge in G.edges():
		# 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		# else:
		# 	for edge in G.edges():
		# 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		# 		alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	# return csr_matrix(J), csr_matrix(q)
	return J.astype(np.uint16), q.astype(np.float32)
	# return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]