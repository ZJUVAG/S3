# coding=utf-8
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import os
import numpy as np
import copy
import time
import networkx as nx
import random
import json
from scipy.sparse import csr_matrix, csgraph
from gk_weisfeiler_lehman import compare

from sklearn.cluster import DBSCAN

# binary search algorithm
# find the index of the max element no bigger than x
def bisect_ch(a, x, lo = 0, hi = None):
	if lo < 0:
		raise ValueError('lo must be non-negative')
	if hi is None:
		hi = len(a)
	while lo < hi:
		mid = (lo + hi) // 2
		if a[mid] <= x:
			lo = mid + 1
		else:
			hi = mid
	return lo

class S3():
	'''
	Similar Structure Search Algorithm
	'''
	def __init__(self, matrix, embedding):
		self.matrix = matrix
		self.embedding = embedding
		self.min_nodes_thrsd = 5
		self.max_nodes_thrsd = 100
		self.sim_thrsd = 0.99
		self.k = 100
		self.group = 0.01

	def set_k(self, k):
		self.k = k
	
	def set_max_nodes(self, max):
		self.max_nodes_thrsd = max
	
	def set_min_nodes(self, min):
		self.min_nodes_thrsd = min

	def set_similarity_threshold(self, sim_thrsd):
		self.sim_thrsd = sim_thrsd

	def gl_dbscan(self, nodes, vectors_list = None):
		if vectors_list is None:
			vectors_list = []
			for i in nodes:
				vector = self.embedding[i]
				vectors_list.append(vector)

		dbs = DBSCAN(eps = self.group, min_samples = 2,
					metric = 'euclidean', metric_params = None, algorithm = 'auto', leaf_size = 30, p = None,
					n_jobs = 1).fit(vectors_list)

		label_graph = dbs.fit_predict(vectors_list)
		r = {}
		for i in range(len(label_graph)):
			n = nodes[i]
			label = label_graph[i]
			if label not in r:
				r[label] = []
			r[label].append(n)
		label_arr = []
		# -1 represents that the node doesn't belong any cluster
		if -1 in r:
			for i in r[-1]:
				label_arr.append([i])
			del r[-1]
		label_arr.extend(list(r.values()))
		return label_arr
	
	def gl_search_nodes_extract(self, nodes):  # nodes : id
		"""do clustering on the exemplar nodes(dbscan)

			This function can be used to generate the graph after cluster that
			we called type graph. In the type graph, each type node represents
			one cluster conducted by the dbscan algorithm so that each type node
			in the type graph contains several nodes of the original graph. We
			can establish links between type nodes if there are links between
			the original nodes they contains.

			Parameters
			----------
			nodes: numpy array
				the exemplar nodes to be searched

			Return
			------
			{
				'type_list': list
					the type nodes list defined before
				'type_links': list of object,
					each object contains one source and one target.
					both of them are the type nodes.
				'node_links':
					origin links between the exemplar nodes.
			}
		"""
		nodes = list(map(lambda x: int(x), nodes))
		res = {'type_list': [], 'type_links': [], 'node_links': []}
		label_arr = self.gl_dbscan(nodes)
		res['type_list'] = label_arr
		node_to_type = {}
		for i in range(len(label_arr)):
			for j in label_arr[i]:
				node_to_type[j] = i
		
		ma = self.matrix[np.array(nodes)[:, None], np.array(nodes)] #.toarray()
		x_arr, y_arr = np.where(ma == 1)
		link_set = set([])
		type_set = set([])
		for i in range(len(x_arr)):
			x = int(nodes[x_arr[i]]) # source
			y = int(nodes[y_arr[i]]) # target
			# undirected edges
			if str(x) + '+' + str(y) not in link_set and str(y) + '+' + str(x) not in link_set:
				link_set.add(str(x) + '+' + str(y))
				res['node_links'].append({'source': x, 'target': y})
				x_type = int(node_to_type[x]) # source type
				y_type = int(node_to_type[y]) # target type

				if x_type != y_type and str(x_type) + '+' + str(y_type) not in type_set and str(y_type) + '+' + str(
						x_type) not in type_set:
					type_set.add(str(x_type) + '+' + str(y_type))
					res['type_links'].append({'source': x_type, 'target': y_type})

		return res

	def sub_knn_graph(self, knn_nodes_list):
		"""do clustering on the exemplar nodes(dbscan)

			This function can be used to generate the graph after cluster that
			we called type graph. In the type graph, each type node represents
			one cluster conducted by the dbscan algorithm so that each type node
			in the type graph contains several nodes of the original graph. We
			can establish links between type nodes if there are links between
			the original nodes they contains.

			Parameters
			----------
			nodes: numpy array
				the exemplar nodes to be searched

			opt: string
				'draw' represents that the sketch mode is turned on.

			Return
			------
			{
				'type_list': list
					the type nodes list defined before
				'type_links': list of object,
					each object contains one source and one target.
					both of them are the type nodes.
				'node_links':
					origin links between the exemplar nodes.
			}
		"""
		stack = list(set(knn_nodes_list))
		stack = np.array(stack)

		if len(stack) == 0:
			return {}

		# the matrix of the knn nodes
		ma = self.matrix[stack[:, None], stack]

		# each node will have a label
		# which represents the connected components it belongs to.
		num, labels = csgraph.connected_components(ma, directed = False)

		connected_components = {} # key: label, value: list of nodes
		for i in range(len(labels)):
			if str(labels[i]) not in connected_components:
				connected_components[str(labels[i])] = []
			connected_components[str(labels[i])].append(int(stack[i]))

		# sort the connected components
		sorted_connected_components = sorted(connected_components.items(), key = lambda item: len(item[1]), reverse = True)

		k = 0
		res = {}
		for i, nodes_in_components in sorted_connected_components:
			if len(nodes_in_components) >= self.max_nodes_thrsd:
				continue
			if len(nodes_in_components) < self.min_nodes_thrsd:
				break
			res[str(k)] = nodes_in_components
			k += 1
		return res  # 排序好，去除一个

	def sub_knn_graph_match_new(self, search_nodes, knn_nodes_graph, distant, vectors = []):
		draw_f = True
		if vectors == []:
			draw_f = False
		match_label = {}
		save_nodes = set([])
		r = {}
		rz = 0
		# knn_nodes_graph 已经排好序 去掉只有一个点的了
		search_nodes_arg_np = np.array(search_nodes)
		# search_nodes_arg_ma = gl.matrix[search_nodes_arg_np[:, None], search_nodes_arg_np].toarray()
		# for i in range(len(search_nodes)):
		#
		for ii in knn_nodes_graph.keys():
			sub_list = copy.copy(knn_nodes_graph[ii])  ## 当前的子图
			# sub_set = set(sub_list)  ## 当前的子图
			# if len(i) < 2:  # len(search_nodes):#knn的子图多一点
			# 	continue
			r[str(rz)] = {}
			match_label[str(rz)] = {}
			d = {}
			# sub_list_np = np.array(sub_list)
			# sub_list_ma = gl.matrix[sub_list_np[:,None],sub_list_np].toarray()
			for a_i in range(len(search_nodes)):  # 找两两之间最小值
				a = search_nodes[a_i]
				d[a] = {}
				# a_degree = len(np.where(search_nodes_arg_ma[a_i]==1)[0])
				for b_i in range(len(sub_list)):
					b = sub_list[b_i]
					# b_degree = len(np.where(sub_list_ma[b_i] == 1)[0])
					# if len(distant) == 0:
					if draw_f:
						vector1 = vectors[a]  # 这里是新的向量
						vector2 = self.embedding[b]
						# if gl.dis == 'e':
						# 	d[a][b] = np.linalg.norm(vector1 - vector2)
						# else:
						d[a][b] = 0.5 + 0.5 * np.dot(vector1, vector2) / (
								np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

					# 	d[a][b] = 100
					else:
						if a in distant and str(b) in distant[a]:
							d[a][b] = distant[a][str(b)]
						else:
							d[a][b] = 100
				# if b_degree-a_degree>0:
				# 	d[a][b] = math.sqrt(b_degree-a_degree)
				# else:
				# 	d[a][b] = abs(b_degree-a_degree)
				d[a] = sorted(d[a].items(), key = lambda item: item[1])
			search_index_list = list(range(len(search_nodes)))
			sub_l = 0
			while sub_l != len(sub_list):
				for j in range(len(search_index_list)):  # 当前应该搜的原图点顺序
					node_old = search_nodes[search_index_list[j]]  # 当前原图点
					for vl in range(len(d[node_old])):  # 在最短list里面找没有被选走的sub_knn点
						v = d[node_old][vl][0]
						if v in save_nodes:
							continue
						if str(node_old) not in r[str(rz)]:
							r[str(rz)][str(node_old)] = []
						r[str(rz)][str(node_old)].append(int(v))  # 添加进来
						match_label[str(rz)][int(v)] = int(node_old)
						save_nodes.add(v)  #
						sub_l += 1
						del d[node_old][vl]
						break
					t = search_index_list[j]
					del search_index_list[j]
					search_index_list.append(t)
					break
			rz += 1
		return r, match_label
	
	def search(self, search_nodes):
		# find the vectors of search nodes
		vectors_obj = {}
		vectors_list = []
		for i in search_nodes:
			vector = self.embedding[i]
			vectors_obj[i] = vector
			vectors_list.append(vector)

		# find the knns of search nodes
		# ? some problem with the n_neighbors parameter
		nbrs = NearestNeighbors(n_neighbors = len(self.embedding), algorithm = "auto", metric = 'cosine').fit(self.embedding)
		distances, knn_nodes = nbrs.kneighbors(vectors_list)
		max_e = np.max(distances)
		min_e = np.min(distances)

		# filter out some unsimilar nodes
		knn_nodes_set = set([])
		for i in range(len(search_nodes)):
			# Find the most like elements of the first (1-self.sim_thrsd)
			# ? some problem with the meaning the sim_thrsd(cos_min)
			index = bisect_ch(distances[i], (max_e - min_e) * (1 - self.sim_thrsd) + min_e)
			if index > self.k:
				index = self.k
			knn_nodes_set = knn_nodes_set | set(knn_nodes[i][0:index])
		
		type_graph = self.gl_search_nodes_extract(search_nodes)
		knn_nodes_graph = self.sub_knn_graph(knn_nodes_set)
		knn_graph_match, mathch_label = self.sub_knn_graph_match_new(search_nodes, knn_nodes_graph, [], vectors_obj)
		knn_type_graphs = [[]] * len(knn_graph_match.keys())

		r = {}
		for i in range(len(type_graph['type_list'])):
			for j in type_graph['type_list'][i]:
				r[j] = i

		for i, v in knn_graph_match.items():
			knn_type_graphs[int(i)] = []
			for ii in range(len(type_graph['type_list'])):
				knn_type_graphs[int(i)].append([])
			for j, vj in v.items():
				knn_type_graphs[int(i)][r[int(j)]].extend(vj)  # !!!

		simi_obj = {}
		time_start1 = time.clock()
		for i, v in knn_nodes_graph.items():
			v_np = np.array(v)
			labels = []
			for j in v:
				labels.append(mathch_label[i][j])
			v_ma = self.matrix[v_np[:, None], v_np]
			labels_set = set(labels)
			search_nodes_now = list(filter(lambda x: x in labels_set, search_nodes))
			search_nodes_now_np = np.array(search_nodes_now)
			search_nodes_now_ma = self.matrix[search_nodes_now_np[:, None], search_nodes_now_np]
			simi_obj[str(i)] = compare(search_nodes_now_ma, v_ma, search_nodes_now, labels)
		simi_obj = sorted(simi_obj.items(), key = lambda item: item[1], reverse = True)
		r = {}
		rz = 0
		simi_map = {}
		for i in simi_obj:
			idd = i[0]
			v = i[1]
			r[str(rz)] = knn_graph_match[str(idd)]
			simi_map[str(rz)] = v
			rz += 1
		knn_graph_match = r
		zzz = 0
		for i, v in simi_map.items():
			zzz += float(v)
		if simi_map:
			zzz = zzz / len(simi_map)
		return knn_graph_match, type_graph, knn_type_graphs, simi_map, zzz, knn_nodes_graph


if __name__ == '__main__':
	# search_nodes = [0, 1, 2, 3, 4]
	search_nodes = [1, 799, 1854]
	vectors = np.load('./data/author_ma_vetor.npy')
	matrix = np.load('./data/author_ma.npy')
	s = S3(matrix, vectors)
	res = s.search(search_nodes)
	# res = back_test(search_nodes, k, cos_min, max_num, min_num)
	print(res)
