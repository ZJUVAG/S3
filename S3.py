# coding=utf-8
from sklearn.neighbors import NearestNeighbors
import numpy as np
import copy
from scipy.sparse import csgraph
from gk_weisfeiler_lehman import compare
from sklearn.cluster import DBSCAN

# binary search algorithm
# find the index of the max element no bigger than x
def binary_search(a, x, lo = 0, hi = None):
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
		
		# parameters
		self.min_nodes_thrsd = 5 # min nodes count threshold of the detected structure
		self.max_nodes_thrsd = 100 # max nodes count threshold of the detected structure
		self.sim_thrsd = 0.99
		self._k = 100
		self._epsilon = 0.01
		
		# exemplar
		self.exemplar_nodes = []
		self.exemplar_vectors = []
		self.exemplar_compound_graph = None

		# knn
		self.knn_nodes = None
		self.knn_connected_components = None
		self.exemplar_knn_maps = None
		self.knn_exemplar_maps = None
		self.knn_compound_graphs = None
		self.connected_components_similarity = None
	
	def k(self, k = None):
		if k is None:
			return self._k
		else:
			self._k = k
	
	def max_nodes_threshold(self, max = None):
		if max is None:
			return self.max_nodes_thrsd
		else:
			self.max_nodes_thrsd = max
	
	def min_nodes_threshold(self, min = None):
		if min is None:
			return self.min_nodes_thrsd
		else:
			self.min_nodes_thrsd = min

	def similarity_threshold(self, sim_thrsd = None):
		if sim_thrsd is None:
			return self.sim_thrsd
		else:
			self.sim_thrsd = sim_thrsd

	def epsilon(self, epsilon = None):
		if epsilon is None:
			return self._epsilon
		else:
			self._epsilon = epsilon

	def exemplar(self, nodes):
		self.exemplar_nodes = nodes
		self.exemplar_compound_graph = None
		self.knn_nodes = None
		self.knn_connected_components = None
		self.exemplar_knn_maps = None
		self.knn_exemplar_maps = None
		self.knn_compound_graphs = None
		self.connected_components_similarity = None
		self.exemplar_vectors_list = []
		self.exemplar_vectors_dict = {}
		for i in nodes:
			vector = self.embedding[i]
			self.exemplar_vectors_dict[i] = vector
			self.exemplar_vectors_list.append(vector)		

	def __exemplar_dbscan__(self):
		dbs = DBSCAN(eps = self._epsilon, min_samples = 2,
					metric = 'euclidean', metric_params = None, algorithm = 'auto', leaf_size = 30, p = None,
					n_jobs = 1).fit(self.exemplar_vectors_list)

		label_graph = dbs.fit_predict(self.exemplar_vectors_list)
		r = {}
		for i in range(len(label_graph)):
			n = self.exemplar_nodes[i]
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
	
	def __compute_connected_components__(self, nodes_list):
		""" get the connected components with given nodes list
			filter out the components with too large size（ >= max_nodes_thrsd )
			and the components with too small size ( < min_nodes_thrsd )

			Parameters
			----------
			nodes_list: list

			Return
			------
			res: dict
				key: the sequence, value: the components
		"""
		stack = np.array(nodes_list)

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

		connected_components_filtered = []
		for i, nodes_in_components in sorted_connected_components:
			if len(nodes_in_components) >= self.max_nodes_thrsd:
				continue
			if len(nodes_in_components) < self.min_nodes_thrsd:
				break
			connected_components_filtered.append(nodes_in_components)
		return connected_components_filtered

	def __map_knn_connected_components_to_exemplar__(self):
		""" map nodes in the knn connected components to exemplar nodes,
			establish correspondence between them

			Parameters
			----------
			knn_connected_components: list
			[
				[knn_nodes_00, knn_nodes_01, ...],
				[knn_nodes_10, knn_nodes_11, ...],
				...
			]

			Return
			------
			exemplar_knn_maps: dict
			example:
			{
				0: {
					exemplar0: [knn_nodes_00, knn_nodes_01, ...],
					exemplar1: [knn_nodes_10, knn_nodes_11, ...],
					...
				},
				...
			}
			knn_exemplar_maps: dict
			the reverse of exemplar_knn_maps, example:
			{
				0: {
					knn_nodes_0: exemplar0,
					knn_nodes_1: exemplar1
				},
				...
			}
		"""
		if self.exemplar_knn_maps is not None and self.knn_exemplar_maps is not None:
			return self.exemplar_knn_maps, self.knn_exemplar_maps
		
		if self.exemplar_nodes is None or len(self.exemplar_nodes) <= 0:
			raise Exception("Empty Exemplar, try to use S3.exemplar()!")
		try:
			self.knn_connected_components = self.get_knn_connected_components()
		except Exception:
			raise Exception
		else:
			knn_connected_components = self.knn_connected_components

		mapped_nodes = set([])
		exemplar_knn_maps = {}
		knn_exemplar_maps = {}
		# establish the map from exemplar nodes to the nodes in knn components
		for seq in range(len(knn_connected_components)):
			# knn conneceted components nodes
			knn_nodes = copy.copy(knn_connected_components[seq])
			exemplar_knn_maps[seq] = {}
			knn_exemplar_maps[seq] = {}
			distance = {}

			# calculate the distance matrix between exemplar nodes and knn nodes
			for i in range(len(self.exemplar_nodes)):
				exemplar_node = self.exemplar_nodes[i]
				distance[exemplar_node] = {}
				for j in range(len(knn_nodes)):
					knn_node = knn_nodes[j]
					vector1 = self.exemplar_vectors_dict[exemplar_node] # * prepare for sketch mode
					vector2 = self.embedding[knn_node]
					# cosine distance, map [-1, 1) to [0, 1)
					distance[exemplar_node][knn_node] = 0.5 + 0.5 * np.dot(vector1, vector2) / (
							np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
				# sort with ascending order
				distance[exemplar_node] = sorted(distance[exemplar_node].items(), key = lambda item: item[1])
			
			i = 0
			while i != len(knn_nodes):
				# evenly distributed
				exemplar_node = self.exemplar_nodes[i % len(self.exemplar_nodes)]
				
				most_similar_knn_node = None
				# find the most similar knn node which has not been mapped before
				for j in range(len(distance[exemplar_node])):
					most_similar_knn_node = distance[exemplar_node][j][0] # the most similar knn node
					if most_similar_knn_node not in mapped_nodes: # if the node has already been mapped
						del distance[exemplar_node][j]
						break
				
				if most_similar_knn_node is not None:
					if exemplar_node not in exemplar_knn_maps[seq]:
						exemplar_knn_maps[seq][exemplar_node] = []
					exemplar_knn_maps[seq][exemplar_node].append(most_similar_knn_node)
					knn_exemplar_maps[seq][most_similar_knn_node] = exemplar_node
					mapped_nodes.add(most_similar_knn_node)
					i += 1

		self.exemplar_knn_maps = exemplar_knn_maps
		self.knn_exemplar_maps = knn_exemplar_maps
		return exemplar_knn_maps, knn_exemplar_maps
	
	def get_exemplar_compound_graph(self): 
		"""do clustering on the exemplar nodes(dbscan)
			This function can be used to generate the graph after cluster that
			we call compound graph. In the compound graph, each compound node represents
			one cluster conducted by the dbscan algorithm so that each compound node
			in the compound graph contains several nodes of the original graph. We
			can establish links between compound nodes if there are links between
			the original nodes they contains.

			Parameters
			----------
			Return
			------
			{
				'nodes': list
					the compound nodes list defined before
				'links': list,
					each list item contains one source and one target.
					both of them are the compound nodes.
			}
		"""
		if self.exemplar_compound_graph is not None:
			return self.exemplar_compound_graph
		
		if self.exemplar_nodes is None or len(self.exemplar_nodes) <= 0:
			raise Exception("Empty Exemplar, try to use S3.exemplar()!")
		
		nodes = list(map(lambda x: int(x), self.exemplar_nodes))
		compound_graph = {'nodes': [], 'links': []}
		label_arr = self.__exemplar_dbscan__()
		compound_graph['nodes'] = label_arr
		node_to_compound_node = {}
		for i in range(len(compound_graph['nodes'])):
			for j in compound_graph['nodes'][i]:
				node_to_compound_node[j] = i
		
		ma = self.matrix[np.array(nodes)[:, None], np.array(nodes)] #.toarray()
		x_arr, y_arr = np.where(ma == 1)
		
		compound_link_set = set([])
		for i in range(len(x_arr)):
			x = int(nodes[x_arr[i]]) # source
			y = int(nodes[y_arr[i]]) # target
			source_compound_node = int(node_to_compound_node[x])
			target_compound_node = int(node_to_compound_node[y])

			is_same_compund_node = source_compound_node == target_compound_node
			is_added_before = str(source_compound_node) + '+' + str(target_compound_node) in compound_link_set

			if not is_same_compund_node and not is_added_before:
				compound_link_set.add(str(source_compound_node) + '+' + str(target_compound_node))
				compound_link_set.add(str(target_compound_node) + '+' + str(source_compound_node))

				compound_graph['links'].append({'source': source_compound_node, 'target': target_compound_node})

		self.exemplar_compound_graph = compound_graph
		return compound_graph

	def get_knn_nodes(self):
		if self.knn_nodes is not None:
			return self.knn_nodes
		if self.exemplar_nodes is None or len(self.exemplar_nodes) <= 0:
			raise Exception("Empty Exemplar, try to use S3.exemplar()!")
		else:
			# find the knns of search nodes
			# ? some problem with the n_neighbors parameter
			nbrs = NearestNeighbors(n_neighbors = len(self.embedding), algorithm = "auto", metric = 'cosine').fit(self.embedding)
			distances, knn_nodes = nbrs.kneighbors(self.exemplar_vectors_list)
			max_dis = np.max(distances)
			min_dis = np.min(distances)

			# filter out some unsimilar nodes
			knn_nodes_set = set([])
			for i in range(len(self.exemplar_nodes)):
				# Find the most like elements of the first (1-self.sim_thrsd)
				# ? some problem with the meaning the sim_thrsd(cos_min)
				index = binary_search(distances[i], (max_dis - min_dis) * (1 - self.sim_thrsd) + min_dis)
				if index > self._k:
					index = self._k
				knn_nodes_set = knn_nodes_set | set(knn_nodes[i][0:index])

			self.knn_nodes = list(knn_nodes_set)
			return self.knn_nodes

	def get_knn_connected_components(self):
		if self.knn_connected_components is not None:
			return self.knn_connected_components
		try:
			self.knn_nodes = self.get_knn_nodes()
		except Exception as error:
			raise Exception(error)
		else:
			if self.knn_nodes is None or len(self.knn_nodes) <= 0:
				raise Exception('Cannot find kNN nodes!')
			else:
				self.knn_connected_components = self.__compute_connected_components__(self.knn_nodes)
				return self.knn_connected_components
	
	def get_knn_compound_graphs(self):
		try:
			self.exemplar_compound_graph = self.get_exemplar_compound_graph()
		except Exception as error:
			print(error)
		else:
			# set each exemplar node's label: the compound graph node which it belongs to
			exemplar_nodes_label = {}
			for i in range(len(self.exemplar_compound_graph['nodes'])):
				for j in self.exemplar_compound_graph['nodes'][i]:
					exemplar_nodes_label[j] = i

			# compound graph for each connected components
			knn_compound_graphs = []
			for x in self.knn_connected_components:
				knn_compound_graphs.append([])
			for i, exemplar_knn_map in self.exemplar_knn_maps.items(): # exemplar_knn_map.keys(): exemplar_node, exemplar_knn_map.values(): knn nodes
				i = int(i) # i is the sequence of the connected components
				for node in self.exemplar_compound_graph['nodes']:
					knn_compound_graphs[i].append([])
				for exemplar_node, knn_nodes in exemplar_knn_map.items():
					compound_label = exemplar_nodes_label[exemplar_node]
					knn_compound_graphs[i][compound_label].extend(knn_nodes)

			self.knn_compound_graphs = knn_compound_graphs
			return knn_compound_graphs
	
	def get_connected_components_similarity(self):
		# connected components similarity, key: components label, value: similarity with exemplar
		if self.connected_components_similarity is not None and len(self.connected_components_similarity) >= 0:
			return self.connected_components_similarity

		cnntd_cmpnts_similarity = {}
		i = 0
		for conneceted_component_nodes in self.knn_connected_components:
			labels = [] # the exemplar nodes
			for j in conneceted_component_nodes:
				labels.append(self.knn_exemplar_maps[i][j])

			conneceted_component_nodes = np.array(conneceted_component_nodes)
			conneceted_components_matrix = self.matrix[conneceted_component_nodes[:, None], conneceted_component_nodes]

			labels_set = set(labels)
			# filter out the exemplar nodes which doesn't have correspondence with knn nodes
			exemplar_nodes_filtered = list(filter(lambda x: x in labels_set, self.exemplar_nodes))
			exemplar_nodes_filtered_np = np.array(exemplar_nodes_filtered)
			exemplar_nodes_filtered_matrix = self.matrix[exemplar_nodes_filtered_np[:, None], exemplar_nodes_filtered_np]

			# get the similarity between exemplare and connected_components
			cnntd_cmpnts_similarity[i] = compare(exemplar_nodes_filtered_matrix, conneceted_components_matrix, exemplar_nodes_filtered, labels)
			
			i += 1

		cnntd_cmpnts_similarity = sorted(cnntd_cmpnts_similarity.items(), key = lambda item: item[1], reverse = True)

		self.connected_components_similarity = cnntd_cmpnts_similarity
		return cnntd_cmpnts_similarity
	
	def get_exemplar_to_knn_nodes_maps(self):
		if self.exemplar_knn_maps is None:
			self.exemplar_knn_maps, self.knn_exemplar_maps = self.__map_knn_connected_components_to_exemplar__()
		return self.exemplar_knn_maps

	def get_knn_nodes_to_exemplar_maps(self):
		if self.knn_exemplar_maps is None:
			self.exemplar_knn_maps, self.knn_exemplar_maps = self.__map_knn_connected_components_to_exemplar__()
		return self.knn_exemplar_maps

	def search(self):
		try:
			self.exemplar_compound_graph = self.get_exemplar_compound_graph()
			self.knn_nodes = self.get_knn_nodes()
			self.knn_connected_components = self.get_knn_connected_components()
			# establish correnspondence
			self.exemplar_knn_maps, self.knn_exemplar_maps = self.__map_knn_connected_components_to_exemplar__()
			self.connected_components_similarity = self.get_connected_components_similarity()
		except Exception as error:
			print(error)

if __name__ == '__main__':
	search_nodes = [1, 799, 1854]
	vectors = np.load('./data/author_ma_vetor.npy')
	matrix = np.load('./data/author_ma.npy')
	s = S3(matrix, vectors)
	s.min_nodes_threshold(3)
	s.max_nodes_threshold(10)
	s.exemplar(search_nodes)
	s.search()
	exemplar_compound_graph = s.get_exemplar_compound_graph()
	knn_nodes = s.get_knn_nodes()
	connected_components = s.get_knn_connected_components()
	connected_components_similarity = s.get_connected_components_similarity()
	
	exemplar_knn_maps = s.get_exemplar_to_knn_nodes_maps()
	knn_exemplar_maps = s.get_knn_nodes_to_exemplar_maps()