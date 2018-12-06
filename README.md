# S3: Similar Structure Search algorithm based on graph embedding

## Introduction

It's quite a big challenge to find similar structures of a given substructure in a large scale network. To achieve this goal, we developed **S3** , a similar substructure search algorithm in the paper: [Structure-Based Suggestive Exploration: A New Approach for Effective Exploration of Large Networks](https://ieeexplore.ieee.org/abstract/document/8440813). Based on graph embedding aproaches, when given a substructure in a large network, we can easily find similar structures by S3.

## Requirements

Some Python libraries:

- sklearn
- numpy
- scipy
- networkx

## How to use

1. Compute the graph embedding of your network and get a vector for each node. We recommend some methods(Note: this is not contained by our algorithm yet):

   - GraphWave(best)
   - Graphlet Kernel
   - Feature-based method
   - Struc2Vec
   - Node2Vec

2. Import S3 and try to use, for example: 

   ```python
   search_nodes = [1, 799, 1854] # give the nodes of the exemplar structure
   s = S3(matrix, vectors) # init S3 with the matrix and vectors of the whole graph
   s.exemplar(search_nodes) # eastablish the exemplar structure
   s.search() # begin to search similar substructures in the whole graph
   
   exemplar_compound_graph = s.get_exemplar_compound_graph()
   # get the compound graph of exemplar after perform clustering algorith (Approach Details step 2)
   # {'nodes': [[1, 2], [3, 4, 5], [6], [7, 8]],
   # 'links': [{'source': 0, 'target': 1},
   # {'source': 0, 'target': 2},
   # {'source': 1, 'target': 2},
   # {'source': 1, 'target': 3}]}
   # it repsents a compound graph like this:
   # [1, 2] -- [3, 4, 5] -- [7， 8]
   #    └---[6]---┘
   
   knn_nodes = s.get_knn_nodes()
   # get the list of knn nodes of exemplar according to the embedding results (Approach Details step 3)
   # [1, 3, 21, 43, 657, ...]
   
   connected_components = s.get_knn_connected_components()
   # extract the connected components (Approach Details step 4):
   # [[1, 799, 1854, 1472, 1856, 1857], # nodes of connected component 0
   # [292, 293, 294, 295, 296, 297], # nodes of connected component 1
   # ...]
   
   connected_components_similarity = s.get_connected_components_similarity()
   # get the connected components' similarity to the exemplar structure.
   # [(4, 1.0), #(connected components index, similarity)
   # (0, 0.5477225575051661),
   # (2, 0.5103103630798288),
   # ...]
   
   knn_compound_graphs = s.get_knn_compound_graphs()
   exemplar_knn_maps = s.get_exemplar_to_knn_nodes_maps()
   knn_exemplar_maps = s.get_knn_nodes_to_exemplar_maps()
   ```



## Approach Details

1. Map the given exemplar into vector space(embedding space), it means that we can get a vector for each exemplar node from the graph embedding result. 

   ![Dec-06-2018 10-46-43](./assets/Dec-06-2018 10-46-43.gif)

2. Perform clustering algorithm(DBScan) to cluster the exemplar nodes into a compound graph. Since the exemplar nodes might be similar to each other, this step will improve the precision of the whole search algorithm.

   As we can see, the 4 exemplar nodes now turn to a compound graph with 3 compound nodes(clusters).

   ![Dec-06-2018 10-56-01](./assets/Dec-06-2018 10-56-01.gif)

3. Use k nearest neighbor algorithm to find candidate nodes that are similar to the nodes in the exemplar.

   ![image-20181206105958593](./assets/image-20181206105958593.png)

4. Extract the connected components from the candidate nodes. Since we have compound graph(clusters) generated before, each node in the connected components will be classified into one compound node(cluster), according to the distance between the node and the clusters.

![image-20181206110325811](./assets/image-20181206110325811.png)

5. Filter out components that have too many or too few nodes ,and sort the remaining components according to their similarities to the specified exemplar. After that we get the results which are considered to be similar to the exemplar.

   ![image-20181206111455600](./assets/image-20181206111455600.png)

## Test Results

to be added...