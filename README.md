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

1. First, you should compute the graph embedding of your network and get a vector for each node. We recommend some methods:

   - GraphWave(best)
   - Graphlet Kernel
   - Feature-based method
   - Struc2Vec
   - Node2Vec

2. Second, import S3 and try to use, for example: 

   ```python
   search_nodes = [1, 799, 1854] # give the nodes of the exemplar structure
   s = S3(matrix, vectors) # init S3 with the matrix and vectors of the whole graph
   s.min_nodes_threshold(3) # set minimum nodes count of the similar substructure
   s.max_nodes_threshold(10) # set maximum nodes count of the similar substructure
   s.exemplar(search_nodes) # eastablish the exemplar structure
   res = s.search() # search similar substructure of the exemplar
   ```

## Approach Details

to be added...



## Test Results

to be added...