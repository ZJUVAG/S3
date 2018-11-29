import numpy as np
import os


def deal(f1, f2, name):
	author = np.loadtxt(f1, dtype = np.str, delimiter = '\n')
	obj = {}
	for i in range(len(author)):
		obj[author[i]] = i
	ma = np.zeros([len(author), len(author)], dtype = int)
	links = []
	relation = np.loadtxt(f2, dtype = np.str, delimiter = '\n')
	for i in range(len(relation)):
		line = relation[i].split(',')
		source = obj[line[0]]
		target = obj[line[1]]
		links.append([source, target])
		ma[source, target] = 1
		ma[target, source] = 1
	np.save('./data/' + name, ma)
	np.savetxt('./data/' + name, links, fmt = '%d')
	os.system('RAGE.exe ./data/' + name)
	f2 = './Results/UNDIR_RESULTS_.csv'
	tmp = np.loadtxt(f2, dtype = np.str, delimiter = ",")
	data = tmp[1:, 0:len(tmp[0]) - 1].astype('int64')
	vectors_obj = {}
	vectors_list = []
	for i in data:
		nodeId = int(i[0])
		vector = i[1:]
		vectors_obj[nodeId] = vector
	for i in range(len(data)):
		if i in vectors_obj:
			vectors_list.append(vectors_obj[i])
		else:
			vectors_list.append([0]*11)
	np.save('./data/' + name + '_vetor', np.array(vectors_list))


if __name__ == '__main__':
	f1 = './data/authors.csv'
	f2 = './data/relations.csv'
	name = 'author_ma'
	deal(f1, f2, name)
