import sys
import math
import scipy.io.arff as arff
import numpy
import random
from random import shuffle
import math
import itertools

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

train_file = sys.argv[1]
test_file = sys.argv[2]
fl = sys.argv[3]

train_data = []
train_class = []
train_labels = []
train_features = []
feature_vec_length = 0
#---------------------------------------------------- Read the training data --------------------------------
data, meta = arff.loadarff(train_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]).replace("'",""))
	train_data.append(l)
	
mstr = str(meta)
mstr = mstr.split("\n\t")
for im in mstr:
	if not im.startswith("class") and "type is numeric" in im:
		train_features.append([im.split('\'s')[0], None])
		feature_vec_length += 1
	elif not im.startswith("class") and "type is nominal" in im:
		feature = im.split('\'s')[0]
		attr = im.split("(")[1].split(")")[0].replace(" ","").replace("'","").split(",")
		train_features.append([feature, attr])
		feature_vec_length += 1
	elif im.startswith("class"):
		flag = 0
		train_labels = im.split("(")[1].split(")")[0].replace(" ","").replace("'","").split(",")
		train_class = ['class', train_labels]

for i in range(len(train_features)):
	for j in range(len(train_features[i][1])):
			train_features[i][1][j] = train_features[i][1][j].replace("\"","")

#Additional processing for single quotes
for i in range(len(train_labels)):
	train_labels[i] = train_labels[i].replace("\"","")
train_class = ['class', train_labels]

#print train_class, train_labels
#print train_features
#print "train data :",train_data[0]
#----------------------------------------------------------Read the test data------------------------------------
test_data = []
data, meta = arff.loadarff(test_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]).replace("'",""))
 	test_data.append(l)
#-----------------------------------------------------------------------------------------------------------------
if fl == 'n': #Naive Bayes
	dict_y = {}
	count_y = {}
	dict_xy = {}
	for i in range(len(train_labels)):
		label = train_labels[i]
		dict_xy[label] = {}
		countl = 0
		for j in range(len(train_data)):
			if train_data[j][feature_vec_length] == label:
				countl += 1
		count_y[label] = countl
		dict_y[label] = float (countl + 1)/(len(train_data) + len(train_labels))

	for key in dict_xy:
		for i in range(len(train_features)):
			dict_xy[key][train_features[i][0]] = {}
				
	#Compute all conditional probabilities
	for key in dict_xy:
		for i in range(len(train_features)): #iterate over all features
			for j in range(len(train_features[i][1])): #iterate over the values of a feature 
				countx_y = 0
				for k in range(len(train_data)):
					if train_data[k][i] == train_features[i][1][j]:
						if train_data[k][feature_vec_length] == key:
							countx_y += 1
				dict_xy[key][train_features[i][0]][train_features[i][1][j]] = float(countx_y + 1)/(count_y[key] + len(train_features[i][1]))

	for i in range(len(train_features)):
		print train_features[i][0] + " " +"class"
	print

	#Use NB classifier
	count_corr = 0
	for i in range(len(test_data)):
		actual = test_data[i][feature_vec_length]

		#Assuming binary classification
		p1 = dict_y[train_labels[0]]
		p2 = dict_y[train_labels[1]]

		for j in range(feature_vec_length):
			p1 *= dict_xy[train_labels[0]][train_features[j][0]][test_data[i][j]]
			p2 *= dict_xy[train_labels[1]][train_features[j][0]][test_data[i][j]]

		if p1 > p2:
			predicted = train_labels[0]
			val = str(p1 /(p1 + p2))
		else:
			predicted = train_labels[1]
			val = str(p2 /(p1 + p2))
		if predicted == actual:
			count_corr += 1
		print predicted + " " + actual + " " + val

	print 
	print count_corr
#-------------------------------------------------------------------------------TAN----------------------------------------------------
elif fl == 't': #TAN

	#Compute P(x | y), P(x1, x2 | y), P(x1, x2, y),
	d = {} # P(x | y)
	d[train_labels[0]] = {}
	d[train_labels[1]] = {}
	for i in range(len(train_features)):
		for j in range(len(train_features[i][1])):
			ct0 = 0
			ct1 = 0
			count0 = 0
			count1 = 0
			for k in range(len(train_data)):
				if train_data[k][feature_vec_length] == train_labels[0]:
					if train_data[k][i] == train_features[i][1][j]:
						ct0 += 1
					count0 += 1
				else:
					if train_data[k][i] == train_features[i][1][j]:
						ct1 += 1
					count1 += 1
			d[train_labels[0]][train_features[i][0] + " " + train_features[i][1][j]] = float (ct0 + 1)/ (count0 + len(train_features[i][1]))
			d[train_labels[1]][train_features[i][0] + " " + train_features[i][1][j]] = float (ct1 + 1)/ (count1 + len(train_features[i][1]))

	d1 = {} # P(x1, x2 | y)
	d1[train_labels[0]] = {}
	d1[train_labels[1]] = {}
	for i in range(len(train_features)):
		for j in range(len(train_features)):
			if i != j :
				for k in range(len(train_features[i][1])):
					x1 = train_features[i][1][k]
					for l in range(len(train_features[j][1])):
						x2 = train_features[j][1][l]
						ct0 = 0
						ct1 = 0
						count0 = 0
						count1 = 0
						for m in range(len(train_data)):
							if train_data[m][feature_vec_length] == train_labels[0]:
								if train_data[m][i] == x1 and train_data[m][j] == x2:
									ct0 += 1
								count0 += 1
							else:
								if train_data[m][i] == x1 and train_data[m][j] == x2:
									ct1 += 1
								count1 += 1
						d1[train_labels[0]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]] = float (ct0 + 1)/(count0 + len(train_features[i][1])*len(train_features[j][1]))
						d1[train_labels[1]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]] = float (ct1 + 1)/(count1 + len(train_features[i][1])*len(train_features[j][1]))
									
							
	d2 = {} # P(x1, x2, y)
	d2[train_labels[0]] = {}
	d2[train_labels[1]] = {}
	for i in range(len(train_features)):
		for j in range(len(train_features)):
			if i != j :
				for k in range(len(train_features[i][1])):
					x1 = train_features[i][1][k]
					for l in range(len(train_features[j][1])):
						x2 = train_features[j][1][l]
						ct0 = 0
						ct1 = 0
						for m in range(len(train_data)):
							if train_data[m][i] == x1 and train_data[m][j] == x2:
								if train_data[m][feature_vec_length] == train_labels[0]:
									ct0 += 1
								else:
									ct1 += 1
						
						d2[train_labels[0]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]] = float (ct0 + 1)/(len(train_data) + 2*len(train_features[i][1])*len(train_features[j][1]))
						d2[train_labels[1]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]] = float (ct1 + 1)/(len(train_data) + 2*len(train_features[i][1])*len(train_features[j][1]))
					
		
	wts = [] #Calculate I( x1, x2 | y)
	for i in range(len(train_features)):
		wt_i = []
		for j in range(len(train_features)):
			if i == j:
				wt_i.append(-1.0)
			else:
				I = 0 
				for k in range(len(train_features[i][1])):
					for l in range(len(train_features[j][1])):
						v1 = d2[train_labels[0]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]]
						v2 = d2[train_labels[1]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]]
						v3 = d1[train_labels[0]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]]
						v4 = d1[train_labels[1]][train_features[i][0]+" "+train_features[i][1][k]+" "+train_features[j][0]+" "+train_features[j][1][l]]
						v5 = d[train_labels[0]][train_features[i][0] + " " + train_features[i][1][k]]
						v6 = d[train_labels[1]][train_features[i][0] + " " + train_features[i][1][k]]
						v7 = d[train_labels[0]][train_features[j][0] + " " + train_features[j][1][l]]
						v8 = d[train_labels[1]][train_features[j][0] + " " + train_features[j][1][l]]
							
						I += v1* math.log((v3/(v5*v7)),2) + v2* math.log((v4/(v6*v8)),2)

				wt_i.append(I)
		wts.append(wt_i)

	#for i in range(len(wts)):
	#	print wts[i]

	# -------------------------------------------Prim's algorithm to find the maximum weight spanning tree-----------
	V = [train_features[0][0]]
	U = [[item[0], 'n'] for item in train_features]
	f = [item[0] for item in train_features]
	U[0][1] = 'y'
	U.reverse()
	#print U
	
	root = Node(train_features[0][0])
	while len(V) < feature_vec_length:
		e = -1000
		for i in range(len(V)):
			for j in range(len(U)):
				if U[j][1] != 'y' and wts[f.index(V[i])][f.index(U[j][0])] > e:
					e = wts[f.index(V[i])][f.index(U[j][0])]
					start = V[i] 
					end = U[j]
		V = [end[0]] + V
		end[1] = 'y'
		new_node = Node(end[0])
		
		to_visit = [root]
		curr_node = None
		while len(to_visit) > 0:
			curr_node = to_visit.pop(0)
			for i in range(len(curr_node.children)):
				to_visit.append(curr_node.children[i])
			if curr_node.data == start:
				curr_node.add_child(new_node)
		
	#Create a dictionary of nodes and the parents
	node_dict = {}
	to_visit = [root]
	while len(to_visit) > 0:
		curr_node = to_visit.pop(0)
		for i in range(len(curr_node.children)):
			if curr_node.children[i].data not in node_dict:
				node_dict[curr_node.children[i].data] = [curr_node.data]
			else:
				node_dict[curr_node.children[i].data].append(curr_node.data)
			to_visit.append(curr_node.children[i])
		
	node_dict[train_features[0][0]] = []
	for key in node_dict:
		node_dict[key].append('class')
	
	cpt = {}
	for i in range(len(train_features)):
		s = train_features[i][0]
		cpt[s] = {}
		for j in range(len(node_dict[train_features[i][0]])):
			s += " "+ node_dict[train_features[i][0]][j]
		print s

	print
	#-------------------------------------------Compute CPT----------------------------------------------------------------
	for i in range(len(train_features)): # for each feature compute its CPT 
		s = train_features[i][0]
		indices = []
		for j in range(len(node_dict[train_features[i][0]]) - 1):
			indices.append(f.index(node_dict[train_features[i][0]][j]))
		
		attrs = []
		for j in range(len(indices)):
			attrs.append(train_features[indices[j]][1])
		attrs.append(train_labels)
		#print indices, attrs
		combs = list(itertools.product(*attrs))

		for j in range(len(train_features[i][1])):
			for k in range(len(combs)):
				ct= 0
				count = 0
				for l in range(len(train_data)):
					sample = []
					for m in range(len(indices)):
						sample.append(train_data[l][indices[m]])
					sample.append(train_data[l][feature_vec_length])
					if list(combs[k]) == sample:
						if train_data[l][i] == train_features[i][1][j]:
							ct += 1
						count += 1
				
				cpt[s][train_features[i][1][j] + " "+ ' '.join(map(str, indices + [feature_vec_length] + list(combs[k])))] = float(ct + 1)/ (count + len(train_features[i][1]))

	dict_y = {}
	for i in range(len(train_labels)):
		label = train_labels[i]
		countl = 0
		for j in range(len(train_data)):
			if train_data[j][feature_vec_length] == label:
				countl += 1
		dict_y[label] = float (countl + 1)/(len(train_data) + len(train_labels))

	count_corr = 0
	for i in range(len(test_data)):
		actual = test_data[i][feature_vec_length]

		p1 = dict_y[train_labels[0]]
		p2 = dict_y[train_labels[1]]
		for j in range(len(train_features)):
			s = train_features[j][0]
			indices = []
			for k in range(len(node_dict[train_features[j][0]]) - 1):
				indices.append(f.index(node_dict[train_features[j][0]][k]))
			indices += [feature_vec_length]
			vals = [test_data[i][v] for v in indices]
	
			vals[len(vals)-1] = train_labels[0]
			p1 *= cpt[s][test_data[i][j] + " "+ ' '.join(map(str, indices + vals))]

			vals[len(vals)-1] = train_labels[1]
			p2 *= cpt[s][test_data[i][j] + " "+ ' '.join(map(str, indices + vals))]

		if p1 > p2:
			predicted = train_labels[0]
			val = str(p1 /(p1 + p2))
		else:
			predicted = train_labels[1]
			val = str(p2 /(p1 + p2))
		if predicted == actual:
			count_corr += 1
		print predicted + " " + actual + " " + val 

	print
	print count_corr
			
			
		
						

				
			
		
	
		
