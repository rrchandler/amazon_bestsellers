import pandas as pd
import numpy as np
import matplotlib
from IPython.display import display
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import resample
from sklearn.tree import export_graphviz
import graphviz

from scipy.stats import randint
from scipy import stats

from IPython.display import Image
import graphviz

from random import seed
from random import randrange
import time
from csv import reader
from math import sqrt
from math import inf

# Load CSV File
def load_csv(filename, features):
 dataset = list()
 rowNum = 1
 with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        if rowNum == 1:
            rowNum = 2
            features.append(row)
        else:
            dataset.append(row)
 return dataset
            
# Table to Float
def table_to_float(dataset):
	for i in range(0, len(dataset[0])):
		for row in dataset: 
			if row[i] == 'True':
				row[i] = 1.0
			if row[i] == 'False':
				row[i] = 0.0
			if isinstance(row[i], str):
				row[i] = float(row[i])

def cross_validation_split(dataset, k_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / k_folds)
	for i in range(k_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(0, len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def calcGini(groups, classes):
	total = 0.0
	for grouping in groups:
		total += float(len(grouping))
	gini = 0.0
	for grouping in groups:
		if float(len(grouping)) == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in grouping].count(class_val) / float(len(grouping))
			score += p * p
		gini += (1.0 - score) * (float(len(grouping)) / total)
	return gini

def splitDataset(index, value, dataset):
	right, left = list(), list()
	for row in dataset:
		if row[index] >= value:
			right.append(row)
		else:
			left.append(row)
	return left, right

def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = float(inf), float(inf), float(inf), None
	features = list()
	for x in range(n_features):
		index = randrange(0, len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = splitDataset(index, row[index], dataset)
			gini = calcGini(groups, class_values)
			if gini < b_score:
				b_index, b_groups, b_value, b_score, = index, groups, row[index], gini
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def leaf(group):
	outcomes = list()
	for row in group:
		outcomes.append(row[-1])
	return max(set(outcomes), key=outcomes.count)

def splitNode(node, maxDepth,  min_sample_split, numFeatures, depth):
	left, right = node['groups']
	del(node['groups'])
	if not right or not left:
		node['left'] = node['right'] = leaf(right + left)
		return
	if depth >= maxDepth:
		node['left'], node['right'] = leaf(left), leaf(right)
		return
	if len(right) > min_sample_split:
		node['right'] = get_split(right, numFeatures)
		splitNode(node['right'], maxDepth,  min_sample_split, numFeatures, depth+1)
	else:
		node['right'] = leaf(right)
	if len(left) >  min_sample_split:
		node['left'] = get_split(left, numFeatures)
		splitNode(node['left'], maxDepth,  min_sample_split, numFeatures, depth+1)		
	else:
  		node['left'] = leaf(left)

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def subsample(dataset, ratio):
	sample = list()
	numSample = round(len(dataset) * ratio)
	for x in range(numSample):
		index = randrange(0, len(dataset))
		sample.append(dataset[index])
	return sample

def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

def fit(train, test, maxDepth, min_sample_split, sample_size, numTrees, numfeatures):
	trees = list()
	for i in range(numTrees):
		trees.append(decisionTree(subsample(train, sample_size), maxDepth, min_sample_split, numfeatures))
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

def decisionTree(train, maxDepth,  min_sample_split, numFeatures):
	root = get_split(train, numFeatures)
	splitNode(root, maxDepth,  min_sample_split, numFeatures, 1)
	return root

def evaluate_algorithm(dataset, k_folds, maxDepth,  min_sample_split, sample_size, n_trees, n_features):
	folds = cross_validation_split(dataset, k_folds)
	acc, rec, pre = list(), list(), list()

	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None

		y_pred = fit(train_set, test_set, maxDepth,  min_sample_split, sample_size, n_trees, n_features)
		y_test = [row[-1] for row in fold]
		acc.append(accuracy_score(y_test, y_pred)* 100.00)
		pre.append(precision_score(y_test, y_pred)* 100.0)
		rec.append(recall_score(y_test, y_pred)* 100.0)
	return acc, rec, pre

filename = 'downsampled.csv'
features = list()
dataset = load_csv(filename, features)

# Convert everything to floats
table_to_float(dataset)

seed(2)
k_folds = 2
maxDepth = 2
min_sample_split = 5
sample_size = 0.1
numFeatures = int(sqrt(len(dataset[0])-1))
numTrees = 1

start = time.time()
acc, pre, rec = evaluate_algorithm(dataset, k_folds, maxDepth, min_sample_split, sample_size, numTrees, numFeatures)
print('Trees: %d' % numTrees)
print('maxDepth: %d' % maxDepth)
print('k_folds: %d' % k_folds)
print('sample_size: %f' % sample_size)
print('Mean Accuracy: %.3f%%' % (sum(acc)/float(len(acc))))
print('Mean Recall: %.3f%%' % (sum(rec)/float(len(rec))))
print('Mean Precision: %.3f%%' % (sum(pre)/float(len(pre))))
end = time.time()
print(end - start)
print("seconds")

print("hi")
#random state, n estimators default criterion, max depth, min size, n features, n estimators

# Sources:
# https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
# https://www.ritchieng.com/machinelearning-one-hot-encoding/

# https://www.datacamp.com/tutorial/random-forests-classifier-python
# https://thecleverprogrammer.com/2020/12/17/why-random_state42-in-machine-learning/
# https://www.w3schools.com/python/ref_func_open.asp
# https://wellsr.com/python/upsampling-and-downsampling-imbalanced-data-in-python/
# https://www.geeksforgeeks.org/how-to-remove-an-item-from-the-list-in-python/

