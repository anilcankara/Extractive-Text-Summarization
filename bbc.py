import os, glob
from os.path import dirname, abspath
import pickle
import random
import numpy
from copy import deepcopy

#HER ŞEYİ LOWERCASE YAP
#FOR LOOPLAR SIRALI MI DÖNÜYOR BAK

def remove_punctuations(text):
	marks = ["'", '%', ',', '$', '"'] 		    #This list should be extended
	for mark in marks:							# For every punctuation in the puctuations list
		text = text.replace(mark, " ")			# Replace the mark with a whitespace 
	return text


with open('glove.6B.200d.pkl', 'rb') as f:
    data = pickle.load(f)

#print(str(data["s"]))

ARTICLES_PATH = dirname(dirname(abspath(__file__))) + '\\Upload' + '\\articles'
#print(ARTICLES_PATH)

def parse_documents_to_sentences():
    docs_by_sentences = {}

    for filename in glob.glob(os.path.join(ARTICLES_PATH, 'business_007.txt')):
        file = open(filename, 'r')
        punctuation_free = remove_punctuations(file.read())
        sentences = punctuation_free.split('.')  # "...", "?"
        docs_by_sentences[filename] = sentences
    return docs_by_sentences

#print(str(parse_documents_to_sentences()))

docs_by_sentences = parse_documents_to_sentences()
#print(str(docs_by_sentences))


def sent2vec(docs_by_sentences):
    docs_as_vectors = {}
    # SOME CODE HERE
    for key in docs_by_sentences:
    	allSentences = []
    	sentences = docs_by_sentences[key]
    	for sentence in sentences:
    		sentence = sentence.strip()
    		if sentence == "":
    			continue
    		#print("Sentence is : " + str(sentence))
    		sentenceVector = [0] * len(data["table"])
    		sentenceTokens = sentence.strip().split()
    		divider = len(sentenceTokens)
    		for word in sentenceTokens:
    			if word.lower() in data:
    				wordVector = data[word.lower()]
    				sentenceVector = [x + y for x, y in zip(sentenceVector, wordVector)]
    			else:
    				divider = divider  - 1  # ignore the word

    		if divider == 0:    # it means none of the words in the sentence are in the vocabulary
    		    continue      
    		sentenceVector = [x / divider for x in sentenceVector]
    		allSentences.append(sentenceVector)
    	docs_as_vectors[key] = allSentences
    return docs_as_vectors

docs_as_vectors = sent2vec(docs_by_sentences)
#print(docs_as_vectors)

def calculate_distance(a,b):
	#print(str(a))
	vector1 = numpy.array(a)
	vector2 = numpy.array(b)
	distance = numpy.linalg.norm(vector1-vector2)
	return numpy.linalg.norm(distance)
#print(calculate_distance([5,5,5,5], [5,5,5,5]))
def recalculate_centers(clusters):
	newCenters = []
	for cluster in clusters:
		newCenter = [0] * 3
		for sentence in cluster:
			#print("Problem: " + str(sentence) )
			newCenter = [x + y for x, y in zip(newCenter, sentence)]
		divider = len(cluster)
		newCenter = [x / divider for x in newCenter]
		newCenters.append(newCenter)
	return newCenters

def cluster(docs_as_vectors):
    extracted_summaries = {}
    # SOME CODE HERE
    K = 2
    # For each document
    for i in range(1):
    	sentenceVectors = docs_as_vectors
    	# Pick K vectors randomly from the document
    	seeds = random.sample(sentenceVectors, K)
    	print("Initial seeds: " + str(seeds))
    	clusters = [[]] * K
    	#print("Empty clusters: " + str(clusters))
    	clusterCenters = seeds   # initially, centers are the seeds

    	'''
    	clusters[0] = clusters[0] + [seeds[0]]
    	clusters[1] = clusters[1] + [seeds[1]]
    	print("Clusters'ın yeni hali: " + str(clusters))
    	'''
    	'''
    	for x, cluster in enumerate(clusters):
    		print("Seeds'deki " + str(x) + " inci eleman ekleniyor: " + str(seeds[x]))
    		clusters[x].append(seeds[x])
    		print("Clusters'ın yeni hali: " + str(clusters))
    	'''
    	'''
    	for (a, b) in zip(clusters, seeds): 
    		a.append(b)
    	'''
    	cList = []
    	for seed in seeds:
    		cluster = [seed]
    		cList.append(cluster)
    	print("CList: " + str(cList))



    	print("Initial clusters : " + str(cList))
    	dummyVector = [0] * 3
    	oldCenters = [dummyVector] * K
    	error = 1
    	while error > 0.1:
    		error = calculate_distance(oldCenters, clusterCenters)
    		for i, sentence in enumerate(sentenceVectors):
    			print("Working on this sentence now: " + str(sentence))
    			minDistance  = float('inf') # BUNA BAK
    			minDistanceCluster = -1
    			print("Cluster centers are: " + str(clusterCenters))
    			for j, center in enumerate(clusterCenters):
    				distance = calculate_distance(sentence,center)
    				if distance < minDistance:
    					minDistance = distance
    					minDistanceCluster = j

    			print("Min distance cluster is->  " + str(minDistanceCluster) )
    			if sentence not in cList[minDistanceCluster]:
    				print("ENTERED HERE" )
    				for cluster in cList:
    					if sentence in cluster:
    						cluster.remove(sentence)
    						break
    				cList[minDistanceCluster].append(sentence)
    				
    		
    		oldCenters = deepcopy(clusterCenters)
    		#print("Old centers is: " + str(oldCenters))
    		clusterCenters = recalculate_centers(cList)   
    		#print("Cluster centers are updated.")   
    		print("Error is: " + str(error)) 
    		print("CList is updated as: " + str(cList))
    		print("Cluster centers are: " + str(clusterCenters))
    	
    return extracted_summaries

#cluster(docs_as_vectors)
cluster([[12,34,21], [23,34,23], [25,36,25]])


def evaluation(gold_summaries, extracted_summaries):
    rouge_scores = []
    # SOME CODE HERE
    return rouge_scores

