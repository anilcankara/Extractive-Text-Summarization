import os, glob
from os.path import dirname, abspath
import pickle
import random
import numpy
from copy import deepcopy
from rouge import Rouge
import re
from operator import itemgetter

#HER ŞEYİ LOWERCASE YAP
#FOR LOOPLAR SIRALI MI DÖNÜYOR BAK

def remove_punctuations(text):
	marks = ["'", '%', ',', '$', '"', "\\n", ";"] 		    #This list can be extended
	for mark in marks:							# For every punctuation in the puctuations list
		if mark == ',':
			text = text.replace(mark, "")
		else:
			text = text.replace(mark, " ")			# Replace the mark with a whitespace 
	return text


with open('glove.6B.200d.pkl', 'rb') as f:
    data = pickle.load(f)

ARTICLES_PATH = dirname(dirname(abspath(__file__))) + '\\Upload' + '\\articles'
GOLD_SUMMARIES_PATH = dirname(dirname(abspath(__file__))) + '\\Upload' + '\\gold_summaries'
#print(ARTICLES_PATH)

def parse_documents_to_sentences():
    docs_by_sentences = {}

    for filename in glob.glob(os.path.join(ARTICLES_PATH, '*.txt')):
        file = open(filename, 'r')
        next(file)          										# Skip the header of the article
        punctuation_free = remove_punctuations(file.read())         # Remove punctuations
        punctuation_free = punctuation_free.replace('...', '.')
        punctuation_free = punctuation_free.replace('?', '.')
        punctuation_free = punctuation_free.replace('!', '.')
        punctuation_free = punctuation_free.replace('. ', '.')
        #punctuation_free = punctuation_free.replace('.\\n', '.')
        sentences = punctuation_free.split('.')  # "...", "?" OR . \n
        #sentences = re.split('.| ?| !', punctuation_free)
        #print(str(sentences))
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
    	for position, sentence in enumerate(sentences):
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
    		allSentences.append([sentenceVector, sentence, position])
    	docs_as_vectors[key] = allSentences
    return docs_as_vectors

docs_as_vectors = sent2vec(docs_by_sentences)


def calculate_distance(a,b):
	vector1 = numpy.array(a)
	vector2 = numpy.array(b)
	distance = numpy.linalg.norm(vector1-vector2)
	return numpy.linalg.norm(distance)

def recalculate_centers(clusters, oldCenters):
	newCenters = []
	for i, cluster in enumerate(clusters):
		oldCenter = oldCenters[i]                  # Burda hata yoktur diye umuyorum
		divider = len(cluster)
		if divider == 0:
			newCenters.append(oldCenter)
			continue
		newCenter = [0] * 200
		for sentence in cluster:
			newCenter = [x + y for x, y in zip(newCenter, sentence[0])]
		
		newCenter = [x / divider for x in newCenter]
		newCenters.append(newCenter)
	return newCenters

def pick_most_representative(cluster, clusterCenter):
	minDistance = float('inf')
	representativeSentence = ""
	representativePosition = 0
	for sentenceWithString in cluster:
		distance = calculate_distance(sentenceWithString[0], clusterCenter)
		if distance < minDistance:
			minDistance = distance
			representativeSentence = sentenceWithString[1]
			representativePosition = sentenceWithString[2]
	return [representativeSentence, representativePosition]



def cluster(docs_as_vectors, isFixedKModel):
    extracted_summaries = []
    # SOME CODE HERE
    K = 2
    # For each document
    for key in docs_as_vectors:

    	sentenceVectors = docs_as_vectors[key]  # Burda hem vektörler hem stringler var
    	#strings = docs_as_vectors[key][1]
    	#print("Size is: " + str(len(sentenceVectors)) + " in document " + str(key))
    	if isFixedKModel == False:
    		K = int(len(sentenceVectors) / 3)

    	# Pick K vectors randomly from the document
    	seeds = random.sample(sentenceVectors, K)
    	#print("Initial seeds: " + str(seeds))
    	clusters = [[]] * K
    	#print("Empty clusters: " + str(clusters))
    	
    	#clusterCenters = seeds   # initially, centers are the seeds
    	clusterCenters = []
    	for seed in seeds:
    		clusterCenters.append(seed[0])


    	cList = []
    	for seed in seeds:
    		cluster = [seed]
    		cList.append(cluster)
    	#print("CList: " + str(cList))

    	#print("Initial clusters : " + str(cList))
    	dummyVector = [0] * 200
    	oldCenters = [dummyVector] * K
    	error = 1
    	while error > 0:
    		error = calculate_distance(oldCenters, clusterCenters)
    		for i, sentence in enumerate(sentenceVectors):
    			sentenceWithString = sentence
    			sentenceVector = sentence[0]
    			#print("Working on this sentence now: " + str(sentence))
    			minDistance  = float('inf') # BUNA BAK
    			minDistanceCluster = -1
    			#print("Cluster centers are: " + str(clusterCenters))
    			for j, center in enumerate(clusterCenters):
    				distance = calculate_distance(sentenceVector,center)
    				if distance < minDistance:
    					minDistance = distance
    					minDistanceCluster = j

    			#print("Min distance cluster is->  " + str(minDistanceCluster) )
    			if sentenceWithString not in cList[minDistanceCluster]:
    				#print("ENTERED HERE" )
    				for cluster in cList:
    					if sentenceWithString in cluster:
    						cluster.remove(sentenceWithString)
    						break
    				cList[minDistanceCluster].append(sentenceWithString)
    				
    		
    		oldCenters = deepcopy(clusterCenters)
    		#print("Old centers is: " + str(oldCenters))
    		clusterCenters = recalculate_centers(cList, oldCenters)   
    		#print("Cluster centers are updated.")   
    		print("Error is: " + str(error)) 
    		#print("CList is updated as: " + str(cList))
    		#print("Cluster centers are: " + str(clusterCenters))

    	# Pick the most representative sentence from each cluster
    	summary = []
    	summaryWithPositions = []
    	for index, cluster in enumerate(cList):
    		if len(cluster) != 0:
    			representative = pick_most_representative(cluster, clusterCenters[index])
    			summaryWithPositions.append(representative)  
    	
    	print("Summary with positions: " + str(summaryWithPositions))
    	sortedByPositions = sorted(summaryWithPositions, key=itemgetter(1))
    	print("Sorted by positions: " + str(sortedByPositions))
    	for element in sortedByPositions:
    		summary.append(element[0])
    	summaryString = ""
    	for sentence in summary:
    		summaryString = summaryString + sentence + ". "
    	extracted_summaries.append(summaryString)

    	#print(str(extracted_summaries))
    	print(summaryString)
  	
    return extracted_summaries

extracted_summaries1 = cluster(docs_as_vectors, True)   # Model 1
extracted_summaries2 = cluster(docs_as_vectors, False)  # Model 2

gold_summaries = []

#cluster([[12,34,21], [23,34,23], [25,36,25], [100000,10000,100000]])
for filename in glob.glob(os.path.join(GOLD_SUMMARIES_PATH, '*.txt')):
	file = open(filename, 'r')
	summary = file.read()
	gold_summaries.append(summary)


def evaluation(gold_summaries, extracted_summaries):
    validation_scores = []
    rouge = Rouge()

    gold_folds = numpy.array_split(numpy.array(gold_summaries), 10)
    extracted_folds = numpy.array_split(numpy.array(extracted_summaries), 10)

    foldScoresRouge1 = []
    foldScoresRouge2 = []
    foldScoresRougeL = []
    for i, fold in enumerate(extracted_folds):
    	foldScoreR1 = []
    	foldScoreR2 = []
    	foldScoreRL = []
    	gold_fold = gold_folds[i]     # Keep track of chunks, get the correct chunk
    	for j, extracted_summary in enumerate(fold):
    		documentScores = rouge.get_scores(extracted_summary, gold_fold[j])  # Get the correct document
    		foldScoreR1.append(documentScores[0]['rouge-1']['f'])
    		foldScoreR2.append(documentScores[0]['rouge-2']['f'])
    		foldScoreRL.append(documentScores[0]['rouge-l']['f'])
    		
    	R1 = numpy.mean(foldScoreR1)
    	R2 = numpy.mean(foldScoreR2)
    	RL = numpy.mean(foldScoreRL)

    	foldScoresRouge1.append(R1)
    	foldScoresRouge2.append(R2)
    	foldScoresRougeL.append(RL)

    meanR1 = numpy.mean(foldScoresRouge1)
    meanR2 = numpy.mean(foldScoresRouge2)
    meanRL = numpy.mean(foldScoresRougeL)

    stdR1 = numpy.std(foldScoresRouge1)
    stdR2 = numpy.std(foldScoresRouge2)
    stdRL = numpy.std(foldScoresRougeL)


    val_1 = [meanR1 - stdR1, meanR1 + stdR1]
    val_2 = [meanR2 - stdR2, meanR2 + stdR2]
    val_l = [meanRL - stdRL, meanRL + stdRL]

    validation_scores.append(val_1)
    validation_scores.append(val_2)
    validation_scores.append(val_l)

    #print("Validation scores are: " + str(validation_scores))
    return validation_scores


# Split the data into training and test sets
number_of_docs = len(gold_summaries) 
test_set_size = int(number_of_docs / 5)
train_set_size = number_of_docs - test_set_size

trainingIndices = random.sample(range(number_of_docs), train_set_size)
gold_train_set = []
gold_test_set = []

extracted_train_set1 = []
extracted_test_set1 = []

extracted_train_set2 = []
extracted_test_set2 = []

for i, summary in enumerate(gold_summaries):
	if i in trainingIndices:
		gold_train_set.append(summary)
	else:
		gold_test_set.append(summary)


for i, summary in enumerate(extracted_summaries1):
	if i in trainingIndices:
		extracted_train_set1.append(summary)
	else:
		extracted_test_set1.append(summary)

for i, summary in enumerate(extracted_summaries2):
	if i in trainingIndices:
		extracted_train_set2.append(summary)
	else:
		extracted_test_set2.append(summary)


# Pass the training sets to get validation scores
model1_validation_scores = evaluation(gold_train_set, extracted_train_set1)  # Evaluation for model 1
model2_validation_scores = evaluation(gold_train_set, extracted_train_set2)  # Evaluation for model 2

# Pass the test sets to get test scores
model1_test_scores = evaluation(gold_test_set, extracted_test_set1)  # Evaluation for model 1
model2_test_scores = evaluation(gold_test_set, extracted_test_set2)  # Evaluation for model 2


print("RESULTS FOR MODEL 1:")
print("")
print("Validation Scores: ")
for i, element in enumerate(model1_validation_scores):
	if i == 0:
		print("Rouge 1: " + str(element))
	elif i == 1: 
		print("Rouge 2: " + str(element))
	else:
		print("Rouge L: " + str(element))

print("")
print("Test Scores: ")
for i, element in enumerate(model1_test_scores):
	if i == 0:
		print("Rouge 1: " + str(element))
	elif i == 1: 
		print("Rouge 2: " + str(element))
	else:
		print("Rouge L: " + str(element))

print("")
print("RESULTS FOR MODEL 2:")
print("")
print("Validation Scores: ")
for i, element in enumerate(model2_validation_scores):
	if i == 0:
		print("Rouge 1: " + str(element))
	elif i == 1: 
		print("Rouge 2: " + str(element))
	else:
		print("Rouge L: " + str(element))

print("")
print("Test Scores: ")
for i, element in enumerate(model2_test_scores):
	if i == 0:
		print("Rouge 1: " + str(element))
	elif i == 1: 
		print("Rouge 2: " + str(element))
	else:
		print("Rouge L: " + str(element))


#model1scores = evaluation(gold_summaries, extracted_summaries1)  # Evaluation for model 1
#model2scores = evaluation(gold_summaries, extracted_summaries2)  # Evaluation for model 2



