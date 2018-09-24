import argparse
import pandas as pd
import numpy as np
import operator
import nltk as nl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import ParameterGrid
import statistics
import random


from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

##########################################################################################

#### FUNCTIONS DIRECTLY COPIED FROM TOPICS (AND HENCE TO BE ABSTRACTED TO ANOTHER FILE):

##########################################################################################

def getInputDataAndDisplayStats(filename,processDate,printSummary=False):

	df=pd.read_csv(filename)

	df=df.drop_duplicates('content')
	df=df[~df['content'].isnull()]

	# There are a large number of junk articles, many of which either don't make sense or
	# just contain a headline - as such they are useless for this analysis and may distort
	# results if left in place
	df=df[df['content'].str.len()>=200]

	# Find and remove summary NYT "briefing" articles to avoid confusing the clustering
	targetString="(Want to get this briefing by email?"
	df['NYT summary']=df['content'].map(lambda d: d[:len(targetString)]==targetString)
	df=df[df['NYT summary']==False]

	# The following removes a warning that appears in many of the Atlantic articles.
	# Since it is commonly at the beginning, it brings a lot of noise to the search for similar articles
	# And subsequently to the assessment of sentiment
	targetString="For us to continue writing great stories, we need to display ads.             Please select the extension that is blocking ads.     Please follow the steps below"
	df['content']=df['content'].str.replace(targetString,'')

	# This is also for some Atlantic articles for the same reasons as above
	targetString="This article is part of a feature we also send out via email as The Atlantic Daily, a newsletter with stories, ideas, and images from The Atlantic, written specially for subscribers. To sign up, please enter your email address in the field provided here."
	df=df[df['content'].str.contains(targetString)==False]

	# This is also for some Atlantic articles for the same reasons as above
	targetString="This article is part of a feature we also send out via email as Politics  Policy Daily, a daily roundup of events and ideas in American politics written specially for newsletter subscribers. To sign up, please enter your email address in the field provided here."
	df=df[df['content'].str.contains(targetString)==False]

	# More Atlantic-specific removals (for daily summaries with multiple stories contained)
	df=df[df['content'].str.contains("To sign up, please enter your email address in the field")==False]

	# Remove daily CNN summary
	targetString="CNN Student News"
	df=df[df['content'].str.contains(targetString)==False]

	if printSummary:
		print("\nArticle counts by publisher:")
		print(df['publication'].value_counts())

		print("\nArticle counts by date:")
		print(df['date'].value_counts())
		
	# Restrict to articles on the provided input date.
	# This date is considered mandatory for topic clustering but is not required for sentiment
	# since sentiment only processes a specified list of articles.
	# For topic clustering it is essential to have the date as it is
	# enormously significant in article matching.
	if processDate!=None:
		df=df[df['date']==processDate]
	df.reset_index(inplace=True, drop=True)

	# Remove non-ASCII characters
	df['content no nonascii']=df['content'].map(lambda x: removeNonASCIICharacters(x))

	print("\nFinal dataset:\n\nDate:",processDate,"\n")
	print(df['publication'].value_counts())

	return df
	
##########################################################################################

def removeNonASCIICharacters(textString): 
    return "".join(i for i in textString if ord(i)<128)
    
##########################################################################################
    
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

##########################################################################################

def setupStoryMapAndReportList(args=None,reportArticleList=None,storyMapFileName=None):
	# Story Map is used in fitting if grid search is applied (As ground truth)
	# It is also used in graph if no threshold provided (to determine colours, not to determine location)
	# Report Article List is used at the end to create a report with, for each
	# article in the list, the set of articles within tolerance, and the key words for each
	if args==None:
		articleList=reportArticleList
		fileName=storyMapFileName
	else:
		articleList=args.article_id_list
		fileName=args.story_map_validation

	reportArticleList=articleList
	if fileName!=None:
		storyMap=readStoryMapFromFile(fileName)
		if reportArticleList==None:
			reportArticleList=[]
			for story, articleList in storyMap.items():
				reportArticleList.append(articleList[0])
	else:
		storyMap=None
	return storyMap,reportArticleList

##########################################################################################

def readStoryMapFromFile(filename):
	return readDictFromCsvFile(filename,'StoryMap')

##########################################################################################

def readGridParameterRangeFromFile(filename):
	return readDictFromCsvFile(filename,'GridParameters')

##########################################################################################

def readDictFromCsvFile(filename,schema):
	gridParamDict={}
	with open(filename, 'r') as f:
		for row in f:
			row=row[:-1] # Exclude the carriage return
			row=row.split(",")
			key=row[0]
			vals=row[1:]

			if schema=='GridParameters':
				if key in ['story_threshold','tfidf_maxdf']:
					finalVals=list(float(n) for n in vals)
				elif key in ['ngram_max','tfidf_mindf','max_length','sentiment_sentences']:
					finalVals=list(int(n) for n in vals)
				elif key in ['lemma_conversion','tfidf_binary']:
					finalVals=list(str2bool(n) for n in vals)
				elif key in ['parts_of_speech']:
					listlist=[]
					for v in vals:
						listlist.append(v.split("+"))
					finalVals=listlist
				elif key in ['tfidf_norm','nlp_library','sentiment_library']:
					finalVals=vals
				else:
					print(key)
					print("KEY ERROR")
					return
			elif schema=='StoryMap':
				finalVals=list(int(n) for n in vals)
			else:
				print(schema)
				print("SCHEMA ERROR")
				return
			
			gridParamDict[key]=finalVals
	return gridParamDict

##########################################################################################

#### END OF FUNCTIONS DIRECTLY COPIED FROM TOPICS (AND HENCE TO BE ABSTRACTED TO ANOTHER FILE):

##########################################################################################

class SentimentAnalyser():

	scaleMin=-1.
	scaleMax=1.

    # Initializer / Instance attributes
	def __init__(self, library):
		if library=='google':
			self.analyser=GoogleSentimentAnalyser()
		elif library=='stanford':
			self.analyser=StanfordSentimentAnalyser()
		elif library=='vader':
			self.analyser=NLTKVaderSentimentAnalyser()
		else:
			print("ERROR - NO RECOGNISED LIBRARY")

	def getOverallArticleScore(self,articleResults):

		# Google returns a document score, but it is an int, which is useful when comparing documents
		# Hence computing the average of the sentences here instead
		# Google's document score is here: articleResults.document_sentiment.score
		numSentences=0.
		totalSentScore=0.
		for sentence in articleResults:
			numSentences+=1
			totalSentScore+=self.analyser.getSentenceScoreFromResults(sentence)

		value=(totalSentScore/numSentences-self.analyser.scaleMin)/(self.analyser.scaleMax-self.analyser.scaleMin)
		return self.scaleMin+value*(self.scaleMax-self.scaleMin)

	def generateResults(self,textToAnalyse):
		return self.analyser.generateResults(textToAnalyse)

##########################################################################################

class GoogleSentimentAnalyser():

	scaleMin=-1.
	scaleMax=1.

	def __init__(self):
		self.client=language.LanguageServiceClient()
		return

	def generateResults(self,textToAnalyse):
		document=types.Document(
								content=textToAnalyse,
								type=enums.Document.Type.PLAIN_TEXT
								)
		return self.client.analyze_sentiment(document=document).sentences

	def getSentenceScoreFromResults(self,sentenceResults):
		return sentenceResults.sentiment.score
		
##########################################################################################

class StanfordSentimentAnalyser():

	scaleMin=0.
	scaleMax=4.

	def __init__(self):
		from pycorenlp import StanfordCoreNLP
		self.nlp=StanfordCoreNLP('http://localhost:9000')
		return

	def generateResults(self,textToAnalyse):
		return self.nlp.annotate(textToAnalyse,
								properties={
											 'annotators': 'sentiment',
											 'outputFormat': 'json',
											 'timeout': 100000,  # NB The original example had 1000 and that caused time-out errors
											})["sentences"]

	def getSentenceScoreFromResults(self,sentenceResults):
		return int(sentenceResults["sentimentValue"])
		
##########################################################################################

class NLTKVaderSentimentAnalyser():
# Per NLTK Vader user guide: https://pypi.org/project/vaderSentiment/
# Typical threshold values (used in the literature cited on this page) are: 
#. **positive sentiment**: ``compound`` score >= 0.05 
#. **neutral sentiment**: (``compound`` score > -0.05) and (``compound`` score < 0.05) 
#. **negative sentiment**: ``compound`` score <= -0.05 

	scaleMin=-1.
	scaleMax=1.

	def __init__(self):
		self.nltkVaderAnalyser=SentimentIntensityAnalyzer()
		return

	def generateResults(self,textToAnalyse):
		ss=[]
		for sentence in nl.sent_tokenize(textToAnalyse):
			ss.append(self.nltkVaderAnalyser.polarity_scores(sentence))
		return ss

	def getSentenceScoreFromResults(self,sentenceResults):
		return sentenceResults['compound']

##########################################################################################

def collapseRequestedArticleListIntoStoryList(requestedArticleList,storyMap):
	# Check that the explicitly requested articles are all contained in the storyList
	# If they aren't, add a new story to contain them

	# If the storyMap was empty, it will be None,
	# so initialise as a dictionary ready for adding new values
	if storyMap==None:
		newStoryMap={}
	else:
		newStoryMap=storyMap.copy()

	# If there are no articles to collapse in, just return the copy
	# It's probably best to do it this way in case there is an empty list
	# in one grid iteration and not in others - which would mean the cacheing would
	# otherwise cause problems.
	if requestedArticleList!=None:
		found=False
		for story,articleListFromMap in newStoryMap.items():
			if len(articleListFromMap)==len(requestedArticleList):
				y=sum([x in articleListFromMap for x in requestedArticleList])
				if y==len(articleListFromMap):
					found=True

		# If there is no complete story exactly matching then add a new story to the list
		# With the first article ID as the key (arbitrarily)
		if not found:
			newStoryMap[requestedArticleList[0]]=requestedArticleList
	return newStoryMap
	
##########################################################################################

def computePopulationBalanceScore(articleScoreDict,sentimentClass):
	# Extract values from dict, then normalise to be within -1 to +1
	# Then compute population standard deviation as the balance score
	population=[-1.+(x-sentimentClass.scaleMin)/(sentimentClass.scaleMax-sentimentClass.scaleMin)*(1.-(-1.)) for x in articleScoreDict.values()]
	return statistics.pstdev(population)

##########################################################################################

def computePopulationBalanceScoreHistoMean(articleScoreDict,sentimentClass):
	# Extract values from dict, then normalise to be within -1 to +1
	# Then compute population standard deviation as part of the balance score
	numBuckets=len(articleScoreDict)
	articleValues=pd.Series(articleScoreDict)
	
	# Based on 10,000 random article samples, Google's sentiment score for these articles lies within +/- 0.86
	# So, scale all scores by dividing by that value to rescale to +/- 1.00 before computing balance score
	# Ideally this should factored in at the individual NLP library class level 
	articleValues=articleValues/0.86

	populatedBuckets=0
	for i in range(numBuckets):
		bucketFrom=sentimentClass.scaleMin+i*(sentimentClass.scaleMax-sentimentClass.scaleMin)/numBuckets
		bucketTo=bucketFrom+(sentimentClass.scaleMax-sentimentClass.scaleMin)/numBuckets
		# The following is to ensure the top of the highest bucket is counted somewhere
		# and doesn't fall out due to treatment of inequalities in ranges
		if bucketTo==sentimentClass.scaleMax:
			bucketTo+=0.001
		numSamples=((bucketFrom<=articleValues) & (articleValues<bucketTo)).sum()
		if numSamples>0:
			populatedBuckets+=1

	# Score computed as proportion of buckets which are populated (more buckets implies a more balanced view)
	# This has a value between 0 and 1.
	# This is in turn multiplied by the distance between the mean and 1.
	# So, if mean is in center (i.e. at 0) then things are balanced, so score is not decreased
	# Otherwise, score is decreased proportionately
	return (populatedBuckets/numBuckets * (1.-abs(articleValues.mean())))

##########################################################################################

def createRunParamsDict(args):

	if args.grid_parameter_file!=None:
		runParams=readGridParameterRangeFromFile(args.grid_parameter_file)
	else:
		# Threshold deliberately not included here...
		# Its purpose is to determine scoring for evaluating best parameters in
		# a grid search - and its absence is relied upon to avoid that
		runParams={'sentiment_library':[args.sentiment_library],
				   'sentiment_sentences':[args.sentiment_sentences]}

	# The following are constants across all runs (if a grid is requested)
	# They will be placed into the runParams dict so that there is a single
	# point of interface, which will make porting code to a notebook easier
	# i.e. in a notebook, just set up the dict, as only that is referenced subsequently
	runParams['input_file']=[args.input_file]
	runParams['article_stats']=[args.article_stats]
	runParams['article_id_list']=[args.article_id_list]

	# Use parameter grid even if there is only set of parameters
	return runParams
	
##########################################################################################

def main(args):

	# Load story map and article list for validation, if relevant
	storyMap,reportArticleList=setupStoryMapAndReportList(args=args)

	# Create parameters dict from input args, loading from input file as required/specified
	runParams=createRunParamsDict(args)

	# Use parameter grid even if there is only set of parameters
	parameterGrid=ParameterGrid(runParams)

	# Load the corpus of articles from file
	articleDataFrame=getInputDataAndDisplayStats(args.input_file,None,args.article_stats)

	# Loop through each set of parameters and perform the main algorithm on each set
	for i,currentParams in enumerate(parameterGrid):
		if len(parameterGrid)>1:
			print("Combination:",i+1,"of",len(parameterGrid))
			print(currentParams)

		# The base storyMap may be appended with an iteration specific
		# article list, depending on the request details in the parameter grid
		iterationStoryMap=collapseRequestedArticleListIntoStoryList(currentParams['article_id_list'],
																	storyMap)

		sentimentAnalyser=SentimentAnalyser(currentParams['sentiment_library'])

		for story,articleList in iterationStoryMap.items():
			articleSentScores={}
			print("ANALYSING STORY:",story,"using",currentParams['sentiment_library'])
			print("Number of articles in story:",len(articleList))
			for article in articleList:

				articleContent=articleDataFrame[articleDataFrame['id']==article].iloc[0]['content']

				# if requested, only use the first few sentences for analysis
				if currentParams['sentiment_sentences']!=None:
					articleSentences=nl.sent_tokenize(articleContent)
					textToAnalyse=' '.join(articleSentences[:currentParams['sentiment_sentences']])	
				else:
					textToAnalyse=articleContent

				results=sentimentAnalyser.generateResults(textToAnalyse)

				articleSentScores[article]=sentimentAnalyser.getOverallArticleScore(results)

			# Sort and display results
			sortedArticleSentScores=sorted(articleSentScores.items(), key=operator.itemgetter(1))
			print("\nArticle sentiments, most positive first:")
			for article in reversed(sortedArticleSentScores):
				print(article[0],":", round(article[1],3),articleDataFrame[articleDataFrame['id']==article[0]].iloc[0]['publication'])

			# This only works because each article's score is constrained to be in -1 to +1
			# So maximum possible population standard deviation is 1 and minimum is 0
			# Should arguably build this into the classes somewhere, but so far I don't have any
			# classes that pertain to the population, rather than to an individual article
			# I could do the calculation here based on the class min and max... (and building a function to consume the list) 
			print("\nBALANCE SCORE:",round(computePopulationBalanceScoreHistoMean(articleSentScores,SentimentAnalyser),3)*100.,"\n")
		
	return

##########################################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Provide arguments for topic mining')

	parser.add_argument('--sentiment-library', help='library used for sentiment analysis', default='stanford')

	parser.add_argument('--input-file', help='file containing news articles to process', default='./data/FullNewsCorpus/articles123ApostSyn.csv')
	parser.add_argument('--article-id-list', help='list of article IDs to rank by sentiment',nargs='+',type=int)
	parser.add_argument('--sentiment-sentences', help='number of sentences of article to analyse for sentiment',type=int)
	
	parser.add_argument('--article-stats', help='print number of available articles by date and publication', type=str2bool, default=False) 
	parser.add_argument('--grid-parameter-file', help='parameter ranges for grid search')
	parser.add_argument('--story-map-validation', help='file containing links between articles for validation')

	args = parser.parse_args()

	main(args)

