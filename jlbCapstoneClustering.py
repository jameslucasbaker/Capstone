import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
import csv


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

def removeSpacesAndPunctuation(textString): 
    return "".join(i for i in textString if (ord(i)>=48 and ord(i)<=57) or (ord(i)>=97 and ord(i)<=122))

##########################################################################################

def stringSpaCyProcess(nlp,stringToConvert,partsOfSpeech,maxWords,stop_words,lemmatize):
	doc=nlp(stringToConvert)
	if partsOfSpeech==None:
		spacyTokens=[w for w in doc]
	else:
		spacyTokens=[w for w in doc if w.pos_ in partsOfSpeech]

	str=[]
	for spt in spacyTokens:
		if lemmatize:
			wrd=spt.lemma_
		else:
			wrd=spt.text
		wrdlower=removeSpacesAndPunctuation(wrd.lower())
		# The middle term below is correctly wrd.lower() not wrdlower since the function call
		# above strips out the --, and I don't want to compare with 'pron' in case that
		# finds false matches
		if wrdlower not in stop_words and wrd.lower()!='-pron-' and not wrdlower=='':
			if maxWords==None or len(str)<maxWords:
				str.append(wrdlower)
		if maxWords!=None and len(str)==maxWords:
				return ' '.join(str)		
	return ' '.join(str)

##########################################################################################

def stringNLTKProcess(nl,stringToConvert,partsOfSpeech,stop_words,maxWords=None,lemmatizer=None):
	sentences=nl.sent_tokenize(stringToConvert)
	str=[]
	for sentence in sentences:
		wordString=[]
		for word,pos in nl.pos_tag(nl.word_tokenize(sentence)):
			# The following condition avoids any POS which corresponds to punctuation (and takes all others)
			if partsOfSpeech==None:
				if pos[0]>='A' and pos[0]<='Z':
					wordString.append(word)
			elif pos in partsOfSpeech:
				wordString.append(word)
		for wrd in wordString:
			wrdlower=wrd.lower()
			if wrdlower not in stop_words and wrdlower!="'s":
				if maxWords==None or len(str)<maxWords:
					if lemmatizer==None:
						str.append(wrdlower)
					else:
						str.append(lemmatizer.lemmatize(wrd.lower(), pos='v'))
			if maxWords!=None and len(str)==maxWords:
				return ' '.join(str)
	return ' '.join(str)

##########################################################################################

def loadStopWords(stopWordsFileName):
	stop_words=[]
	f=open(stopWordsFileName, 'r')
	for l in f.readlines():
		stop_words.append(l.replace('\n', ''))
	return stop_words

##########################################################################################

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

##########################################################################################

def printTopNwordsForArticle(tfidfVectors,terms,articleNum,n):
	vect=tfidfVectors[articleNum].toarray()[0]
	topn1=np.argsort(vect)
	for t in reversed(topn1[-n:]):
		if vect[t]>0.001:
			print(terms[t],":",round(vect[t],5))

##########################################################################################

def initialiseAllNonZeroCoords(tfidfVectors):
# This function just exists since it seems to be expensive and I'd rather not call it multiple times
# Hence it is intended to be called outside of loops in order to simplify the row specific processing
	values=[]
	nzc=zip(*tfidfVectors.nonzero())

	# In Python 3 the zip can only be iterated through one time before it is automatically released
	# So need to copy the results otherwise the main loop below will no longer work
	pointList=[]
	for i,j in nzc:
		pointList.append([i,j])		

	for row in range(tfidfVectors.shape[0]):
		rowList=[]
		for i,j in pointList:
			if row==i:
				rowList.append(j)
		values.append(rowList)

	return values

##########################################################################################

def productRelatednessScores(tfidfVectors,nonZeroCoords,refRow):
# NB TF-IDF Needs to be non-normalised else this will give completely meaningless results
	scores=[0.]*tfidfVectors.shape[0]

	for toRow in range(tfidfVectors.shape[0]):
		scores[toRow]=sum([(float(tfidfVectors[toRow,w])*float(tfidfVectors[refRow,w])) for w in nonZeroCoords[refRow] if w in nonZeroCoords[toRow]])
	return scores

##########################################################################################

def mergeInputFiles():
	a=pd.read_csv("./data/FullNewsCorpus/articles1.csv")
	b=pd.read_csv("./data/FullNewsCorpus/articles2.csv")
	c=pd.read_csv("./data/FullNewsCorpus/articles3.csv")

	merged=pd.concat([a,b,c])
	merged.to_csv("./data/FullNewsCorpus/articles123.csv", index=False)

##########################################################################################

def correctApostrophesAndResave():
	# Read in the file
	with open('./data/FullNewsCorpus/articles123.csv', 'r') as file :
		filedata=file.read()

	# Replace the target string
#	filedata=filedata.replace("NEEDS TO BE ANGLED APOSTROPHE", "'")

	# Write the file out again
	with open('./data/FullNewsCorpus/articles123Apost.csv', 'w') as file:
		file.write(filedata)

##########################################################################################
  
def applySynonymChangesToFile():
	synonyms=pd.read_csv('./data/TranslateAtBeginning.csv')

	# Read in the file
	with open('./data/FullNewsCorpus/articles123Apost.csv', 'r') as file :
		filedata=file.read()

	for index, synoynm in synonyms.iterrows():
		# Replace the target string
		filedata=filedata.replace(synonyms['From'][index],synonyms['To'][index])
    
	# Write the file out again
	with open('./data/FullNewsCorpus/articles123ApostSyn.csv', 'w') as file:
		file.write(filedata)

##########################################################################################

def preprocessAndVectorize(articleDataFrame,args,pos_nlp_mapping,nlp,nl,wordnet_lemmatizer,stop_words):
	# Map the input parts of speech list to the coding required for the specific NLP library
	if args['parts_of_speech'][0]!='ALL':
		partsOfSpeech=[]
		for pos in args['parts_of_speech']:
			partsOfSpeech.append(pos_nlp_mapping[args['nlp_library']][pos])
		partsOfSpeech=[item for sublist in partsOfSpeech for item in sublist]
	else:
		partsOfSpeech=None

	# Processing of text depends on NLP library choice
	if args['nlp_library']=='spaCy':
		articleDataFrame['input to vectorizer']=articleDataFrame['content no nonascii'].map(lambda x: stringSpaCyProcess(nlp,
																									   x,
																									   partsOfSpeech=partsOfSpeech,
																									   maxWords=args['max_length'],
																									   stop_words=stop_words,
																									   lemmatize=args['lemma_conversion']))
	elif args['nlp_library']=='nltk':
		articleDataFrame['input to vectorizer']=articleDataFrame['content no nonascii'].map(lambda x: stringNLTKProcess(nl,
																									  x,
																									  partsOfSpeech=partsOfSpeech,
																									  stop_words=stop_words,
																									  maxWords=args['max_length'],
																									  lemmatizer=wordnet_lemmatizer))
	else:
		print("PROBLEM... NO VALID NLP LIBRARY... MUST BE nltk OR spaCy")

	# To get default values a couple of parameters need to be not passed if not specified on the command line
	# Passing as None behaves differently to passing no parameter (which would invoke the default value)
	optArgsForVectorizer={}
	if args['tfidf_maxdf'] != None:
		optArgsForVectorizer['max_df']=args['tfidf_maxdf']
	if args['tfidf_mindf'] != None:
		optArgsForVectorizer['min_df']=args['tfidf_mindf']

	# Create and run the vectorizer
	vectorizer=TfidfVectorizer(analyzer='word',
   	    	                   ngram_range=(1,args['ngram_max']),
       	    	               lowercase=True,
           	    	    	   binary=args['tfidf_binary'],
               		    	   norm=args['tfidf_norm'],
							   **optArgsForVectorizer)
	tfidfVectors=vectorizer.fit_transform(articleDataFrame['input to vectorizer'])
	terms=vectorizer.get_feature_names()
	return tfidfVectors, terms

##########################################################################################

def scoreCurrentParamGuess(tfidfVectors,storyMap,articleDataFrame,threshold,printErrors=False):
	# Work with distances relative to first item in each cluster - even though this is clearly arbitrary since that
	# point could be an outlier in the cluster and hence might cause problems.
	# But I have to start somewhere - and can refine it later if needed.

	nonZeroCoords=initialiseAllNonZeroCoords(tfidfVectors)
	score=0
	outGood=0
	outBad=0
	inGood=0
	inBad=0
	for story, storyArticles in storyMap.items():
		leadArticleIndex=articleDataFrame[articleDataFrame['id']==storyArticles[0]].index[0]
		# Compute score of all articles in corpus relative to first article in story (.product)
		# Then count through list relative to threshold (add one for a good result, subtract one for a bad result)
		scores=productRelatednessScores(tfidfVectors,nonZeroCoords,leadArticleIndex)
		rankedIndices=np.argsort(scores)
		foundRelatedArticles=[]
		# THE SORTING HERE IS NOT STRICTLY REQUIRED, BUT I COULD USE IT SO THAT ONCE THE THRESHOLD IS PASSED
		# IN THE LOOP, THEN I INFER THE REMAINING RESULTS
		for article in reversed(rankedIndices):
			thisArticleIndex=articleDataFrame['id'][article]
			if thisArticleIndex in storyArticles:
				if scores[article]>=threshold:
					score+=1
					inGood+=1
				else:
					score-=1
					inBad+=1
					if printErrors:
						print("ERROR:",thisArticleIndex,"should be in",story)
			else: # article not supposed to be in range
				if scores[article]<=threshold:
					score+=1
					outGood+=1
				else:
					score-=1
					outBad+=1
					if printErrors:
						print("ERROR:",thisArticleIndex,"should NOT be in",story)
	scoreDict={'score':score,'inGood':inGood,'inBad':inBad,'outGood':outGood,'outBad':outBad}
	return scoreDict

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
				elif key in ['ngram_max','tfidf_mindf','max_length']:
					finalVals=list(int(n) for n in vals)
				elif key in ['lemma_conversion','tfidf_binary']:
					finalVals=list(str2bool(n) for n in vals)
				elif key in ['parts_of_speech']:
					listlist=[]
					for v in vals:
						listlist.append(v.split("+"))
					finalVals=listlist
				elif key in ['tfidf_norm','nlp_library']:
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

# Reduce vector space to two dimensions
# Then produce Bokeh graph
def graphVectorSpace(tfidfVectors,extraColumns,dateForTitle,storyMap,threshold):
	# Better results seem to be obtained by breaking the dimensionality reduction into two steps

	# First reduce to fifty dimensions with SVD
	from sklearn.decomposition import TruncatedSVD
	svd=TruncatedSVD(n_components=50, random_state=0)
	svdResults=svd.fit_transform(tfidfVectors)

	# Next continue to two dimensions with TSNE
	from sklearn.manifold import TSNE
	tsneModel=TSNE(n_components=2, verbose=1, random_state=0, n_iter=500)
	tsneResults=tsneModel.fit_transform(svdResults)
	tfidf2dDataFrame=pd.DataFrame(tsneResults)
	tfidf2dDataFrame.columns=['x','y']

	tfidf2dDataFrame['publication']=extraColumns['publication']	
	tfidf2dDataFrame['id']=extraColumns['id']	
	tfidf2dDataFrame['content']=extraColumns['content no nonascii'].map(lambda x: x[:200])

	# All articles will be marked as NA to indicate that they have not been assigned to a story
	# Then those which have been assigned one will be updated to refer to that
	tfidf2dDataFrame['category']='NA'

	# If the threshold is not provided, then just graph the vector space as is
	# With colours indicating desired story grouping
	# This still has value because it shows how well stories cluster together
	if threshold==None:
		graphTitle=("TF-IDF article clustering - story assignment from map - "+dateForTitle[0])
		for story, storyArticles in storyMap.items():
			for article in storyArticles:
				if len(tfidf2dDataFrame[tfidf2dDataFrame['id']==article].index)==1:
					i=tfidf2dDataFrame[tfidf2dDataFrame['id']==article].index[0]
					tfidf2dDataFrame['category'][i]=story
	else:
		graphTitle=("TF-IDF article clustering - story assignment computed - "+dateForTitle[0])
		nonZeroCoords=initialiseAllNonZeroCoords(tfidfVectors)
		for story, storyArticles in storyMap.items():
			leadArticleIndex=extraColumns[extraColumns['id']==storyArticles[0]].index[0]
			# Compute score of all articles in corpus relative to first article in story (.product)
			scores=productRelatednessScores(tfidfVectors,nonZeroCoords,leadArticleIndex)
			rankedIndices=np.argsort(scores)
			for article in rankedIndices:
				if scores[article]>=threshold:
					tfidf2dDataFrame['category'][article]=story

	import bokeh.plotting as bp
	from bokeh.models import HoverTool
	from bokeh.plotting import show
	from bokeh.palettes import d3
	import bokeh.models as bmo

	plot_tfidf=bp.figure(plot_width=700, plot_height=600, title=graphTitle,
						 tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
						 x_axis_type=None, y_axis_type=None, min_border=1)

	numCats=len(tfidf2dDataFrame['category'].unique())
	palette=d3['Category20'][numCats]
	color_map=bmo.CategoricalColorMapper(factors=tfidf2dDataFrame['category'].map(str).unique(), palette=palette)

	plot_tfidf.scatter(x='x', y='y', color={'field': 'category', 'transform': color_map}, 
						legend='category',source=tfidf2dDataFrame)
	hover=plot_tfidf.select(dict(type=HoverTool))
	plot_tfidf.legend.click_policy="hide"
	hover.tooltips={"id": "@id", "publication": "@publication", "content":"@content", "category":"@category"}

	show(plot_tfidf)

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
		articleList=args['article_id_list']
		fileName=args['story_map_validation']

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

def produceRequestedReportDetails(tfidfVectors,articleDataFrame,reportArticleList,threshold,storyMap,terms):

	# tfidfVectors is a sparse matrix, for efficiency it's useful to determine once only which
	# coordinates have non-zero values
	nonZeroCoords=initialiseAllNonZeroCoords(tfidfVectors)

	topNwords=25

	# Create list of articles to process
	# If a list is provided in command line arguments, use that
	storyMapGood=0.0
	encounteredStoriesList=[]
	for index,row in articleDataFrame.iterrows():
		if row['id'] in reportArticleList:
			ref_index=index
			print("-----")
			print("-----")
			print("LEAD ARTICLE IN STORY:",row['id'])
			print("-----")

			if threshold==None:
				articleIndexList=[index]
			else:
				# Score and rank all articles relative to this one
				# Count number of items that are greater than or equal to threshold
				# Then truncate the list beyond those items
				scores=productRelatednessScores(tfidfVectors,nonZeroCoords,ref_index)
				rankedIndices=np.argsort(scores)
				numItemsInRange=sum(x>=threshold for x in scores)
				articleIndexList=rankedIndices[-numItemsInRange:]

			# If there is a story map, find out which story this article is meant to belong to
			targetStory=None
			if storyMap!=None:
				for story,articleList in storyMap.items():
					if row['id'] in articleList:
						targetStory=story
						targetArticleList=articleList
						encounteredStoriesList.append(targetStory)
					
			# For just those articles that are within threshold of the lead article
			# Print out the key terms and their tf-idf scores
			# Then count the number of articles that are correctly assigned to the story
			# (if there is a ground truth storyMap provided)
			for article in reversed(articleIndexList):
				if targetStory!=None:
					# If this is officially part of the same story, update the counts
					if articleDataFrame['id'][article] in targetArticleList:
						storyMapGood+=1.0

				print("MEMBER ARTICLE:",articleDataFrame['id'][article])
				if threshold!=None:
					print("Score :",scores[article])
				print(articleDataFrame['publication'][article])
				print(articleDataFrame['content'][article][:500])
				print("PASSED TO VECTORIZER AS:")
				print(articleDataFrame['input to vectorizer'][article])
				print()
				printTopNwordsForArticle(tfidfVectors,terms,articleNum=article,n=topNwords)
				print("-----")
			print("-----")

	# If there is a storyMap, print out the percentage results for the inferred allocation
	# Note that it should be just relative to the number of stories actually encountered
	# So if the user requests a specific set of articles and those articles don't cover
	# the full set of stories, then they shouldn't be counted as errors.
	if storyMap!=None:
		storyMapSize=sum([len(storyMap[story]) for story in encounteredStoriesList])
		print("\n\nPERCENTAGE OF STORIES ALLOCATED IN LINE WITH MAP:",100.*float(storyMapGood)/float(storyMapSize))

	return

##########################################################################################

def createRunParamsDict(args):

	if args['grid_parameter_file']!=None:
		runParams=readGridParameterRangeFromFile(args['grid_parameter_file'])
	else:
		# Threshold deliberately not included here...
		# Its purpose is to determine scoring for evaluating best parameters in
		# a grid search - and its absence is relied upon to avoid that
		runParams={'ngram_max':[args['ngram_max']],
				   'tfidf_maxdf':[args['tfidf_maxdf']],
				   'tfidf_mindf':[args['tfidf_mindf']],
				   'max_length':[args['max_length']],
				   'parts_of_speech':[args['pos_list']],
				   'lemma_conversion':[args['lemma_conversion']],
				   'tfidf_binary':[args['tfidf_binary']],
				   'tfidf_norm':[args['tfidf_norm']],
				   'nlp_library':[args['nlp_library']],
				   'story_threshold':[args['story_threshold']]}
	# The following are constants across all runs (if a grid is requested)
	# They will be placed into the runParams dict so that there is a single
	# point of interface, which will make porting code to a notebook easier
	# i.e. in a notebook, just set up the dict, as only that is referenced subsequently
	runParams['input_file']=[args['input_file']]
	runParams['article_stats']=[args['article_stats']]
	runParams['process_date']=[args['process_date']]
	runParams['stop_words_file']=[args['stop_words_file']]
	runParams['display_graph']=[args['display_graph']]
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


	# Load and initialise required NLP libraries
	pos_nlp_mapping={}
	nl=None
	wordnet_lemmatizer=None
	nlp=None
	if 'spaCy' in runParams['nlp_library']:
		import spacy
		nlp=spacy.load('en')
		pos_nlp_mapping['spaCy']={'VERB':['VERB'],'PROPER':['PROPN'],'COMMON':['NOUN']}

	if 'nltk' in runParams['nlp_library']:
		import nltk as nl
		if True in runParams['lemma_conversion']:
			from nltk.stem import WordNetLemmatizer
			wordnet_lemmatizer=WordNetLemmatizer()
		else:
			wordnet_lemmatizer=None
		pos_nlp_mapping['nltk']={'VERB':['VB','VBD','VBG','VBN','VBP','VBZ'],'PROPER':['NNP','NNPS'],'COMMON':['NN','NNS']}


	# Load corpus of articles from file
	# 0 index is required because the parameters are forced to be lists by ParameterGrid
	articleDataFrame=getInputDataAndDisplayStats(runParams['input_file'][0],
												 runParams['process_date'][0],
												 runParams['article_stats'][0])


	# Load stop words now - these will be deleted from final text by processor before vectorizing
	# 0 index is required because the parameters are forced to be lists by ParameterGrid
	stop_words=loadStopWords(runParams['stop_words_file'][0])


	# Loop across all parameter combinations in grid to determine best set
	# If not doing grid search, will just pass through the loop once
	bestParamScoreDict={'score':-1000000}
	bestParams=parameterGrid[0]
	for i,currentParams in enumerate(parameterGrid):
		if len(parameterGrid)>1:
			print("Combination:",i+1,"of",len(parameterGrid))
			print(currentParams)

		# Determine tf-idf vectors
		# terms is just used later on if analysis of final results is requested
		tfidfVectors,terms=preprocessAndVectorize(articleDataFrame,
												  currentParams,
												  pos_nlp_mapping,
												  nlp,
												  nl,
												  wordnet_lemmatizer,
												  stop_words)

		# Compute scores if threshold provided (meaning as part of grid search)
		if 'story_threshold' in currentParams and currentParams['story_threshold']!=None:
			scoreDict=scoreCurrentParamGuess(tfidfVectors,storyMap,articleDataFrame,currentParams['story_threshold'])
			print(scoreDict)

			# Update best so far
			if scoreDict['score']>=bestParamScoreDict['score']:
				if len(parameterGrid)>1:
					print(i+1,"is the best so far!")
				bestParams=currentParams
				bestParamScoreDict=scoreDict
		# End grid/parameter loop


	# Set threshold to input value from best (and possibly only) run for use in results analysis
	# Unless not specified at all
	if 'story_threshold' in bestParams and bestParams['story_threshold']!=None:
		threshold=bestParams['story_threshold']
	else:
		threshold=None


	# If there was a real parameter grid, then output/refresh results
	if len(parameterGrid)>1:
		print("BEST PARAMETERS:")
		print(bestParams)
		print(bestParamScoreDict)
		scoreCurrentParamGuess(tfidfVectors,storyMap,articleDataFrame,threshold,printErrors=True)
		# Recreate vector for best results in loop
		# terms is just used later on if analysis of final results is requested
		tfidfVectors,terms=preprocessAndVectorize(articleDataFrame,
												  bestParams,
												  pos_nlp_mapping,
												  nlp,
												  nl,
												  wordnet_lemmatizer,
												  stop_words)


	# If requested, generate Bokeh graph
	# 0 index is required because the parameters are forced to be lists by ParameterGrid
	if runParams['display_graph'][0]:
		graphVectorSpace(tfidfVectors,
						 articleDataFrame[['id','publication','content no nonascii']],
						 runParams['process_date'],
						 storyMap,
						 threshold)


	# Continue with outputting from best results if requested
	if reportArticleList!=None:
		produceRequestedReportDetails(tfidfVectors,articleDataFrame,reportArticleList,threshold,storyMap,terms)


	return

##########################################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Provide arguments for topic mining')

	parser.add_argument('--input-file', help='file containing news articles to process', default='./data/articles.csv')
	parser.add_argument('--article-id-list', help='list of article IDs to produce topic matches for',nargs='+',type=int)
	parser.add_argument('--article-stats', help='print number of available articles by date and publication', type=str2bool, default=False) 
	parser.add_argument('--story-map-validation', help='file containing links between articles for validation')
	parser.add_argument('--process-date', help='process articles from this date', default='2016-08-22')
	parser.add_argument('--stop-words-file', help='file containing stop words for omission', default='./data/stopWords.txt')

	parser.add_argument('--nlp-library', help='library used for text analysis', default='spaCy')

	parser.add_argument('--pos-list', help='parts-of-speech to restrict to i.e. VERB PROPER COMMON',nargs='+',type=str,default=['ALL'])

	parser.add_argument("--lemma-conversion", help='convert article words to root form (spaCy only)', type=str2bool, default=False)
	parser.add_argument("--ngram-max", help='maximum length for ngrams', type=int, default=1)

	parser.add_argument('--max-length', help='constrain processed article length - in number of words', type=int)
	parser.add_argument('--story-threshold', help='minimum score for two articles to be considered part of same story',type=float)

	parser.add_argument('--tfidf-maxdf', help='max_df for tf_idf vectorizer FLOAT i.e. PROPORTION OF DOCUMENTS',type=float)
	parser.add_argument('--tfidf-mindf', help='min_df for tf_idf vectorizer INTEGER i.e. NUMBER OF DOCUMENTS',type=int)
	parser.add_argument('--tfidf-binary', help='binary df for tf_idf vectorizer', type=str2bool, default=False)
	parser.add_argument('--tfidf-norm', help='optionally apply normalization in tf_idf vectorizer - l1, l2', default=None)
	parser.add_argument('--display-graph', help='generate Bokeh graph of SVD of results', type=str2bool, default=False)
	parser.add_argument('--grid-parameter-file', help='parameter ranges for grid search')

	args = vars(parser.parse_args())

	main(args)

##########################################################################################
