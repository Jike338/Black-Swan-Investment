import pandas as pd
import numpy as np
import string
import itertools
import spacy

# function to remove punctuation from text (input is a string)
def clean_text(sentence):
	
	clean_sentence = "".join(l for l in sentence if l not in string.punctuation)
	
	return clean_sentence

# function to calculate the cosine
def cosine_similarity_calc(vec_1,vec_2):
	
	sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
	
	return sim

# function to calculate cosine similarity using word vectors (input is a series)
def embeddings_similarity(sentences):
	
	# first we need to get data into | sentence_a | sentence_b | format
	sentence_pairs = list(itertools.combinations(sentences, 2))
	
	sentence_a = [pair[0] for pair in sentence_pairs]
	sentence_b = [pair[1] for pair in sentence_pairs]
	
	sentence_pairs_df = pd.DataFrame({'sentence_a':sentence_a, 'sentence_b':sentence_b})
	
	# get unique combinations of sentance_a and sentance_b
	sentence_pairs_df = sentence_pairs_df.loc[pd.DataFrame(np.sort(sentence_pairs_df[['sentence_a', 'sentence_b']],1)
														   ,index=sentence_pairs_df.index).drop_duplicates(keep='first').index]

	# remove instances where sentence a == sentence b
	sentence_pairs_df = sentence_pairs_df[sentence_pairs_df['sentence_a'] != sentence_pairs_df['sentence_b']]
	
	# load word embeddings (will use these to convert sentence to vectors)
	# Note you will need to run the following command (from cmd) to download embeddings: 
	# 'python -m spacy download en_core_web_lg'
	embeddings = spacy.load('en_core_web_lg')
	
	# now we are ready to calculate the similarity
	
	sentence_pairs_df['similarity'] = sentence_pairs_df.apply(lambda row: cosine_similarity_calc(embeddings(clean_text(row['sentence_a'])).vector,
																					embeddings(clean_text(row['sentence_b'])).vector), axis=1)
	
	return sentence_pairs_df

# calculate similarity for sample sentences
sentences = ['Hi, how are you?', 'Hey what\'s up?']
print(embeddings_similarity(sentences))