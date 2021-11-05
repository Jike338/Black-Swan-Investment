import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# function to remove punctuation from text (input is a string)
def clean_text(sentence):
	
	clean_sentence = "".join(l for l in sentence if l not in string.punctuation)
	
	return clean_sentence

# function to calculate cosine similarity using bow representation (input is a dataframe)
def bow_similarity(sentences_df):
	
	# first lets clean the text by removing punctuation
	sentences_df['clean_text'] = sentences_df.apply(lambda row: clean_text(row['sentence_text']), axis=1)
	
	# initialise the bag of words tokeniser and apply it to our clean text
	# this will create vector representations for each word
	count_vec = CountVectorizer()
		
	dtm = count_vec.fit_transform(sentences_df['clean_text']).toarray()
	
	# calculate similarity, returns an NxN matrix with 1's across diagonal
	similarity_df = pd.DataFrame(cosine_similarity(dtm)).reset_index()
	
	# here we are going to unpivot the similairty matrix to return
	# the data in a format like:
	# | sentence_a | sentence_b | similarity |
	
	# unpivot the similarity df
	df_unpiv = pd.melt(similarity_df, id_vars=['index'])
	
	# get unique combinations of sentance_a and sentance_b
	df_unpiv_unique = (df_unpiv.loc[pd.DataFrame(np.sort(df_unpiv[['index', 'variable']],1),index=df_unpiv.index)
						.drop_duplicates(keep='first')
						.index])


	# remove instances where sentence a == sentence b
	df_unpiv_unique = df_unpiv_unique[df_unpiv_unique['index'] != df_unpiv_unique['variable']]
	
	# now finally join on the original df to get the required output
	# join the text
	df_with_text = pd.merge(df_unpiv_unique, sentences_df.reset_index()
							, left_on='index'
							, right_on='index')
	df_with_text = pd.merge(df_with_text, sentences_df.reset_index()
							, left_on='variable'
							, right_on='index')

	df_with_text = df_with_text.loc[:,['sentence_text_y', 'sentence_text_x', 'value']]
	df_with_text.columns = ['sentence_a', 'sentence_b', 'similarity']
	
	return df_with_text

sentences = ['Hi, how are you?', 'Hey what\'s up?']
sentences_df = pd.DataFrame({'sentences':sentences})

print(bow_similarity(sample_sentences))