import pandas as pd
from tabulate import tabulate
import csv
import time
import datetime
import dateutil.parser
import re
import unicodedata

'''
filepath_dict = {'trump':   './tweetdata/trump_06162015-06012019.csv',
                 'warren': 'tweetdata/warren_01202015-06012019.csv',
                 'sanders':   'tweetdata/sanders_05012009-06012019.csv'}
'''

filepath_dict = {'trump':   './tweetdata/trump_06162015-06012019.csv'}

print('Dough: Init r01-0606_2019')
tweets_df = None
tweets_source = None
for source, filepath in filepath_dict.items():
    tweets_source = source
    print('Dough: Reading tweets csv data from {0}...'.format(filepath))
    reader = csv.DictReader(open(filepath), delimiter=';')
    csv_dict = []

    for line in reader:
        line_convert_dict = dict(line)

        data = {}
        data['id'] = line_convert_dict['id']
        data['username'] = line_convert_dict['username']
        data['retweets'] = int(line_convert_dict['retweets'])
        data['likes'] = int(line_convert_dict['favorites'])
        data['permalink'] = line_convert_dict['permalink']

        normalized_tweet_text = unicodedata.normalize("NFKD", line_convert_dict['text'])
        extract_url = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', str(normalized_tweet_text)) #re.findall(r'(https?://[^\s]+)', line_convert_dict['text'])
        if extract_url:
            data['link'] = extract_url[0]

        text_without_url = re.sub(r'https?://\S+', '', normalized_tweet_text) #r"http\S+"
        twitter_pic_url = re.findall(r'pic.twitter\S+', text_without_url)
        twitter_pic_url_rmquotes = ''
        if twitter_pic_url:
            twitter_pic_url_rmquotes = re.sub('"', '', twitter_pic_url[0])

        text_without_any_url = re.sub(r'pic.twitter\S+', '', text_without_url)

        text_remove_atsymbolspacing = re.sub(r'(@)\s+([a-zA-Z0-9])', r' \1\2', text_without_any_url) 
        text_remove_hashtagspacing = re.sub(r'(#)\s+([a-zA-Z0-9])', r' \1\2', text_remove_atsymbolspacing) 

        text_get_hashtags = re.findall(r'#(\w+)', text_remove_hashtagspacing)
        text_get_mentionusernames = re.findall(r'@(\w+)', text_remove_hashtagspacing)
        text_get_mentionusernames_without_self = [ item for item in text_get_mentionusernames if line_convert_dict['username'] not in item ]

        data['tweet'] = text_remove_hashtagspacing
        data['tweet_mentions'] = text_get_mentionusernames_without_self
        data['tweet_hashtags'] = text_get_hashtags
        data['tweet_pic'] = twitter_pic_url_rmquotes
        dt_tweetdate = dateutil.parser.parse(line_convert_dict['date'])
        
        # for 2016election - trump 
        dt_afterelection = dateutil.parser.parse('01/21/2017')
        dt_firstyear = dateutil.parser.parse('01/21/2018')
        if dt_tweetdate < dt_firstyear:
            continue

        unixts_tweetdate = int(time.mktime(dt_tweetdate.timetuple()))
        data['timestamp'] = unixts_tweetdate

        csv_dict.append(data)
        
    tweets_df = pd.DataFrame(csv_dict)

#for LDA
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import text 
from pprint import pprint

#gensim preprocess tokenization and cleaning
print('Dough: Tokenizing tweets dataframe...')
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

tweets_tokens = list(sent_to_words(tweets_df['tweet'].values.astype('U')))

#spacy lemmatization
print('Dough: Lemmatizing tweets dataframe...')
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

#python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
tweets_lemmatized = lemmatization(tweets_tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#stopwords config and countvectorizer
custom_stopwords = ['realdonaldtrump', 'trump', 'just', 'donald', 'live']
stopwords_list = text.ENGLISH_STOP_WORDS.union(custom_stopwords)

print('Dough: Creating Doc-Word matrix using CountVectorizer_fit/transform...')
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words=stopwords_list,             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

tweets_vectorized = vectorizer.fit_transform(tweets_lemmatized)
print('Dough: Processing the sparse data...')
data_dense = tweets_vectorized.todense()

#Sparsity = Percentage of Non-Zero cells
print("Dough: Data sparsity is ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

#LDA modeling
'''
print('Dough: LDA Modeling...')
lda_model = LatentDirichletAllocation(n_components=10,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(tweets_vectorized)
print('Dough: LDA Model - {0}'.format(lda_model)) 

#Log Likelyhood: Higher the better, Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Dough: LDA Model Log Likelihood: ", lda_model.score(tweets_vectorized))
print("Dough: LDA Model Perplexity: ", lda_model.perplexity(tweets_vectorized))
'''

#GridSearch to find optimal n_components
search_params = {'n_components': [6, 15, 20], 'learning_decay': [.7,.9]}
print('Dough: Finding optimal model configuration using GridSearchCV...')
print('Dough: This will take up a few minutes...')
lda = LatentDirichletAllocation(n_jobs= -1)
model = GridSearchCV(lda, param_grid=search_params)
model.fit(tweets_vectorized)

best_lda_model = model.best_estimator_
print("Optimal model parameters: ", model.best_params_)

#Log Likelyhood: Higher the better, Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Optimal model log Likelihood Score: ", model.best_score_)
print("Optimal model Perplexity: ", best_lda_model.perplexity(tweets_vectorized))

print('Dough: Dataframe - topic keyword matrix')
lda_output = best_lda_model.transform(tweets_vectorized)
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)] # column names
docnames = ["Doc" + str(i) for i in range(len(data))] # index names

df_topic_keywords = pd.DataFrame(best_lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
print(df_topic_keywords.head())

print('Dough: Finding top 15 topics...')# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
    
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

print('Dough: Generating dataframe - top 15 topics')
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print(tabulate(df_topic_keywords))

topics_input = input('Dough: Provide topics for the list: ')
df_topic_keywords["topics"] = eval(topics_input)
print(tabulate(df_topic_keywords))
tweets_df['topic_cluster'] = lda_output.argmax(axis=1)  

print('Dough: Adding provided topics to the dataframe...')
topic_cluster_keyword = []
for x in tweets_df['topic_cluster'].tolist():
    topic_cluster_keyword.append(df_topic_keywords["topics"][x])

print('Dough: Generating final result files...')
tweets_df['topic_keywords'] = topic_cluster_keyword
print(tweets_df.head())
tweets_df.to_csv('./result_topic_cluster_' + tweets_source + '-secondyear.csv', sep='\t', index=False)
tweets_df.to_json('./result_topic_cluster_' + tweets_source + '-secondyear.json', orient='records')
