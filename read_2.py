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
        unixts_tweetdate = int(time.mktime(dt_tweetdate.timetuple()))
        data['timestamp'] = unixts_tweetdate

        #rint(data)
        csv_dict.append(data)
        
    tweets_df = pd.DataFrame(csv_dict)
    #print(tweets_df['tweet'][0:50])

#for LDA
#print(tweets_df.head())


import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
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
custom_stopwords = ['realdonaldtrump']
stopwords_list = text.ENGLISH_STOP_WORDS.union(custom_stopwords)


def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])


# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_list)
tfidf = tfidf_vectorizer.fit_transform(tweets_tokens)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_list)
tf = tf_vectorizer.fit_transform(tweets_tokens)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 10

# Run NMF
nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_W = nmf_model.transform(tfidf)
nmf_H = nmf_model.components_

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

no_top_words = 15
no_top_documents = 30
print('nmf:')
display_topics(nmf_H, nmf_W, tfidf_feature_names, tweets_tokens, no_top_words, no_top_documents)
print('LDA:')
display_topics(lda_H, lda_W, tf_feature_names, tweets_tokens, no_top_words, no_top_documents)