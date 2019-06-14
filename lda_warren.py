import pandas as pd
from tabulate import tabulate
import csv
import time
import datetime
import dateutil.parser
import re
import unicodedata
#for LDA
import numpy as np
import pandas as pd
import re, nltk, spacy
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
#python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

filepath_dict = {'warren':   './tweetdata/warren_01202015-06012019.csv'}
print('Dough: Init r06-0611_2019-warren')

def preprocess_tweets(startDate=None, endDate=None):
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

            if startDate is not None and endDate is not None:
                dt_startDate = dateutil.parser.parse(startDate)
                dt_endDate = dateutil.parser.parse(endDate)
                if dt_tweetdate < dt_startDate or dt_tweetdate > dt_endDate:
                    continue
            elif startDate is not None:
                dt_startDate = dateutil.parser.parse(startDate)
                if dt_tweetdate < dt_startDate:
                    continue
            elif endDate is not None:
                dt_endDate = dateutil.parser.parse(endDate)
                if dt_tweetdate > dt_endDate:
                    continue

            unixts_tweetdate = int(time.mktime(dt_tweetdate.timetuple()))
            data['timestamp'] = unixts_tweetdate

            csv_dict.append(data)
            
        return source, pd.DataFrame(csv_dict)

#define dataset for processing
tweets_init = preprocess_tweets()
tweets_type = 'full'

#df from dataset
tweets_df = tweets_init[1]
tweets_source = tweets_init[0]

#nltk stopwords
stop_words = stopwords.words('english')
stop_words.extend(['just', 'great', 'live', 'thank', 'people', 'today', 'watch']) #custom per-type

#gensim preprocess tokenization and cleaning
def sent_to_words(sentences):
    print('Dough: Tokenizing tweets in the dataframe...')
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

tweets_tokenized = list(sent_to_words(tweets_df['tweet']))
print('Dough: DONE Tokenizing tweets')

# Build the bigram and trigram models
print('Dough: Building bigram/trigram models...')
bigram = gensim.models.Phrases(tweets_tokenized, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[tweets_tokenized], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

#spacy lemmatization
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    print('Dough: Lemmatizing tweets in the dataframe...')
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return texts_out

print('Dough: Pre-processing tokens...')
tweets_tokenized_nostops = remove_stopwords(tweets_tokenized)
tweets_tokenized_bigrams = make_bigrams(tweets_tokenized_nostops)
tweets_lemmatized = lemmatization(tweets_tokenized_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print('Dough: Generating dictionary and corpus...')
id2word = corpora.Dictionary(tweets_lemmatized)
tweets_lemmatized_corpustexts = tweets_lemmatized
tweets_lemmatized_corpustdf = [id2word.doc2bow(text) for text in tweets_lemmatized_corpustexts]

mallet_path = './mallet-2.0.8/bin/mallet' # update this path
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

print('Dough: Calculating coherence values using mallet, could take a few minutes...')
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=tweets_lemmatized_corpustdf, texts=tweets_lemmatized, start=5, limit=20, step=2)

print('Dough: Displaying the graph')
x = range(5, 20, 2)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

optimal_model_index = eval(input('Dough: Provide an optimal model index:  '))
optimal_model = model_list[optimal_model_index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['dominant_topic_no', 'topic_contribution_percentage', 'topic_keyword']

    # Add original text to the end of the output
    sent_topics_df = pd.concat([sent_topics_df, texts], axis=1)
    return(sent_topics_df)

print('Dough: Generating full dataset with topic keywords')
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=tweets_lemmatized_corpustdf, texts=tweets_df)
df_dominant_topic = df_topic_sents_keywords.reset_index()
#df_dominant_topic.columns = ['doc_no', 'dominant_topic_no', 'topic_contribution_percentage', 'topic_keyword', 'tweet']

print(df_dominant_topic.head(10))
df_dominant_topic.to_csv('./output/result_lda_full_' + tweets_source + '-' + tweets_type + '.csv', sep='\t', index=False)
df_dominant_topic.to_json('./output/result_lda_full_' + tweets_source + '-' + tweets_type + '.json', orient='records')
print('Dough: Saved full dataset with topic data in ', ' ./output/result_lda_full_' + tweets_source + '-' + tweets_type + '.csv')

# Group top 5 sentences under each topic
print('Dough: Generating top 5 documents dataset in each topic...')
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('dominant_topic_no')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['topic_contribution_percentage'], ascending=[0]).head(10)], 
                                            axis=0)

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
#sent_topics_sorteddf_mallet.columns = ['topic_no', "topic_contribution_percentage", "topic_keyword", "tweet"]

print(sent_topics_sorteddf_mallet.head())
sent_topics_sorteddf_mallet.to_csv('./output/result_lda_top_' + tweets_source + '-' + tweets_type + '.csv', sep='\t', index=False)
sent_topics_sorteddf_mallet.to_json('./output/result_lda_top_' + tweets_source + '-' + tweets_type + '.json', orient='records')
print('Dough: Saved top 5 document dataset in ', ' ./output/result_lda_top_' + tweets_source + '-' + tweets_type + '.csv')