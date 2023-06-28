import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gensim
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import joblib
from tqdm import tqdm

train = pd.read_csv('train_tweet.csv')
test = pd.read_csv('test_tweets.csv')

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train.tweet)


def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

nltk.download('stopwords') #If this dosen't work manually copy the 'nltk_data' folder into C:\\ or /usr 


tokenized_tweet = train['tweet'].apply(lambda x: x.split()) 

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size=200, 
            window=5, 
            min_count=2,
            sg = 1, 
            hs = 0,
            negative = 10, 
            workers= 2, 
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']),

tqdm.pandas(desc="progress-bar")

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output


labeled_tweets = add_label(tokenized_tweet)
labeled_tweets[:6]

test_corpus = []
for i in range(0, 17197):
    review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)






