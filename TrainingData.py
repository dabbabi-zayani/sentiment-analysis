__author__ = 'Sanjeev K C'

from collections import Counter
import csv
import re
import string
import cPickle as pickle


from nltk import probability
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger

#Taken from "https://github.com/vivekn/sentiment/blob/master/info.py" and modified for my purpose
#Thank you to vivek

def negate_sequence(text):
    negation = False
    delimiters = "?.,!:;"
    result = []
    refined_text = ""
    words = text.split(" ")
    for word in words:
        stripped = word.strip(delimiters).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation
        if any(c in word for c in delimiters):
            negation = False
    refined_text = " ".join(result)
    return refined_text

def strip_nouns(text):
    blob = TextBlob(text, pos_tagger=PerceptronTagger())
    expr  = '|'.join(blob.noun_phrases)
    regex = re.compile(r'\b('+expr+r')\b', flags=re.IGNORECASE)
    txt = regex.sub("", text)
    return txt

def extract_data(file_name):
    MyValues = [] #create an empty list
    rows = csv.reader(open(file_name, 'rb'), delimiter=b',')
    for row in rows:
        txt = negate_sequence(row[1])
        #line = re.sub('[%"-&\\\/]|(@\w+|#\w+|http\S+|[0-9])', '', txt)
        line = re.sub('(['+string.punctuation.replace("'","")+']|@\w+|#\w+|http\S+|[0-9])','',row[1])
        if row[0] == '0':
            MyValues.append((line,'negative'))
        elif row[0] == '2':
            MyValues.append((line,'neutral'))
        else:
            MyValues.append((line,'positive'))
    return MyValues

def filterTweets(trData):
    trainData = []
    inFile = open("stopwords.txt",'r')
    stopWords = inFile.read().split("\n")

    for text,label in trData:
        txt = text.lower().split(' ')
        trainData.append(([t for t in txt if t not in stopWords],label))
    #print "trainData","\n",trainData,"\n" #[(['love', 'car'], 'positive')]
    return trainData

def _get_features(f_tweets):

    _feature_vector = []
    pos_features_dist = []
    neg_features_dist = []
    neutral_features_dist = []

    for token,label in f_tweets:
        if label == 'positive':
            pos_features_dist.extend(token)
        elif label == 'negative':
            neg_features_dist.extend(token)
        else:
            neutral_features_dist.extend(token)

    pos_features_dist = probability.FreqDist(pos_features_dist)
    for key,value in pos_features_dist.iteritems():
        _feature_vector.append(({key:value},'positive'))

    neg_features_dist = probability.FreqDist(neg_features_dist)
    for key,value in neg_features_dist.iteritems():
        _feature_vector.append(({key:value},'negative'))

    neutral_features_dist = probability.FreqDist(neutral_features_dist)
    for key,value in neutral_features_dist.iteritems():
        _feature_vector.append(({key:value},'neutral'))

    #print "_feature_vector",'\n',_feature_vector,"\n" #[({'car': 1}, 'positive')]
    return _feature_vector

def prior_prob(f_tweets):

    count = Counter(elem[1] for elem in f_tweets)
    total_classes = count['positive'] + count['negative'] + count['neutral']
    #print total_classes
    pos_prob = float(count['positive'])/total_classes
    neg_prob = float(count['negative'])/total_classes
    neutral_prob = float(count['neutral'])/total_classes

    #print {'p(positive)':pos_prob,'p(negative)':neg_prob,'p(neutral)':neutral_prob},"\n"
    return {'class_positive_prob':pos_prob,'class_negative_prob':neg_prob,'class_neutral_prob':neutral_prob}

def posterior_prob(g_features):

    pos_total_words = 0
    neg_total_words = 0
    neutral_total_words = 0
    total_unique_words = 0

    pos_sets={}
    neg_sets={}
    neu_sets={}

    for each_feature,label in g_features:
        #print each_feature,label
        if label == 'positive':
            pos_total_words += sum(each_feature.values())
        elif label == 'negative':
            neg_total_words += sum(each_feature.values())
        else:
            neutral_total_words += sum(each_feature.values())

    total_unique_words = len(g_features)
    #print pos_total_words,neg_total_words,neutral_total_words,total_unique_words

    for each_feature,label in g_features:
       if label == 'positive':
            pos_sets[[key for key,value in each_feature.iteritems()][0]] =\
                float(sum(each_feature.values())+1)/\
                (pos_total_words + total_unique_words)
       elif label == 'negative':
            neg_sets[[key for key,value in each_feature.iteritems()][0]] =\
                float(sum(each_feature.values())+1)/\
                (neg_total_words + total_unique_words)
       elif label == 'neutral':
            neu_sets[[key for key,value in each_feature.iteritems()][0]] =\
                float(sum(each_feature.values())+1)/\
                (neutral_total_words + total_unique_words)
    #print pos_sets,neg_sets,neu_sets
    return [pos_sets,neg_sets,neu_sets],[pos_total_words,neg_total_words,neutral_total_words,total_unique_words]

def trainNBClassifier(trData):

    tweets = filterTweets(trData)
    with open('TRAIN_SETS.PICKLE', 'wb') as classifier_file:
        pickle.dump(posterior_prob(_get_features(tweets)),classifier_file)
        pickle.dump(prior_prob(tweets), classifier_file)
    print "Classifier written to pickle file in current folder"

if __name__=='__main__':

    # train_data = extract_data('train.csv')
    # trainNBClassifier(train_data)

    print "--FINISHED--"
