__author__ = 'Sanjeev K C'

from nltk import classify
from nltk import probability
from collections import Counter
import cPickle as pickle
import textblob

#updates tweets[] by filtering pos and neg tweets together passing
#through stopwords.txt

def filterTweets(trData):
    trainData = []
    inFile = open("stopwords.txt",'r')
    stopWords = inFile.read().split("\n")

    for text,label in trData:
        txt = text.lower().split(' ')
        trainData.append(([t for t in txt if t not in stopWords],label))
    #print "trainData","\n",trainData,"\n" #[(['love', 'car'], 'positive')]
    return trainData

# noinspection PyUnreachableCode
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

    pos_prob = float(count['positive'])/total_classes
    neg_prob = float(count['negative'])/total_classes
    neutral_prob = float(count['neutral'])/total_classes

    #print {'p(positive)':pos_prob,'p(negative)':neg_prob,'p(neutral)':neutral_prob},"\n"
    return {'p(positive)':pos_prob,'p(negative)':neg_prob,'p(neutral)':neutral_prob}

def posterior_prob(g_features):

    training_sets = []
    pos_total_words = neg_total_words = neutral_total_words = 0
    #global _feature_vector,_trained_Data
    #repeated values included
    for each_feature in g_features:
        if each_feature[1] == 'positive':
            pos_total_words += sum(each_feature[0].values())
        elif each_feature[1] == 'negative':
            neg_total_words += sum(each_feature[0].values())
        else:
            neutral_total_words += sum(each_feature[0].values())
    total_unique_words = len(g_features)
    #print pos_total_words,neg_total_words,total_unique_words

    for each_feature,label in g_features:
        #for key,value in each_feature.iteritems():
        if label == 'positive':
            training_sets.append([
                [key for key,value in each_feature.iteritems()][0],label,
                float(sum(each_feature.values())+1)/
                (pos_total_words + total_unique_words)])
        elif label == 'negative':
            training_sets.append([
                [key for key,value in each_feature.iteritems()][0],label,
                float(sum(each_feature.values())+1)/
                (neg_total_words + total_unique_words)])
        else:
            training_sets.append([
                [key for key,value in each_feature.iteritems()][0],label,
                float(sum(each_feature.values())+1)/
                (neutral_total_words + total_unique_words)])

    #print "training_sets","\n", training_sets
    return training_sets

def trainNBClassifier(trData):

    tweets = filterTweets(trData)
    with open('TRAIN_SETS.PICKLE', 'wb') as classifier:
        pickle.dump(posterior_prob(_get_features(tweets)),classifier)
        pickle.dump(prior_prob(tweets), classifier)

def testNBClassifier():
    #tweets = filterTweets(testData)
    with open('TRAIN_SETS.PICKLE', 'rb') as f:
        classifier = pickle.load(f)
    print classifier

pos_tweets = [('I love this car', 'positive'),
              ('This is amazing', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative')]

neu_tweets = [('I bought this car', 'neutral'),
              ('This is the view', 'neutral'),
              ('I woke up stupid morning', 'neutral')]

td = pos_tweets + neg_tweets + neu_tweets

trainNBClassifier(td)
testNBClassifier()
