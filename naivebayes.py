__author__ = 'Sanjeev K C'

from nltk import classify
from nltk import probability
from collections import Counter
import cPickle as pickle
import textblob

#global variables

pos_total_words = 0
neg_total_words = 0
neutral_total_words = 0
total_unique_words = 0
accuracy = 0

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
    #print total_classes
    pos_prob = float(count['positive'])/total_classes
    neg_prob = float(count['negative'])/total_classes
    neutral_prob = float(count['neutral'])/total_classes

    #print {'p(positive)':pos_prob,'p(negative)':neg_prob,'p(neutral)':neutral_prob},"\n"
    return {'class_positive_prob':pos_prob,'class_negative_prob':neg_prob,'class_neutral_prob':neutral_prob}

def posterior_prob(g_features):

    global pos_total_words,neg_total_words,neutral_total_words,total_unique_words

    pos_sets={}
    neg_sets={}
    neu_sets={}

    #global _feature_vector,_trained_Data
    #repeated values included

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
        #for key,value in each_feature.iteritems():
        #print each_feature,label
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
    return pos_sets,neg_sets,neu_sets

def trainNBClassifier(trData):

    tweets = filterTweets(trData)
    with open('TRAIN_SETS.PICKLE', 'wb') as classifier_file:
        pickle.dump(posterior_prob(_get_features(tweets)),classifier_file)
        pickle.dump(prior_prob(tweets), classifier_file)

def predict_label_aux(classifier,tweet,class_probabilities):

    global accuracy,pos_total_words,neg_total_words,neutral_total_words,total_unique_words
    pos_prob = 0
    neg_prob = 0
    neu_prob = 0

    text = tweet[0]
    label = tweet[1]

    for t in text:

        if t in classifier[0]:
            pos_prob += classifier[0][t]
        else:
            pos_prob += float(1)/(pos_total_words + total_unique_words)
            classifier[0][t] = float(1)/(pos_total_words + total_unique_words)

        if t in classifier[1]:
            neg_prob += classifier[1][t]
        else:
            neg_prob += float(1)/(neg_total_words + total_unique_words)
            classifier[1][t] = float(1)/(neg_total_words + total_unique_words)

        if t in classifier[2]:
            neu_prob += classifier[2][t]
        else:
            neu_prob += float(1)/(neutral_total_words + total_unique_words)
            classifier[2][t] = float(1)/(neutral_total_words + total_unique_words)

    pos_prob *= class_probabilities['class_positive_prob']
    neg_prob *= class_probabilities['class_negative_prob']
    neu_prob *= class_probabilities['class_neutral_prob']

    max_prob = max(pos_prob,neg_prob,neu_prob)
    if max_prob == pos_prob:
        predicted_label = 'positive'
    elif max_prob == neg_prob:
        predicted_label = 'negative'
    else:
        predicted_label = 'neutral'

    if predicted_label == label:
        accuracy+=1

    return predicted_label

def predict_label(classifier,tweets,class_probabilities):
    for tweet in tweets:
        print tweet, " is ",predict_label_aux(classifier,tweet,class_probabilities)

def testNBClassifier(testData):
    with open('TRAIN_SETS.PICKLE', 'rb') as f:
        classifier = pickle.load(f)
        prior_probs = pickle.load(f)
    testData_features = filterTweets(testData)
    predict_label(classifier,testData_features,prior_probs)
    print "Accuracy is ", (float(accuracy)/len(testData))*100,"%"
    print "\n"

pos_tweets = [('I love this car', 'positive'),
              ('This is amazing', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative')]

neu_tweets = [('I bought this car', 'neutral'),
              ('This is the view', 'neutral'),
              ('I woke up stupid morning', 'neutral')]

tr_data = pos_tweets + neg_tweets + neu_tweets

test_data = [('I feel happy this morning','positive'),
             ('Larry is my friend','positive'),
             ('I do not like that man','negative'),
             ('My house is not great','negative'),
             ('Your song is annoying','negative')]

trainNBClassifier(tr_data)
testNBClassifier(test_data)
