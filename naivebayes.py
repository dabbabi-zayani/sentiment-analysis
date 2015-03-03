__author__ = 'Sanjeev K C'

from nltk import classify
from nltk import probability
from collections import Counter
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
import csv,re,cPickle as pickle

accuracy = 0

def strip_nouns(text):
    blob = TextBlob(text, pos_tagger=PerceptronTagger())
    expr  = '|'.join(blob.noun_phrases)
    regex = re.compile(r'\b('+expr+r')\b', flags=re.IGNORECASE)
    txt = regex.sub("", text)
    return txt

def extract_data(file_name):
    MyValues = [] #create an empty list
    rows = csv.reader(open(file_name, 'rb'), delimiter=',')
    for row in rows:
        line = re.sub('([!,".?%-&\)\(\/\\,:;-]|@\w+|#\w+|http\S+|[0-9])', '', row[1])
        t = TextBlob(line)
        if row[0] == '0':
            MyValues.append((t.correct(),'negative'))
            #print t.correct(),"\n"
        elif row[0] == '2':
            MyValues.append((t.correct(),'neutral'))
            #print t.correct(),"\n"
        else:
            MyValues.append((t.correct(),'positive'))
            #print t.correct(),"\n"
    return MyValues

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
    return [pos_sets,neg_sets,neu_sets],[pos_total_words,neg_total_words,neutral_total_words,total_unique_words]

def trainNBClassifier(trData):

    tweets = filterTweets(trData)
    with open('TRAIN_SETS.PICKLE', 'wb') as classifier_file:
        pickle.dump(posterior_prob(_get_features(tweets)),classifier_file)
        pickle.dump(prior_prob(tweets), classifier_file)

def predict_label_aux(classifier,tweet,class_probabilities,words_stats):

    global accuracy

    pos_prob = 0
    neg_prob = 0
    neu_prob = 0

    text = tweet[0]
    label = tweet[1]

    pos_dict = classifier[0]
    neg_dict = classifier[1]
    neu_dict = classifier[2]

    pos_total_words = words_stats[0]
    neg_total_words = words_stats[1]
    neutral_total_words = words_stats[2]
    total_unique_words = words_stats[3]

    for t in text:
        if t in pos_dict:
            pos_prob += pos_dict[t]
            #pos_dict[t] = pos_prob
        else:
            pos_prob += 1/float((pos_total_words + total_unique_words))
            #pos_dict[t] = 1/float((pos_total_words + total_unique_words))
            #pos_total_words += 1

        if t in neg_dict:
            neg_prob += neg_dict[t]
            #neg_dict[t] = neg_prob
        else:
            neg_prob += 1/float((neg_total_words + total_unique_words))
            #neg_dict[t] = 1/float((neg_total_words + total_unique_words))
            #neg_total_words += 1

        if t in neu_dict:
            neu_prob += neu_dict[t]
            #neu_dict[t] = neu_prob
        else:
            neu_prob += 1/float((neg_total_words + total_unique_words))
            #neu_dict[t] = 1/float((neutral_total_words + total_unique_words))
            #neutral_total_words += 1

    #total_unique_words = len((set(pos_dict.keys()) & set(neg_dict.keys())) & set(neu_dict.keys()))

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

def predict_label(classifier,tweets,class_probabilities,words_stats):
    for tweet in tweets:
        print tweet[0], " is ",predict_label_aux(classifier,tweet,class_probabilities,words_stats)

def testNBClassifier(testData):
    with open('TRAIN_SETS.PICKLE', 'rb') as classifier_file:
        extracted_data = pickle.load(classifier_file)
        prior_probabilities = pickle.load(classifier_file)
    #print "Before testing data","\n", extracted_data,"\n"
    classifier = extracted_data[0]
    words_count = extracted_data[1]
    testData_features = filterTweets(testData)
    predict_label(classifier,testData_features,prior_probabilities,words_count)
    #print"After testing data","\n",extracted_data
    print "Accuracy is ", (float(accuracy)/len(testData))*100,"%"
    print "\n"

def filterTweet(text):
    trainData = []
    inFile = open("stopwords.txt",'r')
    stopWords = inFile.read().split("\n")
    txt = text.lower().split(' ')
    trainData.extend([t for t in txt if t not in stopWords])
    #print "trainData","\n",trainData,"\n" #[(['love', 'car'], 'positive')]
    return trainData

def _get_sentiment():
    input = raw_input("Enter your tweet:\n")
    with open('TRAIN_SETS.PICKLE', 'rb') as classifier_file:
        extracted_data = pickle.load(classifier_file)
        prior_probabilities = pickle.load(classifier_file)
    classifier = extracted_data[0]
    words_count = extracted_data[1]
    text = filterTweet(input)
    #predict_label(classifier,,prior_probabilities,words_count)

    pos_prob = 0
    neg_prob = 0
    neu_prob = 0

    pos_dict = classifier[0]
    neg_dict = classifier[1]
    neu_dict = classifier[2]

    pos_total_words = words_count[0]
    neg_total_words = words_count[1]
    neutral_total_words = words_count[2]
    total_unique_words = words_count[3]

    for t in text:
        if t in pos_dict:
            pos_prob += pos_dict[t]
            #pos_dict[t] = pos_prob
        else:
            pos_prob += 1/float((pos_total_words + total_unique_words))
            #pos_dict[t] = 1/float((pos_total_words + total_unique_words))
            #pos_total_words += 1

        if t in neg_dict:
            neg_prob += neg_dict[t]
            #neg_dict[t] = neg_prob
        else:
            neg_prob += 1/float((neg_total_words + total_unique_words))
            #neg_dict[t] = 1/float((neg_total_words + total_unique_words))
            #neg_total_words += 1

        if t in neu_dict:
            neu_prob += neu_dict[t]
            #neu_dict[t] = neu_prob
        else:
            neu_prob += 1/float((neg_total_words + total_unique_words))
            #neu_dict[t] = 1/float((neutral_total_words + total_unique_words))
            #neutral_total_words += 1

    #total_unique_words = len((set(pos_dict.keys()) & set(neg_dict.keys())) & set(neu_dict.keys()))

    pos_prob *= prior_probabilities['class_positive_prob']
    neg_prob *= prior_probabilities['class_negative_prob']
    neu_prob *= prior_probabilities['class_neutral_prob']

    max_prob = max(pos_prob,neg_prob,neu_prob)

    if max_prob == pos_prob:
        print 'positive'
    elif max_prob == neg_prob:
        print 'negative'
    else:
        print 'neutral'

train_data = extract_data('trainset.csv')
trainNBClassifier(train_data)

test_data = extract_data('testset.csv')
testNBClassifier(test_data)

_get_sentiment()
