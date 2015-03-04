__author__ = 'Sanjeev K C'

from TrainingData import *
import math

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
            pos_prob += math.log(pos_dict[t])
        else:
            pos_prob += math.log(1/float((pos_total_words + total_unique_words)))

        if t in neg_dict:
            neg_prob += math.log(neg_dict[t])
        else:
            neg_prob += math.log(1/float((neg_total_words + total_unique_words)))

        if t in neu_dict:
            neu_prob += math.log(neu_dict[t])
        else:
            neu_prob += math.log(1/float((neg_total_words + total_unique_words)))

    pos_prob += math.log(class_probabilities['class_positive_prob'])
    neg_prob += math.log(class_probabilities['class_negative_prob'])
    neu_prob += math.log(class_probabilities['class_neutral_prob'])

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

if __name__=='__main__':

    test_data = extract_data('testset.csv')
    testNBClassifier(test_data)
