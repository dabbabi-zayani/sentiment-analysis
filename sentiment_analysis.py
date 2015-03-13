__author__ = 'Sanjeev K C'

from TestingData import *

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

    pos_prob += math.log(prior_probabilities['class_positive_prob'])
    neg_prob += math.log(prior_probabilities['class_negative_prob'])
    neu_prob += math.log(prior_probabilities['class_neutral_prob'])

    max_prob = max(pos_prob,neg_prob,neu_prob)

    if max_prob == pos_prob:
        print 'positive'
    elif max_prob == neg_prob:
        print 'negative'
    else:
        print 'neutral'

if __name__=='__main__':

    print "Twitter data sets: \n"
    train_data = extract_data_csv_file('train.csv')
    trainNBClassifier(train_data)

    test_data = extract_data_csv_file('test.csv')
    testNBClassifier(test_data)

    print "IMDb data sets: \n"
    train_data = extract_data_txt_file('C:/Users/schittia/Desktop/aclImdb/train')
    trainNBClassifier(train_data)

    test_data = extract_data_txt_file('C:/Users/schittia/Desktop/aclImdb/test')
    testNBClassifier(test_data)

    _get_sentiment()
