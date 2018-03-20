from nltk.corpus import conll2002
from itertools import chain
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
import re
from sklearn.ensemble import RandomForestClassifier
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import precision_recall_fscore_support
from lightgbm import LGBMClassifier

# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, post_tag, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    # Features for main word and neighbors
    features = [(o + 'word', word)]
    # shape of word
    shape = ''
    for i in range(len(word)):
        if re.match('[A-Z]', word[i]):
            shape = shape + 'X'
        elif re.match('[a-z]', word[i]):
            shape = shape + 'x'
        elif re.match('[0-9]', word[i]):
            shape = shape + 'd'
        else:
            shape = shape + word[i]
    features.append((o + 'shape', shape))
    # short shape of word
    short_shape = ''
    last_shape = ''
    punctuation = False
    for i in range(len(word)):
        if re.match('[A-Z]', word[i]):
            new_shape = 'X'
        elif re.match('[a-z]', word[i]):
            new_shape = 'x'
        elif re.match('[0-9]', word[i]):
            new_shape = 'd'
        else:
            punctuation = True
            new_shape = word[i]
        if punctuation or new_shape != last_shape:
            short_shape = short_shape + new_shape
        last_shape = new_shape
    features.append((o + 'short-shape', short_shape))
    features.append((o + 'pos-tag', post_tag))
    features.append((o + "is-upper", word.isupper()))
    features.append((o + "is-title", word.istitle()))
    features.append((o + "is-digit", word.isdigit()))
    features.append((o + 'hyphen', '-' in word)) # giving less score
    features.append((o + "word-lower", word.lower()))

    # Features only for the main word
    if o == '0':
        for i in range(1,5):
            if len(word) >= i:
                # prefix
                features.append(("prefix-"+str(i), word[:i]))
                # sufix
                features.append(("suffix-" + str(i), word[-i:]))
    return features
    

def word2features(sent, i, prev_sent, next_sent):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-2,-1,0,1,2]:
        features_local = []
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            post_tag = sent[i+o][1]
            featlist = getfeats(word, post_tag, o)
            # add positional features
            if o==0:
                if i+o == 0:
                    b = False
                    featlist.append(("beggining", True))
                if i+o == len(sent)-1:
                    b = False # TODO delete
                    featlist.append(("end", True))
                if len(prev_sent) == 1 and prev_sent[0][0] == '-':
                    b = False  # TODO delete
                   # featlist.append(("prev_sent_hyphen", True))
                if len(next_sent) == 1 and next_sent[0][0] == '-':
                    b = False  # TODO delete
                   # featlist.append(("next_sent_hyphen", True))
            features_local.extend(featlist)
        features.extend(features_local)
    dict_features = dict(features)
    combination_features = []
    if i > 0:
        combination_features.append(("shape:-1&0", dict_features['-1shape'] + '&' + dict_features['0shape']))
        combination_features.append(("short-shape:-1&0", dict_features['-1short-shape'] + '&' + dict_features['0short-shape']))
        if i < len(sent)-1:
            combination_features.append(("shape:-1&0&1", dict_features['-1shape'] + '&' + dict_features['0shape'] + '&' + dict_features['1shape']))
            combination_features.append(("short-shape:-1&0&1", dict_features['-1short-shape'] + '&' + dict_features['0short-shape'] + '&' + dict_features['1short-shape']))
    if i < len(sent)-1:
        combination_features.append(("shape:0&1", dict_features['0shape'] + '&' + dict_features['1shape']))
        combination_features.append(("short-shape:0&1", dict_features['0short-shape'] + '&' + dict_features['1short-shape']))

    features.extend(combination_features)
    return dict(features)


def sentencesEngineering(sentences):
    feats = []
    labels = []
    for sent_index in range(len(sentences)):
        sent = sentences[sent_index]
        if sent_index == 0:
            prev_sent = []
        else:
            prev_sent = sentences[sent_index-1]
        if sent_index == len(sentences)-1:
            next_sent = []
        else:
            next_sent = sentences[sent_index+1]
        for i in range(len(sent)):
            current_feats = word2features(sent, i, prev_sent, next_sent)
            feats.append(current_feats)
            labels.append(sent[i][-1])
    return feats, labels

if __name__ == "__main__":
    # Load the training data
    print("loading")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    print("loaded data")
    # TODO save features in files if it start getting too slow
    train_feats, train_labels = sentencesEngineering(train_sents)

    vectorizer = DictVectorizer()
    print("Done with features of training")
    X_train = vectorizer.fit_transform(train_feats)

    print("Done vectorizing")
    # TODO: play with other models
    model = Perceptron(verbose=1)
    #model = AdaBoostClassifier()
    # model = RandomForestClassifier()
    #model = LGBMClassifier()

    model.fit(X_train, train_labels)
    print("model fitted")
    # switch to test_sents for your final results
    test_feats, test_labels = sentencesEngineering(dev_sents)
    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    file_name = 'results_prefix_suffix_shape_shortShape_cobinationSShape_cobinationShape_upper_hyphen_lower_title_position__postag_o-2_2.txt'
   # file_name = 'results_prefix_suffix_shape_shortShape_o-2_2.txt'
    print("Writing to " + file_name)
    # format is: word gold pred
    with open(file_name, "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py " + file_name)






