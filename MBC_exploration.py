from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text

import UB_BB
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # GaussianNB


def naiveBayes(trainingData, labels):
    nb = MultinomialNB(alpha=0.25)
    nb.fit(trainingData, labels)
    return nb


# Returns the trained logistic regression model that we can use to predict on the evaluation set
def logReg(trainingData, labels):
    skLogReg = LogisticRegression(max_iter=1000, dual=True, class_weight="balanced", solver = "liblinear")
    skLogReg.fit(trainingData, labels)
    return skLogReg


def svm(trainingData, labels):
    sv = LinearSVC(class_weight="balanced", C=.01)
    sv.decision_function_shape = "ovr" # Not sure if we should be doing this
    sv.fit(trainingData, labels)
    return sv


def randomForest(trainingData, labels):
    rf = RandomForestClassifier(n_estimators=len(labels) * len(labels), class_weight="balanced", ccp_alpha=.75, random_state=len(labels))
    rf.fit(trainingData, labels)
    return rf


def nGram(gramTuple):
    global gTrainingData
    global gEvaluationData
    global gQueries
    global gAnswers
    global gVectorizer
    my_stop_words = text.ENGLISH_STOP_WORDS
    trainingData = UB_BB.getString(trainSet) # String representation of all the training files in the format[0,1,2,3]
    vectorizer = CountVectorizer(ngram_range=gramTuple, stop_words = my_stop_words, strip_accents="ascii", max_df=.8)  # Get the count vectorizer
    unigramTrainCounts = vectorizer.fit_transform(trainingData) # Fit the vectorizer with the training string
    testingData = UB_BB.getString(evalSet) # String representation of the evaluation data for testing
    evalCounts = vectorizer.transform(testingData)

    trainNB = naiveBayes(unigramTrainCounts, labels)
    trainedLog = logReg(unigramTrainCounts, labels)
    trainedSVM = svm(unigramTrainCounts, labels)
    trainedRF = randomForest(unigramTrainCounts, labels)
    algorithmArray = [trainNB, trainedLog, trainedSVM, trainedRF]

    testCases = UB_BB.genTestCases(evalSet, labels)
    strQueries = testCases[0]
    answers = testCases[1]
    queries = []
    for category in strQueries:
        for txt in category:
            queries.append(vectorizer.transform([txt]))
    predictionArray = []
    for algorithm in algorithmArray:
        predictionArray.append(UB_BB.generatePredictions(queries, algorithm)) # Array of arrays, [nb, log, svm, rf]
    # print(predictionArray)
    algOutput = ["NB", "LR", "SVM", "RF"]
    UBoutput = []
    baseline = "UB"
    if gramTuple == (2,2):
        baseline = "BB"
    else:
        gTrainingData = trainingData#unigramTrainCounts #trainingData
        gEvaluationData = predictionArray
        gQueries = queries
        gAnswers = answers
        gVectorizer = vectorizer
    config = ["(Multinomial, Alpha=.25, Stopwords, Strip Accents Ascii, max_df=.8)", "(Dual = True, solver = liblinear, classweight = "
                                                                        "'balanced', Stopwords, Strip Accents Ascii, max_df=.8)",
              "(C = .01, classweight = 'balanced', Stopwords, Strip Accents Ascii, max_df=.8)", "(class_weight='balanced', ccp_alpha=.75, Stopwords, Strip Accents Ascii, max_df=.8)"]
    for x in range(len(algOutput)):
        UBoutput.append("%s,%s,%s" % (algOutput[x],config[x], UB_BB.tripleScore(answers, predictionArray[x])))
    return UBoutput


if __name__ == '__main__':
    trainSet = sys.argv[1]
    evalSet = sys.argv[2]
    output = sys.argv[3]
    labels = os.listdir(trainSet)

    uniOut = nGram((1, 1))
    # print(uniOut)

    f = open(output, "w")
    for x in range(len(uniOut)):
        f.write(uniOut[x])
        f.write("\n")