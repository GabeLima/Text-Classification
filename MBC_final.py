import UB_BB
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # GaussianNB
from sklearn.feature_extraction import text


def naiveBayes(trainingData, labels):
    nb = MultinomialNB(alpha=0.25)
    nb.fit(trainingData, labels)
    return nb


def nGram(gramTuple):
    trainingData = UB_BB.getString(trainSet) # String representation of all the training files in the format[0,1,2,3]
    my_stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(ngram_range=gramTuple, stop_words = my_stop_words, strip_accents="ascii", max_df=.8)  # Get the count vectorizer
    unigramTrainCounts = vectorizer.fit_transform(trainingData) # Fit the vectorizer with the training string
    testingData = UB_BB.getString(evalSet) # String representation of the evaluation data for testing
    evalCounts = vectorizer.transform(testingData)

    trainNB = naiveBayes(unigramTrainCounts, labels)
    algorithmArray = [trainNB]

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
    algOutput = ["NB"]
    UBoutput = []
    baseline = "UB"
    config = ["(Multinomial, Alpha=.25, Stopwords, Strip Accents Ascii, max_df=.8)"]
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