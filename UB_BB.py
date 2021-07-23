import os
import sys
import sklearn.ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # GaussianNB
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

global gTrainingData
global gEvaluationData
global gQueries
global gAnswers
global gVectorizer


# Returns the trained Multinomial NB
def naiveBayes(trainingData, labels):
    nb = MultinomialNB()
    nb.fit(trainingData, labels)
    return nb


# Returns the trained logistic regression model that we can use to predict on the evaluation set
def logReg(trainingData, labels):
    skLogReg = LogisticRegression(max_iter=1000)
    skLogReg.fit(trainingData, labels)
    return skLogReg


def svm(trainingData, labels):
    # sv= make_pipeline(StandardScaler(),
    #                     LinearSVC(random_state=0, tol=1e-5))
    sv = LinearSVC()
    sv.decision_function_shape = "ovr" # Not sure if we should be doing this
    sv.fit(trainingData, labels)
    return sv


def randomForest(trainingData, labels):
    # rf = RandomForestClassifier(n_estimators=len(labels) * len(labels), max_features="log2") # A tree per label^2?
    rf = RandomForestClassifier(n_estimators=len(labels) * len(labels), max_features=len(labels) * len(labels))
    rf.fit(trainingData, labels)
    return rf


def getString(trainSet):
    subfolders = [f.path for f in os.scandir(trainSet) if f.is_dir()]
    totalFiles = []
    # print(subfolders)
    for x in range(len(subfolders)):
        totalFiles.append("")
        subSubfolders = [f.path for f in os.scandir(subfolders[x])]
        #print(subSubfolders)
        for txtFiles in subSubfolders:
            totalFiles[x] += (skipHeader(txtFiles))
    return totalFiles


def genTestCases(evalSet, labels):
    answers = []
    testCases = []
    # Lets do getString but seperated
    subfolders = [f.path for f in os.scandir(evalSet) if f.is_dir()]
    for x in range(len(subfolders)):
        testCases.append([])
        subSubfolders = [f.path for f in os.scandir(subfolders[x])]
        for txtFiles in subSubfolders:
            testCases[x].append(skipHeader(txtFiles))
            answers.append(labels[x])
    return testCases, answers  # Should be an array of (#classes) arrays where each index is a test case


def skipHeader(txtFile):
    temp = open(txtFile).read()
    x = temp.find("Lines: ")
    if x != -1:
        temp = temp[x:]
    stemmedTemp = ""
    porter = PorterStemmer()
    sentence = temp.split(' ')
    for word in sentence:
        stemmedTemp += porter.stem(word) + " "
    return stemmedTemp


def generatePredictions(queries, algorithm):
    predictionArray = []
    for query in queries:
        predictionArray.append(algorithm.predict(query)) # query.todense()
    return predictionArray


def calculatePrecision(truths, preds):
    # print("Truths: ", len(truths))
    #print(truths)
    # print("Preds: ", len(preds))
    #print(preds)
    return sklearn.metrics.precision_score(truths, preds, average='micro') # hm, why do i need this here? ##############


def calculateRecall(truths, preds):
    return sklearn.metrics.recall_score(truths, preds, average='micro')


def calculateF1(truths, preds):
    return sklearn.metrics.f1_score(truths, preds, average='micro')


def tripleScore(truths, preds): # Truth remains the same, preds vary from algo to algo
    temp = ""
    temp += str(round(calculatePrecision(truths, preds), 2))
    temp += "," + str(round(calculateRecall(truths, preds), 2))
    temp += "," + str(round(calculateF1(truths, preds), 2))
    return temp


def nGram(gramTuple):
    global gTrainingData
    global gEvaluationData
    global gQueries
    global gAnswers
    global gVectorizer
    trainingData = getString(trainSet) # String representation of all the training files in the format[0,1,2,3]
    vectorizer = CountVectorizer(ngram_range=gramTuple)  # Get the count vectorizer
    unigramTrainCounts = vectorizer.fit_transform(trainingData) # Fit the vectorizer with the training string
    testingData = getString(evalSet) # String representation of the evaluation data for testing
    evalCounts = vectorizer.transform(testingData)

    trainNB = naiveBayes(unigramTrainCounts, labels)
    trainedLog = logReg(unigramTrainCounts, labels)
    trainedSVM = svm(unigramTrainCounts, labels)
    trainedRF = randomForest(unigramTrainCounts, labels)
    algorithmArray = [trainNB, trainedLog, trainedSVM, trainedRF]

    testCases = genTestCases(evalSet, labels)
    strQueries = testCases[0]
    answers = testCases[1]
    queries = []
    for category in strQueries:
        for txt in category:
            queries.append(vectorizer.transform([txt]))
    # queries = vectorizer.transform(queries)
    predictionArray = []
    for algorithm in algorithmArray:
        predictionArray.append(generatePredictions(queries, algorithm)) # Array of arrays, [nb, log, svm, rf]
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
    for x in range(len(algOutput)):
        UBoutput.append("%s,%s,%s" % (algOutput[x], baseline, tripleScore(answers, predictionArray[x])))
    return UBoutput


def manualInputSplitting(strings, percent):
    temp = []
    i = 0
    for string in strings:
        temp.append([])
        temp[i] = string[:int(percent * len(string))]
        i += 1
    return temp # should be an array of arrays with each subarray being a string the % size of the normal training data


def selfLearningCurve(algoArray, vectorizer):
    global gQueries
    global gAnswers
    global gTrainingData
    trainSizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    fScores = []
    i = 0
    for x in range(len(algoArray)):
        fScores.append([])
        for trainSize in trainSizes:
            X_train = vectorizer.transform(manualInputSplitting(gTrainingData, trainSize)) # Split data according to trainSizes
            algo = algoArray[x]
            algo.fit(X_train, labels)
            # Gotta make a prediction on the evaluation data now
            preds = generatePredictions(gQueries, algo)
            fScores[x].append(calculateF1(gAnswers, preds))
    plotLearningCurve(fScores, trainSizes)
    # At the end we have an array of arrays [[x f values based on train sizes, algo used][][][]]


def plotLearningCurve(fScores, trainSizes):
    ylim = (0, 1)
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Size of training data (%)")
    plt.ylabel("F1-Score")
    colorArray = ["r", "g", "b", "y"]
    labelArray = ["NB", "LR", "SVM", "RF"]
    plt.grid()
    # print(fScores)
    for x in range(len(fScores)):
        plt.plot(trainSizes, fScores[x], 'o-', color=colorArray[x],
                 label=labelArray[x])
    plt.legend(loc='best')
    plt.show()


def writeOutput(outputFile, uniOut, biOut):
    f = open(outputFile, "w")
    for x in range(len(uniOut)):
        f.write(uniOut[x])
        f.write("\n")
        f.write(biOut[x])
        f.write("\n")


if __name__ == '__main__':
    global gTrainingData
    global gEvaluationData
    trainSet = sys.argv[1]
    evalSet = sys.argv[2]
    output = sys.argv[3]
    displayLC = int(sys.argv[4])
    labels = os.listdir(trainSet)

    uniOut = nGram((1, 1))
    biOut = nGram((2, 2))
    writeOutput(output, uniOut, biOut)
    # print(uniOut)
    # print(biOut)

    if displayLC == 1:
        algoArray = [MultinomialNB(), LogisticRegression(), LinearSVC(), RandomForestClassifier()]
        selfLearningCurve(algoArray, gVectorizer)