import pandas
import numpy
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn import cross_validation

catFeatures = 116
contFeatures = 15
trainingData = None
testData = None
TRAIN_VAL_DATA = 0
TEST_DATA = 1
uniqueValues = []
PATH_TO_TRAIN_FILE = './data/train.csv'
PATH_TO_TEST_FILE = './data/test.csv'

def GetUniqueValues(columns):
    global trainingData
    global testData
    global uniqueValues

    for i in range(catFeatures):
        train = trainingData[columns[i]].unique()
        test = testData[columns[i]].unique()
        uniqueValues.append(list(set(train) | set(test)))
    
def Preprocess(data, preprocess_type=None):
    global trainingData
    global testData
    # You can add your own data preprocessing here, if any
    # input is pandas data frame
    # return type must be numpy array
    
    if (preprocess_type == "onehot"):
        if(len(uniqueValues) == 0):
            GetUniqueValues(data.columns)

        encodedCats = []
        for i in range(catFeatures):
            labelEncoder = LabelEncoder().fit(uniqueValues[i])
            encodedValues = labelEncoder.transform(data.iloc[:,i])
            encodedValues = encodedValues.reshape(data.shape[0], 1) # TODO: what is this?
            onehotEncoder = OneHotEncoder(sparse=False, n_values=len(uniqueValues[i]))
            encodedValues = onehotEncoder.fit_transform(encodedValues)
            encodedCats.append(encodedValues)

        # make a 2D array from a list of 1D arrays. TODO: what is this?
        encodedCats = numpy.column_stack(encodedCats)

        # concatenate continuous features into encoded features
        data = numpy.concatenate((encodedCats, data.iloc[:,catFeatures:].values), axis=1)
        del encodedCats
    else:
        cat_features = [x for x in data.select_dtypes(include=['object']).columns if x not in ['id', 'loss']]

        for c in range(len(cat_features)):
            data[cat_features[c]] = data[cat_features[c]].astype('category').cat.codes

        data = data.as_matrix()

    return data

def GetData(preprocess=None, NFOLDS=5):
    global trainingData
    global testData
    # Reading data
    # 116 categorical features, 14 continuous features, 1 target (loss) - continuous
    trainingData = pandas.read_csv(PATH_TO_TRAIN_FILE)
    trainingData.drop("id", axis=1, inplace=True)

    testData = pandas.read_csv(PATH_TO_TEST_FILE)

    testIDs = testData["id"]
    testData.drop("id", axis=1, inplace=True)

    if(preprocess):
        # loss distribution must be converted to normal, TODO: why?
        trainingData["loss"] = numpy.log1p(trainingData["loss"])

    # TODO: correlation: might be able to reduce number of (continuous) features

    if(preprocess):
        trainingData = Preprocess(trainingData, preprocess_type=preprocess)
        testData = Preprocess(testData, preprocess_type=preprocess)
    else:
        trainingData = trainingData.as_matrix()
        testData = testData.as_matrix()

    # data splitting
    row, col = trainingData.shape

    X = trainingData[:,:col-1]
    Y = trainingData[:,col-1]
    seed = 0
    kf = cross_validation.KFold(n=trainingData.shape[0], n_folds=NFOLDS, shuffle=True, random_state=seed)
    del trainingData

    # return XTrain, XVal, YTrain, YVal, kf, testData, testIDs
    return X, Y, kf, testData, testIDs

def GetMAE(expected, predicted):
    return mean_absolute_error(expected, predicted) 

# workflow
# 1. Get train and validation data
# 2. Train the model and predict with your own model
# 3. Get the MAE

# 1.
X, Y, kf, XTest, IDs = GetData("onehot", 5)
print(len(X), len(Y))
for i, (train, val) in enumerate(kf):
    print(train, val)
    XTrain, XVal = X[train], X[val]
    YTrain, YVal = Y[train], Y[val]
    
    # 2. Your model here...

    # 3. Get MAE
    # print(GetMAE(YVal, yourPredictedValues))
    # average out the MAEs across the 5 folds

    # to avoid out of memory exception
    del XTrain
    del XVal
    del YTrain
    del YVal

