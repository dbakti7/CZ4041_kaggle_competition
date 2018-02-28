from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pandas
import numpy as np
import datetime
import time

def MapStringToInt(table, column):
    # convert values from string to integer in column
    labels = sorted(table[column].unique())
    mapping = dict()
    for i in range(len(labels)):
        mapping[labels[i]] = i
    table = table.replace({column: mapping})
    return table

def CreateSubmissionFile(testData, prediction):
    now = datetime.datetime.now()
    fileName = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Writing to submission file...")
    f = open(fileName, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    testIDs = testData['device_id'].values
    total = 0
    for i in range(len(testIDs)):
        line = str(testIDs[i])
        for j in range(12):
            line += ',' + str(prediction[i][j])
        line += '\n'
        f.write(line)
        total += 1
    f.close()
    print("Predicted: " + str(total) + " lines.")

# brands
# read from csv to Pandas DataFrame
print("Reading phone brands data...")
phoneBrands = pandas.read_csv("./data/phone_brand_device_model.csv", dtype={'device_id': np.str})
phoneBrands.drop_duplicates('device_id', keep='first', inplace = True)
phoneBrands = MapStringToInt(phoneBrands, 'phone_brand')
phoneBrands = MapStringToInt(phoneBrands, 'device_model')
# print(phoneBrands.head())

# trainig set
print("Reading training data...")
trainData = pandas.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
trainData = MapStringToInt(trainData, 'group')
trainData = trainData.drop(['age'], axis=1)
trainData = trainData.drop(['gender'], axis = 1)
trainData = pandas.merge(trainData, phoneBrands, how='left', on='device_id', left_index=True)
trainLabel = trainData['group']
trainData = trainData.drop(['group'], axis=1)

# test set
print("Reading test data...")
testData = pandas.read_csv("./data/gender_age_test.csv", dtype={'device_id': np.str})
testData = pandas.merge(testData, phoneBrands, how='left', on='device_id', left_index=True)

print("Building model...")
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(trainData, trainLabel)
# print(clf.classes_)
print("Predicting...")

CreateSubmissionFile(testData, clf.predict_proba(testData))