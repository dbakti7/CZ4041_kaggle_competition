from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pandas
import numpy as np
import datetime
import time

def MapStringToIntOrString(table, column, toString):
    # convert values from string to integer in column
    labels = sorted(table[column].unique())
    mapping = dict()
    for i in range(len(labels)):
        if(toString):
            mapping[labels[i]] = str(i)
        else:
            mapping[labels[i]] = i
    # table = table.replace({column: mapping})
    # TODO: check why replace method yields error
    # there are same values between source and the mapping when
    # we are using string value.
    table[column] = table[column].map(mapping)
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

# app events
print("Reading app events...")
appEvents = pandas.read_csv("./data/app_events.csv")
appEvents["installed"] = appEvents.groupby(['event_id'])['is_installed'].transform('sum')
appEvents["active"] = appEvents.groupby(['event_id'])['is_active'].transform('sum')
appEvents.drop(['is_installed', 'is_active'], axis=1, inplace=True)
appEvents.drop_duplicates('event_id', keep='first', inplace=True)
appEvents.drop(['app_id'], axis=1, inplace=True)

# events
print("Reading events...")
events = pandas.read_csv('./data/events.csv', dtype={'device_id': np.str})
events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')

events = pandas.merge(events, appEvents, how='left', on='event_id', left_index=True)
events = events[['device_id', 'counts', 'installed', 'active']].drop_duplicates('device_id', keep='first')

# brands
# read from csv to Pandas DataFrame
print("Reading phone brands data...")
phoneBrands = pandas.read_csv("./data/phone_brand_device_model.csv", dtype={'device_id': np.str})
phoneBrands.drop_duplicates('device_id', keep='first', inplace = True)
phoneBrands = MapStringToIntOrString(phoneBrands, 'phone_brand', True)
phoneBrands = MapStringToIntOrString(phoneBrands, 'device_model', True)
phoneBrands = pandas.get_dummies(phoneBrands, columns=['phone_brand', 'device_model'])
# print(phoneBrands.head())

# training set
print("Reading training data...")
trainData = pandas.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
trainData = MapStringToIntOrString(trainData, 'group', False)
trainData = trainData.drop(['age'], axis=1)
trainData = trainData.drop(['gender'], axis = 1)
trainData = pandas.merge(trainData, phoneBrands, how='left', on='device_id', left_index=True)
trainData = pandas.merge(trainData, events, how='left', on='device_id', left_index=True)
trainLabel = trainData['group']
trainData = trainData.drop(['group'], axis=1)
trainData = trainData.drop(['device_id'], axis=1)
trainData.fillna(-1, inplace=True)

# test set
print("Reading test data...")
testData = pandas.read_csv("./data/gender_age_test.csv", dtype={'device_id': np.str})
testData = pandas.merge(testData, phoneBrands, how='left', on='device_id', left_index=True)
testData = pandas.merge(testData, events, how='left', on='device_id', left_index=True)
testData.fillna(-1, inplace=True)

print("Building model...")
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(trainData, trainLabel)
# print(clf.classes_)
print("Predicting...")

CreateSubmissionFile(testData, clf.predict_proba(testData.drop(['device_id'], axis=1)))