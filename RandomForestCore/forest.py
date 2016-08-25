from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os


def test():
    return "Test"


def predict_number(keys):
    print '----'
    print os.getcwd()
    print '----'
    columns = pd.read_csv('../RandomForestCore/training.csv').columns.values
    columns = columns[columns != "DESTINATION"]
    intial_data = [0] * len(columns)
    df = pd.DataFrame([intial_data], columns=columns)

    for col in columns:
        try:
            df.set_value(0, col, keys[col])
        except KeyError:
            continue
    model = joblib.load('../RandomForestCore/dataModel/trainedModel.pkl')

    prediction = model.predict(df)
    print "prediction in predict_number: ", prediction[0]
    result = {"PREDICTION": prediction[0]}
    print "forest result in predict_number: " + str(result)
    return result


def predict_number_prob(keys):
    columns = pd.read_csv('../RandomForestCore/training.csv').columns.values
    columns = columns[columns != "DESTINATION"]
    intial_data = [0] * len(columns)
    df = pd.DataFrame([intial_data], columns=columns)

    for col in columns:
        try:
            df.set_value(0, col, keys[col])
        except KeyError:
            continue
    model = joblib.load('../RandomForestCore/dataModel/trainedModel.pkl')

    prediction = model.predict_proba(df)
    max_index = np.argmax(prediction)
    print 'max class: ' + str(model.classes_[max_index])
    print 'predictions: ' + str(prediction)
    print 'max prob: ' + str(prediction[0][max_index])
    result = {"PREDICTION": model.classes_[max_index], "PROBABILITY": prediction[0][max_index]}
    print result
    return result


def predict_number_prob_all(keys):
    columns = pd.read_csv('../RandomForestCore/training.csv').columns.values
    columns = columns[columns != "DESTINATION"]
    intial_data = [0] * len(columns)
    df = pd.DataFrame([intial_data], columns=columns)

    for col in columns:
        try:
            df.set_value(0, col, keys[col])
        except KeyError:
            continue
    model = joblib.load('../RandomForestCore/dataModel/trainedModel.pkl')

    prediction = model.predict_proba(df)
    print "predict_number_prob_all prediction:", prediction[0]
    print "predict_number_prob_all classes:", model.classes_

    df_result = pd.DataFrame({"DESTINATION": model.classes_, "PROBABILITY": prediction[0]}).sort_values(
        by=["PROBABILITY"], ascending=False, kind="mergesort")
    result_json = {"PREDICTIONS": []}
    c = 0
    for index, row in df_result.iterrows():
        result_json["PREDICTIONS"].append({"PREDICTION": row["DESTINATION"], "PROBABILITY": row["PROBABILITY"]})
        c += 1
        if c == 5:
            break
    print "predict_number_prob_all json:", result_json
    return result_json


def train(keys):
    print "In train keys: ", keys
    train_df = pd.read_csv('../RandomForestCore/training.csv').fillna(value=0)
    new_df = pd.DataFrame(data=keys, index=[1], columns=keys.keys())
    ndf = pd.concat([train_df, new_df], axis=0).fillna(value=0)
    target = ndf['DESTINATION']
    train_set = ndf.drop(['DESTINATION'], axis=1)
    # print "In train target:", target
    # print "In train target:", train
    ndf.to_csv('../RandomForestCore/training.csv', mode='w', header=True, sep=',', index=False, na_rep=0)
    regenerate_forest(train_set, target)
    print(ndf)
    return "success"


def regenerate_forest(train_set, target):
    print "regenerate_forest started"
    rf = RandomForestClassifier(n_estimators=10, n_jobs=1)
    rf.fit(train_set, target)
    joblib.dump(rf, '../RandomForestCore/dataModel/trainedModel.pkl')
    print "regenerate_forest finished"


def reset_csv():
    print "resetting csv in forest"
    df = pd.DataFrame([[0]], columns=["DESTINATION"])
    df.to_csv('../RandomForestCore/training.csv', mode='w', header=True, sep=',', index=False, na_rep=0)
    return "success"


def main():
    training_df = pd.read_csv('../RandomForestCore/training.csv').fillna(0)

    target = training_df['DESTINATION']
    print target
    print "--- target"
    train = training_df.iloc[0:, 1:]
    print train
    print "--- train"

    test = pd.read_csv('test.csv').fillna(value=0)
    print test
    # create and train the random forest
    # rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    # rf.fit(train, target)

    # joblib.dump(rf, './dataModel/trainedModel.pkl')  # Save trained classifier into a file
    clf = joblib.load('../RandomForestCore/dataModel/trainedModel.pkl')

    prediction = clf.predict(test)
    savetxt('result.csv', prediction, delimiter=',', fmt='%f')


if __name__ == "__main__":
    predict_number_prob_all(
        {"STATE": 3, "INTENTION": 2, "CAKE": 1, "VEGETABLE": 1, "MEAT": 1, "DESTINATION": 5, "FISH": 5})
# # main()
#     predict({'STATE': 2, 'INTENTION': 1})
