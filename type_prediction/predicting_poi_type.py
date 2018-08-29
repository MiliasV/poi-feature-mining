import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# Importing Gensim
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

def read_table_as_df(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    return pd.read_sql_query("select * from {tab}".format(tab=table), con=conn)


def gradient_boost(features_train, labels_train, variables):
    params = {'n_estimators': variables["n_estimators"], 'max_depth': variables["max_depth"],
              'min_samples_split': variables["min_samples_split"],
              'learning_rate': variables["learning_rate"], 'loss': variables["loss"]}
    gb_classifier = GradientBoostingClassifier(**params)
    gb_classifier.fit(features_train, labels_train)
    return gb_classifier


if __name__ == '__main__':
    #sf_df = read_table_as_df("matched_scene_features_ams")
    #print(list(sf_df))
    od_oid_df = read_table_as_df("matched_agg_od_oid_ams")
    print(od_oid_df.shape)
    # convert point column to numeric to merge
    od_oid_df["point"] = od_oid_df["point"].astype("int")
    # convert true, false to numeric
    od_oid_df = od_oid_df * 1
    # twitter features
    tf_df = read_table_as_df("matched_text_features_10_25_ams")
    print(tf_df.shape)
    # review features
    rf_df = read_table_as_df("matched_review_features_10_25_ams")
    print(tf_df.shape)

    # google-fsq features
    gf_df = read_table_as_df("matched_gf_features_ams")
    print(list(gf_df))
    gf_df = gf_df.drop(["id", "name", "type", "gid", "fid", "Monday", "Tuesday",
                        "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "fprice"], axis=1)
    gf_df = gf_df.fillna(-1)
    # dfs to merge
    dfs = [rf_df, tf_df, gf_df, od_oid_df]
    # merge by reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='point'), dfs)
    print(df.shape)

    # df = df.drop(["id_x", "name_x", "lat_x", "lng_x", "type_y", "id_y", "name_y", "lat_y", "lng_y",
    #                "timediffavg", "timediffmedian"], axis=1)
    df = df.drop(["id_x", "name_x", "lat_x", "lng_x", "type", "type_y", "id_y", "name_y", "lat_y", "lng_y",
                  "timediffavg", "timediffmedian", "placesid", "panosid", "lat", "lng", "year"], axis=1)
    print(df.shape)

    # for i in list(df):
    #     print(i)
    #
    # # print(df["price"].unique())
    # # conv_dict = {'Cheap':1.,  'Moderate':2., 'Expensive':3., 'Very Expensive':4., None:np.nan}
    # # df = df[[ "type", "checkins", "userscount", "tipcount", "rating", "price", "likescount"]]
    # df = df[["type", "typeofenvironment", "scene1", "scene2", "scene3", "scene4"]]
    # print(df["scene1"].unique())
    # print(df["scene2"].unique())
    #
    # # df["price"] = df.price.apply(conv_dict.get)
    df = df.fillna(-1)
    # #df = df[df["type"] != "nightclub"]
    train, test = train_test_split(df, test_size=0.2)
    # train = train.groupby("type")
    print(train.type_x.value_counts())
    print(test.type_x.value_counts())

    #train = train.apply(lambda x: x.sample(train.size().min()).reset_index(drop=True))
    # grouped_df = train.groupby('type')
    # print(train.type.value_counts())
    trainlabel = pd.DataFrame(train.type_x, columns=["type_x"])
    testlabel = pd.DataFrame(test.type_x, columns=["type_x"])
    train = train.drop(["type_x"], axis=1)
    #
    # print(test.groupby("type").type.value_counts())
    #
    test = test.drop(["type_x"], axis=1)
    # # random forest
    #clf = RandomForestClassifier()
    clf = LinearDiscriminantAnalysis(solver='lsqr', n_components=10)
    #clf = GaussianNB()
    #clf = svm.SVC()
    #clf = linear_model.SGDClassifier()
    #clf = KNeighborsClassifier(n_neighbors=10)
    #estimators = 100
    # loss = "exponential"
    # learning = 0.1
    # params = {'n_estimators': estimators, 'max_depth': 3, 'min_samples_split': 2,
    #             'learning_rate': learning, 'loss': loss}
    #
    clf.fit(train, trainlabel)
    print(clf.classes_)
    print(clf.score(test, testlabel))
    pred = clf.predict(test)
    #print(confusion_matrix(testlabel, pred, clf.classes_))

