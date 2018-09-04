import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sqlalchemy import create_engine

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE


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


def one_hot_conversion(df, cols):
    label_encoder = LabelEncoder()
    df[[cols]] = df[[cols]].apply(label_encoder)
    return df
    # # binary encode
    # onehot_encoder = OneHotEncoder(sparse=False)


def show_correlations(df):
    # plt.matshow(df.corr())
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()


def preprocess_df(df, label):
    if label == "gf":
        df = df.drop(["id", "name", "gid", "fid", "Monday", "Tuesday",
                            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "fprice"], axis=1)
        df = df.fillna(-1)
    elif label == "oid":
        df = df.drop(["placesid", "panosid", "lat", "lng", "year"], axis=1)
        # convert point column to numeric to merge
        df["point"] = df["point"].astype("int")
        # convert true, false to numeric
        df = df * 1
    elif label == "tf":
        df = df.drop(["id", "name", "lat", "lng", "id", "name", "lat", "lng",
                 "timediffavg", "timediffmedian"], axis=1)
    elif label == "rf":
        df = df.drop(["id", "name", "lat", "lng"], axis=1)
    return df


def keep_one_type_col(dfs):
    for i in range(len(dfs)-1):
        dfs[i] = dfs[i].drop(["type"], axis=1)
    return dfs


if __name__ == '__main__':
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    #################
    # read the data #
    #################
    # scene features
    sf_df = read_table_as_df("matched_agg_scene_features_ams")
    # object detection features
    od_oid_df = read_table_as_df("matched_agg_od_oid_ams")
    od_oid_df = preprocess_df(od_oid_df, "oid")
    # twitter features
    tf_df = read_table_as_df("matched_text_features_10_25_ams")
    tf_df = preprocess_df(tf_df, "tf")

    # review features
    rf_df = read_table_as_df("matched_review_features_10_25_ams")
    rf_df = preprocess_df(rf_df, "rf")

    # google-fsq features
    gf_df = read_table_as_df("matched_gf_features_ams")
    gf_df = preprocess_df(gf_df, "gf")

    # dfs to merge gf_df sf_df,
    # tf_df, gf_df, rf_df, od_oid_df, sf_df
    dfs = [gf_df, tf_df, rf_df, sf_df, od_oid_df]
    dfs = keep_one_type_col(dfs)
    # merge by reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='point'), dfs)
    # drop not needed cols
    df = df.fillna(-1)
    # store the df
    #df.to_sql('matched_data_features_ams', db, index=False)
    print(list(df))
    print(df.shape)
    #show_correlations(df)

    ###############
    # CLASSIFYING #
    ###############
    tt_split = 0.2
    balance_min_class = False
    smote = True
    train, test = train_test_split(df, test_size=tt_split)#, random_state=43)
    print("####################################################")
    # balance data - keep minimum number per class
    # train = train.groupby("type")
    # train = train.apply(lambda x: x.sample(train.size().min()).reset_index(drop=True))

    print("TRAIN Data")
    print(train.type)
    print(train.type.value_counts())
    print("####################################################")
    print("TEST Data")
    print(test.type.value_counts())

    # grouped_df = train.groupby('type')
    #print(train.type_x.value_counts())

    trainlabel = pd.DataFrame(train.type, columns=["type"])
    testlabel = pd.DataFrame(test.type, columns=["type"])
    train = train.drop(["type"], axis=1)
    test = test.drop(["type"], axis=1)

    print("Oversampling....")
    sm = SMOTE(kind='svm', out_step=0.2)#random_state=42)#, kind="svm")
    train, trainlabel = sm.fit_sample(train, trainlabel)
    print("Done!")
    print("Train Data after SMOTE")
    print('Resampled dataset shape {}'.format(Counter(trainlabel)))

    clf = RandomForestClassifier(n_estimators=100, max_depth=80, max_features=0.9,
                                 criterion="gini", n_jobs=-1, oob_score=True)#, random_state=42)

        #,class_weight="balanced") #, random_state=10)
    # clf_rf2 = RandomForestClassifier(n_estimators=150, max_depth=70, max_features=0.9,
    #                                 criterion="entropy", n_jobs=-1, oob_score=True)
    # clf_rf3 = RandomForestClassifier(n_estimators=150, max_depth=30, max_features=0.9,
    #                                 criterion="gini", n_jobs=-1, oob_score=True)
    # clf_lda = LinearDiscriminantAnalysis(solver='svd', n_components=150)
    # # # Create adaboost-decision tree classifer object
    # clf_ab = AdaBoostClassifier(n_estimators=1000,
    #                           learning_rate=0.03,
    #                           random_state=0)
    # clf_gb = GaussianNB()
    # clf_svm = svm.SVC()
    # #clf_sgd = linear_model.SGDClassifier()
    # clf_nn = KNeighborsClassifier(n_neighbors=10)
    #estimators = 100
    # loss = "exponential"
    # learning = 0.1
    # params = {'n_estimators': estimators, 'max_depth': 3, 'min_samples_split': 2,
    #             'learning_rate': learning, 'loss': loss}
    #
    # clf = VotingClassifier(estimators=[("lda", clf_rf3), ("rf", clf_rf1),("rf2", clf_rf2)], voting="hard")
    # #                                   ("ab", clf_ab)], voting="hard")#, ("gb", clf_gb),
    # # #                                    #("svm", clf_svm), ("nn", clf_nn)], voting="hard")
    print("Training....")
    clf.fit(train, trainlabel)
    print("Done...")
    print(clf.classes_)
    print(clf.score(test, testlabel))
    pred = clf.predict(test)
    #print(clf.n_features_)
    #print(clf.feature_importances_)
    #print(confusion_matrix(testlabel, pred, clf.classes_))

