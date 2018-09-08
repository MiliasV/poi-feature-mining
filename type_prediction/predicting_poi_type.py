import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sqlalchemy import create_engine

from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


def read_table_as_df(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    return pd.read_sql_query("select * from {tab}".format(tab=table), con=conn)


def majority_voting(variables):
    # classifiers_list = []
    # for classifier_name in variables:
    #     model = variables[classifier_name]
    #     classifiers_list.append((classifier_name, model))
    mv_classifier = VotingClassifier(estimators=variables, voting="hard")
    return mv_classifier


def naive_bayes(variables):
    nb = GaussianNB()
    return nb


def gradient_boost(variables):
    params = {'n_estimators': variables["n_estimators"], 'max_depth': variables["max_depth"],
              'min_samples_split': variables["min_samples_split"],
              'learning_rate': variables["learning_rate"], 'loss': variables["loss"]}
    gb_classifier = GradientBoostingClassifier(**params)
    return gb_classifier


def random_forest(variables):
    n_est = variables["n_estimators"]
    md = variables["max_depth"]
    mf = variables["max_features"]
    mss = variables["min_samples_split"]
    msl = variables["min_samples_leaf"]
    crit = variables["criterion"] #"entropy",
    nj = variables["n_jobs"]
    os=variables["oob_score"]
    bs = variables["bootstrap"]
    cw = variables["class_weight"]
    rf_classifier = RandomForestClassifier(n_estimators=n_est, max_depth=md, max_features=mf,
                                           min_samples_split=mss, min_samples_leaf=msl,
                                           criterion=crit, n_jobs=nj, oob_score=os, bootstrap=bs,
                                           class_weight=cw)
    return rf_classifier


def decision_tree(variables):
    dt_classifier = tree.DecisionTreeClassifier()
    return dt_classifier


def knn(variables):
    k = variables["k"]
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    return knn_classifier


def lda(variables):
    solv = variables["solver"]
    nc = variables["n_components"]
    lda_classifier = LinearDiscriminantAnalysis(solver=solv, n_components=nc)
    return lda_classifier


def train_classifier(clf, features_train, labels_train):
    clf.fit(features_train, labels_train)
    return clf


def get_classifier(name_classifier, variables):
    classifiers = {"knn": knn,
                   "rf": random_forest,
                   "gb": gradient_boost,
                   "nb": naive_bayes,
                   "dt": decision_tree,
                   "lda": lda,
                   "mv": majority_voting}
    return classifiers[name_classifier](variables)


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


def get_best_parameters_for_rf(train, trainlabel):
    le = LabelEncoder()
    le.fit(trainlabel)
    trainlabel = le.transform(trainlabel)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train, trainlabel)
    print(rf_random.best_params_)
    return rf_random


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

    #coco
    od_coco_df = read_table_as_df("matched_agg_od_coco_ams")
    od_coco_df = preprocess_df(od_coco_df, "oid")
    # twitter features
    tf_df = read_table_as_df("matched_text_features_10_25_ams")
    tf_df = preprocess_df(tf_df, "tf")
    # review features
    rf_df = read_table_as_df("matched_review_features_10_25_ams")
    rf_df = preprocess_df(rf_df, "rf")
    # google-fsq features
    gf_df = read_table_as_df("matched_gf_features_ams")
    gf_df = preprocess_df(gf_df, "gf")
    ##################
    # merge the data #
    ##################
    # dfs to merge gf_df sf_df,
    # tf_df, gf_df, rf_df, od_oid_df, sf_df
    dfs = [tf_df, gf_df, rf_df, od_oid_df, sf_df, od_coco_df]
    dfs = keep_one_type_col(dfs)
    # merge by reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='point'), dfs)
    # drop not needed cols
    df = df.fillna(-1)
    print(df.type.value_counts())
    #df = df[df["type"]!="restaurant"]

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
    smote = False
    smote_kind = 'regular'#'svm' # regular
    smote_out_step = 0.5
    cross_val = True
    # rf, lda, nb,
    if not cross_val:
        # trainlabel = pd.DataFrame(train.type, columns=["type"])
        # testlabel = pd.DataFrame(test.type, columns=["type"])
        # train = train.drop(["type"], axforest soundsis=1)
        # test = test.drop(["type"], axis=1)

        data_labels = pd.DataFrame(df.type, columns=["type"])
        data_features = df.drop(["type"], axis=1)

        train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
                                                               test_size=tt_split, random_state=43)

        print("####################################################")
        # balance data - keep minimum number per class
        # train = train.groupby("type")
        # train = train.apply(lambda x: x.sample(train.size().min()).reset_index(drop=True))
        #
        # print("TRAIN Data")
        # print(train.type)
        # print(train.type.value_counts())
        # print("####################################################")
        # print("TEST Data")
        # print(test.type.value_counts())

        # grouped_df = train.groupby('type')
        #print(train.type_x.value_counts())


        # Dimensionality Reduction
        # lda = LinearDiscriminantAnalysis(n_components=250)
        # lda.fit(train, trainlabel)
        # train = lda.transform(train)
        # test = lda.transform(test)
        # pca = PCA(n_components=370)
        # pca.fit(train)
        # train = pca.transform(train)
        # test = pca.transform(test)
    else:
        # data_labels = pd.DataFrame(df.type, columns=["type"])
        # data_features = df.drop(["type"], axis=1)
        #
        # train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
        #                                                       test_size=tt_split , random_state=43)
        trainlabel = pd.DataFrame(df.type, columns=["type"])
        train = df.drop(["type"], axis=1)
        # print(list(train))
        # print(list(trainlabel))
    print("Donee")
    if smote:    # pca = PCA(n_components=370)
        print("Oversampling....")
        sm = SMOTE(kind=smote_kind, out_step=smote_out_step)#random_state=42)#, kind="svm")
        train, trainlabel = sm.fit_sample(train, trainlabel)
    print("Done!")
    print("Training classifier")
    # pca = PCA(n_components=200)
    # pca.fit(train)
    # train = pca.transform(train)
    # test = pca.transform(test)

    print('Resampled dataset shape {}'.format(Counter(trainlabel)))
    #r_random = get_best_parameters_for_rf(train, trainlabel)
    #print(r_random)
    #print(trainlabel.head())

    # Random Forest - set parameters
    # not used -  random_state=42, class_weight="balanced"
    rf_var = {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": False, "bootstrap":False,
              "class_weight": "balanced"}
    # KNN - set parameters
    knn_var = {"k": 10}

    # Gradient Boosting
    gb_var = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': 0.1, 'loss': "deviance"}
    # LDA
    lda_var = {"solver": "svd", "n_components": 200}
    # Majority Voting
    # mv_var = [("rf", get_classifier("rf", rf_var)),
    #           ("knn", get_classifier("knn", knn_var)),
    #           ("gb", get_classifier("gb",  gb_var)),
    #           ("lda", get_classifier("lda",  lda_var))]

    clf_model = get_classifier("rf", rf_var)

    # clf_svm = svm.SVC()
    # #clf_sgd = linear_model.SGDClassifier() #
    print("Training....")
    if cross_val:
        test_pred = cross_val_predict(clf_model, train, trainlabel,
                                      cv=StratifiedKFold(n_splits=10, shuffle=True))
        testlabel = trainlabel
        scores = cross_val_score(clf_model, train, trainlabel, cv=StratifiedKFold(n_splits=10, shuffle=True))
        print("Accuracy: {m} (+/- {s}".format(m=scores.mean(), s=scores.std() * 2))
        #conf_mat = confusion_matrix(y, y_pred)
        #scores = cross_val_score(clf, train, trainlabel, cv=5)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    else:
        clf = train_classifier(clf_model, train, trainlabel)
        #print(clf.classes_)
        test_pred = clf.predict(test)
    print("Done...")
    print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
    precision, recall, fscore, support = score(testlabel, test_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    # print(f1_score(testlabel, test_pred, average="macro"))
    # print(precision_score(testlabel, test_pred, average="macro"))
    # print(recall_score(testlabel, test_pred, average="macro"))

    print(confusion_matrix(testlabel, test_pred))