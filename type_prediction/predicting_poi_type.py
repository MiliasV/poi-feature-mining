import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

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


def svmc(variables):
    clf_svm = svm.SVC(C=variables["C"], kernel=variables["kernel"], degree=variables["degree"],
                      probability=variables["probability"], shrinking=variables["shrinking"],
                      class_weight=variables["class_weight"], gamma=variables["gamma"],
                       decision_function_shape=variables["decision_function_shape"])
    return clf_svm


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
                   "mv": majority_voting,
                   "svm": svmc}
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


def get_best_parameters_for_clf(clf_name, train, trainlabel):
    le = LabelEncoder()
    le.fit(trainlabel)
    trainlabel = le.transform(trainlabel)
    if clf_name == "rf":
        # Create the random grid
        random_grid = {'n_estimators':[int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'bootstrap': [True, False]}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        n_iter = 100
        clf = RandomForestClassifier()
    elif clf_name == "svm":
        # [0.01, 0.1, 1, 10, 100]
        # linear
        n_iter = 32
        random_grid = {'C': [100], 'kernel': ['rbf'], "shrinking":[True, False],
                       "class_weight": ["balanced", None], "decision_function_shape":["ov", "ovr"],
                       "gamma": [0.01, 0.1, 1, 10, 100]}
        clf = svm.SVC()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=n_iter, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    clf_random.fit(train, trainlabel)
    print(clf_random.best_params_)
    return clf_random


def reduce_dimensions(dim, n_comp, train, trainlabel=None):
    if dim == "pca":
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        return pca
    elif dim == "lda":
        lda = LinearDiscriminantAnalysis(n_components=250)
        lda.fit(train, trainlabel)
        return lda


def visualize_clf_accuracy(results, names, num):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    # notch shape box plot
    bplot2 = axes.boxplot(results,
                          notch=False,  # notch shape
                          vert=True,  # vertical box alignment
                          patch_artist=True,  # fill with color
                          labels=names)  # will be used to label x-ticks
    axes.set_title('Classifiers comparison')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'red', 'grey', 'yellow', 'black', 'purple', 'brown']
    colors = colors[0:num]

    for patch, cl in zip(bplot2["boxes"], colors):
        patch.set(facecolor=cl)
    # adding horizontal grid lines
    # for ax in axes:
    axes.yaxis.grid(True)
    axes.set_xlabel('Classifiers')
    axes.set_ylabel('Accuracy')
    plt.show()


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
    red_dim = False
    tt_split = 0.2
    balance_min_class = False
    smote = False
    smote_kind = 'regular'#'svm' # regular
    smote_out_step = 0.5
    cross_val = False
    scaler = StandardScaler()

    # rf, lda, nb,
    if not cross_val:
        data_labels = pd.DataFrame(df.type, columns=["type"])
        data_features = df.drop(["type"], axis=1)
        train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
                                                               test_size=tt_split, random_state=43)
    else:
        if smote:
            data_labels = pd.DataFrame(df.type, columns=["type"])
            data_features = df.drop(["type"], axis=1)
            train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
                                                                  test_size=tt_split , random_state=43)
            train = scaler.fit_transform(train)
            test = scaler.fit_transform(test)
        else:
            trainlabel = pd.DataFrame(df.type, columns=["type"])
            train = df.drop(["type"], axis=1)
            train = scaler.fit_transform(train)
    if smote:
        print("Oversampling....")
        sm = SMOTE(kind=smote_kind, out_step=smote_out_step)#random_state=42)#, kind="svm")
        train, trainlabel = sm.fit_sample(train, trainlabel)
    print("Done!")

    if red_dim:
        print("Dimensionality Reduction!")
        dim = reduce_dimensions("pca", 200, train)
        train = dim.transform(train)
        test = dim.transform(test)

    print('Resampled dataset shape {}'.format(Counter(trainlabel)))
    #print("Getting Best Parameters....")
    #get_best_parameters_for_clf("svm", train, trainlabel)

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
    # SVM
    svm_var = {"C": 100, "kernel": "rbf", "degree": 3, "probability": False, "shrinking": False,
               "class_weight": None, "gamma": "auto", "decision_function_shape": "ovr"}
    # Majority Voting
    mv_var = [("rf", get_classifier("rf", rf_var)),
              ("gb", get_classifier("gb",  gb_var)),
              ("lda", get_classifier("lda",  lda_var))]

    models = []
    # models.append(('SVM', get_classifier("svm", svm_var)))
    models.append(('ENS', get_classifier("mv", mv_var)))
    models.append(('RF', get_classifier("rf", rf_var)))
    models.append(('LDA', get_classifier("lda", lda_var)))
    # models.append(('KNN', get_classifier("knn", knn_var)))
    # models.append(('NB', get_classifier("nb", {})))
    models.append(('gb', get_classifier("gb", gb_var)))
    scoring = "accuracy"
    results = []
    names = []
    #clf_model = get_classifier("rf", rf_var)

    # clf_svm = svm.SVC()
    # #clf_sgd = linear_model.SGDClassifier() #
    print("Training....")
    if cross_val:
        nsplits = 5
        for name, clf_model in models:
            print("Classifier: ", name)
            # test_pred = cross_val_predict(clf_model, train, trainlabel,
            #                               cv=StratifiedKFold(n_splits=10, shuffle=True))
            #testlabel = trainlabel
            scores = cross_val_score(clf_model, train, trainlabel,
                                     cv=StratifiedKFold(n_splits=nsplits, shuffle=True),
                                     scoring=scoring)
            print("Accuracy: {m} (+/- {s}".format(m=scores.mean(), s=scores.std() * 2))
            results.append(scores)
            print(scores)
            names.append(name)
            msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
            print(msg)
    else:
        print("Oversampling....")
        for name, clf_model in models:
            res = []
            for i in range(5):
                data_labels = pd.DataFrame(df.type, columns=["type"])
                data_features = df.drop(["type"], axis=1)
                train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
                                                                      test_size=tt_split)#, random_state=43)
                sm = SMOTE(kind=smote_kind, out_step=smote_out_step)  # random_state=42)#, kind="svm")
                train, trainlabel = sm.fit_sample(train, trainlabel)
                print("Classifier: ", name)
                clf = train_classifier(clf_model, train, trainlabel)
                test_pred = clf.predict(test)
                res.append(accuracy_score(testlabel, test_pred))
                print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
            names.append(name)
            results.append(res)
    print(results)
    print(names)
    visualize_clf_accuracy(results, names, len(models))

    print("Done...")
    # print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
    # precision, recall, fscore, support = score(testlabel, test_pred)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    # print(f1_score(testlabel, test_pred, average="macro"))
    # print(precision_score(testlabel, test_pred, average="macro"))
    # print(recall_score(testlabel, test_pred, average="macro"))

    #print(confusion_matrix(testlabel, test_pred))