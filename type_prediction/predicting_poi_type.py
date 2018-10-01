import pandas as pd
from pandas_ml import ConfusionMatrix
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
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict,\
    cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, \
    VotingClassifier,  AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model, svm
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def read_table_as_df(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    return pd.read_sql_query("select * from {tab}".format(tab=table), con=conn)


def read_features(scene=True, od_oid=True, od_coco=True,
                  twitter=True, review=True, gf=True):
    dfs = []
    # scene features
    if scene:
        sf_df = read_table_as_df("matched_agg_scene_features_ams")
        dfs.append(sf_df)
    # object detection features
    if od_oid:
        od_oid_df = read_table_as_df("matched_agg_od_oid_ams")
        od_oid_df = preprocess_df(od_oid_df, "oid")
        dfs.append(od_oid_df)
    #coco
    if od_coco:
        od_coco_df = read_table_as_df("matched_agg_od_coco_ams")
        od_coco_df = preprocess_df(od_coco_df, "coco")
        dfs.append(od_coco_df)
    # twitter features
    if twitter:
        tf_df = read_table_as_df("matched_text_features_10_25_ams")
        tf_df = preprocess_df(tf_df, "tf")
        dfs.append(tf_df)
    # review features
    if review:
        rf_df = read_table_as_df("matched_review_features_10_25_ams")
        rf_df = preprocess_df(rf_df, "rf")
        dfs.append(rf_df)

    # google-fsq features
    if gf:
        gf_df = read_table_as_df("matched_gf_features_ams")
        gf_df = preprocess_df(gf_df, "gf")
        dfs.append(gf_df)
    ##################
    # merge the data #
    ##################
    # dfs to merge gf_df sf_df,
    # tf_df, gf_df, rf_df, od_oid_df, sf_df
    dfs = keep_one_type_col(dfs)
    # merge by reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='point'), dfs)
    # drop not needed cols
    df = df.fillna(-1)
    return df


def majority_voting(variables):
    mv_classifier = VotingClassifier(estimators=variables, voting="soft")
    return mv_classifier


def naive_bayes(variables):
    nb = GaussianNB()
    return nb


def xgb(variables):
    return XGBClassifier()


def gradient_boost(variables):
    params = {'n_estimators': variables["n_estimators"], 'max_depth': variables["max_depth"],
              'min_samples_split': variables["min_samples_split"],
              'learning_rate': variables["learning_rate"], 'loss': variables["loss"]}
    gb_classifier = GradientBoostingClassifier(**params)
    return gb_classifier


def ada_boost(variables):
    params = {'base_estimator': variables["base_estimator"], 'n_estimators': variables["n_estimators"],'algorithm': variables["algorithm"],
              'learning_rate': variables["learning_rate"]}
    ada_classifier = AdaBoostClassifier(**params)
    return ada_classifier


def random_forest(variables):
    rf_classifier = RandomForestClassifier(n_estimators=variables["n_estimators"],
                                           max_depth=variables["max_depth"]
                                           , max_features=variables["max_features"],
                                           min_samples_split=variables["min_samples_split"],
                                           min_samples_leaf=variables["min_samples_leaf"],
                                           criterion=variables["criterion"] , n_jobs=variables["n_jobs"],
                                           oob_score=variables["oob_score"], bootstrap=variables["bootstrap"],
                                           class_weight=variables["class_weight"])
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
                   "ada": ada_boost,
                   "nb": naive_bayes,
                   "dt": decision_tree,
                   "lda": lda,
                   "mv": majority_voting,
                   "svm": svmc,
                   "xgb": xgb}
    return classifiers[name_classifier](variables)


def get_clf_models(svm_var, mv_var, rf_var, lda_var, knn_var, gb_var, ada_var,
                   svm=True, ens=True, rf=True, lda=True, knn=True, nb=True, gb=True, ada=True, xgb=True):
    models = []
    if rf:
        models.append(('RF', get_classifier("rf", rf_var)))
    if lda:
        models.append(('LDA', get_classifier("lda", lda_var)))
    if svm:
        models.append(('SVM', get_classifier("svm", svm_var)))
    if ens:
        models.append(('ENS', get_classifier("mv", mv_var)))
    if knn:
        models.append(('KNN', get_classifier("knn", knn_var)))
    if nb:
        models.append(('NB', get_classifier("nb", {})))
    if gb:
        models.append(('GB', get_classifier("gb", gb_var)))
    if ada:
        models.append(('ADA', get_classifier("ada", ada_var)))
    if xgb:
        models.append(('XGB', get_classifier("xgb", xgb_var)))

    return models


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
                            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "fprice"
                         # , "day0open", "day0close"
                         # , "day1open", "day1close"
                         # , "day2open", "day2close"
                         # , "day3open", "day3close"
                         # , "day4open", "day4close"
                         # , "day5open", "day5close"
                         # , "day6open", "day6close"]
                     ], axis=1)
        df = df.fillna(-1)
    elif label == "oid":
        # df = df.drop(["placesid", "panosid", "lat", "lng", "year"], axis=1)
        df = df[["point", "type", "tree", "treecount", "houseplant", "houseplantcount", "flower", "flowercount",
                 "building", "buildingcount", "skyscraper", "skyscrapercount", "house", "housecount",
                 "conveniencestore", "conveniencestorecount", "office", "officecount", "streetlight", "streetlightcount",
                 "trafficlight", "trafficlightcount", "trafficsign", "trafficsigncount"]]
        # # convert point column to numeric to merge
        df["point"] = df["point"].astype("int")
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    if label == "coco":
        df = df[["point","type", "trafficlight", "trafficlightcount", "firehydrant", "firehydrant", "stopsign", "stopsigncount",
                 "bench", "benchcount", "pottedplant", "pottedplantcount"]]
        # # convert point column to numeric to merge
        df["point"] = df["point"].astype("int")
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    elif label == "tf":
        df = df.drop(["id", "name", "lat", "lng", "id", "name", "lat", "lng"], axis=1)
        # df = df.drop(["id", "name", "lat", "lng", "id", "name", "lat", "lng",
        #          "timediffavg", "timediffmedian"], axis=1)
    elif label == "rf":
        df = df.drop(["id", "name", "lat", "lng"], axis=1)
    return df


def get_best_parameters_for_clf(clf_name, train, trainlabel, scoring):
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
    elif clf_name == "gb":
        n_iter = 27
        random_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3], 'min_samples_split': [2,3,4],
                  'learning_rate': [0.001, 0.01, 0.1], 'loss': ["deviance"]}
        clf = GradientBoostingClassifier()

        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=n_iter, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1, scoring=scoring)
    # Fit the random search model
    clf_random.fit(train, trainlabel)
    print(clf_random.best_params_)
    return clf_random


def reduce_dimensions(dim, n_comp, data, datalabel=None):
    if dim == "pca":
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        return pca
    elif dim == "lda":
        lda = LinearDiscriminantAnalysis(n_components=250)
        lda.fit(data, datalabel)
        return lda


def visualize_clf_accuracy(score, results, names, num):
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
    axes.set_ylabel(score)
    plt.show()


def keep_one_type_col(dfs):
    for i in range(len(dfs)-1):
        dfs[i] = dfs[i].drop(["type"], axis=1)
    return dfs


if __name__ == '__main__':
    # db = create_engine('postgresql://postgres:postgres@localhost/pois')
    #################
    # read the data #
    #################
    df = read_features(scene=True, od_oid=True, od_coco=True,
                       twitter=True, review=True, gf=True)
    # parameters
    red_dim = False
    tt_split = 0.3
    balance_min_class = False
    smote = True
    smote_kind = 'svm'#'svm' # regular
    smote_out_step = 0.5
    cross_val = True
    scaler = StandardScaler()

    # remove unwanted labels
    #df.loc[df['type'] == "cafe", 'type'] = "bar"
    # df.loc[df['type'] == "nightclub", 'type'] = "bar"

    # df.loc[df['type'] == "cafe", 'type'] = "restaurant"
    # df.loc[df['type'] == "bar", 'type'] = "restaurant"
    # df.loc[df['type'] == "coffee shop", 'type'] = "restaurant"
    # df.loc[df['type'] == "nightclub", 'type'] = "restaurant"
    #
    # df.loc[df['type'] == "Art Gallery", 'type'] = "hotel"
    #
    # df.loc[df['type'] == "college_and_university", 'type'] = "hotel"
    #
    # df.loc[df['type'] == "food_drink_shop", 'type'] = "hotel"



    # #df = df[df.type!="cafe"]
    # df = df[df.type!="nightclub"]
    # df = df[df.type!="hotel"]
    # df = df[df.type!="food_drink_shop"]
    # df = df[df.type!="college_and_university"]
    # df = df[df.type!="coffee shop"]
    # df = df[df.type!="Art Gallery"]


    print(df.type.value_counts())

    # store the df
    #df.to_sql('matched_data_features_ams', db, index=False)
    print(list(df))
    print(df.shape)
    ###############
    # CLASSIFYING #
    ###############


    ##########################
    # SETTING CLF PARAMETERS #
    ##########################
    # Random Forest - 400
    rf_var = {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": False, "bootstrap":False,
              "class_weight": "balanced"}
    rf_var2 = {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": False, "bootstrap":False,
              "class_weight": "balanced"}
    rf_var3 = {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": True, "bootstrap":True,
              "class_weight": "balanced"}
    rf_var4 = {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": False, "bootstrap":True,
              "class_weight": "balanced"}
    # KNN
    knn_var = {"k": 10}
    # Gradient Boosting
    gb_var = {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 3,
              'learning_rate': 0.01, 'loss': "deviance"}
    # LDA
    lda_var = {"solver": "svd", "n_components": 200}
    # SVM
    svm_var = {"C": 100, "kernel": "rbf", "degree": 3, "probability": True, "shrinking": False,
               "class_weight": None, "gamma": "auto", "decision_function_shape": "ovr"}
    # Adaboost
    ada_var = {'base_estimator': get_classifier("rf", rf_var), 'n_estimators': 300, 'algorithm': "SAMME.R", 'learning_rate': 0.01}

    # XGBoost
    xgb_var = {}
    # Ensemble - Majority Voting
    mv_var = [("rf", get_classifier("rf", rf_var)),
              ("ada", get_classifier("ada", ada_var)),
              ("gb", get_classifier("gb", gb_var)),
              ("xgb", get_classifier("xgb", {}))
              #("svm", get_classifier("svm", svm_var))
              ]  ##,


    ######################
    # GETTING THE MODELS #
    ######################
    models = get_clf_models(svm_var, mv_var, rf_var, lda_var, knn_var, gb_var, ada_var,
                            svm=False, ens=False, rf=True, lda=False,
                            knn=False, nb=False, gb=False, ada=False, xgb=False)

    scoring = "f1_macro"#"accuracy"
    results = []
    names = []
    print("Training....")
    data_labels = pd.DataFrame(df.type, columns=["type"])
    data_features = df.drop(["type"], axis=1)
    classes = list(data_labels["type"].unique())
    classes.sort()
    classes = {k: v for k, v in enumerate(classes)}
    print(classes)

    #####################################
    # GRID SEARCHING FOR CLF PARAMETERS #
    #####################################
    # print("Getting Best Parameters....")
    # get_best_parameters_for_clf("gb", data_features, data_labels, scoring="f1_micro")
    if red_dim:
        print("Dimensionality Reduction!")
        dim = reduce_dimensions("pca", 200, data_features)
        data_features = dim.transform(data_features)
        #test = dim.transform(test)
    if cross_val:
        nsplits = 2
        trainlabel = data_labels
        train = data_features
        train = scaler.fit_transform(train)
        for name, clf_model in models:
            print("Classifier RUNNING: ", name)
            test_pred = cross_val_predict(clf_model, train, trainlabel,
                                          cv=StratifiedKFold(n_splits=nsplits, shuffle=True))
            # test_probs = cross_val_predict(clf_model, train, trainlabel,
            #                               cv=StratifiedKFold(n_splits=nsplits, shuffle=True),
            #                               method="predict_proba")
            # get predictions
            # test_pred = np.argmax(test_probs, axis=1)
            testlabel = trainlabel
            # for i, row in enumerate(test_probs):
            #     # if pred = bar and it is <0.4 and the next prediction is restaurant
            #     # make it cafe
            #     if (test_pred[i] == 1 and row[1] < 0.4 and row[7] == np.max(np.append(row[0], row[2:8]))):
            #         test_pred[i] = 2
            #     # if pred = rest and it is <0.4 and the next prediction is bar
            #     # make it cafe
            #     if (test_pred[i] == 7 and row[7] < 0.4 and row[1] == np.max(row[0:7])):
            #         test_pred[i] = 2
            # test_pred = [classes[k] for k in test_pred]
            scores = cross_val_score(clf_model, train, trainlabel.values.ravel(),
                                     cv=StratifiedKFold(n_splits=nsplits, shuffle=True),
                                     scoring=scoring, verbose=3)
            print("F1 SCORE: {m} (+/- {s}".format(m=scores.mean(), s=scores.std() * 2))
            precision, recall, fscore, support = score(testlabel, test_pred, labels=["restaurant", "bar", "hotel",
                                                                                     "food_drink_shop", "cafe",
                                                                                     "college_and_university",
                                                                                     "coffee shop", "Art Gallery"
                                                                                     ])
            print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
            print('precision: {}'.format(precision))
            print('recall: {}'.format(recall))
            print('fscore: {}'.format(fscore))
            print('fscore MEAN: {}'.format(fscore.mean()))

            print('support: {}'.format(support))
            results.append(scores)
            names.append(name)
    else: # not cross vall
        for name, clf_model in models:
            res = []
            acc = []
            for i in range(5):
                train, test, trainlabel, testlabel = train_test_split(data_features, data_labels,
                                                                      test_size=tt_split)#, random_state=43)
                train = scaler.fit_transform(train)
                test = scaler.fit_transform(test)
                if smote:
                    print("Oversampling....")
                    sm = SMOTE(kind=smote_kind, out_step=smote_out_step)  # random_state=42)#, kind="svm")
                    train, trainlabel = sm.fit_sample(train, trainlabel)
                    print('Resampled dataset shape {}'.format(Counter(trainlabel)))

                print("Classifier: ", name)
                clf = train_classifier(clf_model, train, trainlabel)
                test_pred = clf.predict(test)
                precision, recall, fscore, support = score(testlabel, test_pred, labels=["restaurant", "bar", "hotel", "cafe",
                                                                                     "food_drink_shop",
                                                                                     "college_and_university",
                                                                                     "coffee shop", "Art Gallery"
                                                                                     ])
                res.append(fscore.mean())
                print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
                acc.append(accuracy_score(testlabel, test_pred))
                #print('precision: {}'.format(precision))
                #print('recall: {}'.format(recall))
                print('fscore: {}'.format(fscore))
                print('FSCORE MEAN: {}'.format(fscore.mean()))
                print('support: {}'.format(support))
            print("FINAL FSCORE:", np.mean(res))
            print("FINAL ACC:", np.mean(acc))

            names.append(name)
            results.append(res)
    print(results)
    print(names)
    visualize_clf_accuracy("F1 score", results, names, len(models))
    cm = confusion_matrix(testlabel.type.tolist(), test_pred, labels=["restaurant", "bar", "hotel", "cafe",  "food_drink_shop",
                                        "college_and_university", "coffee shop", "Art Gallery"])
    print(test_pred)
    print(testlabel.type.tolist())
    print("Confusion matrix:\n%s" % cm)
    plot_confusion_matrix(cm=cm,
                          normalize=True,
                          target_names=["restaurant", "bar", "hotel", "cafe", "food_drink_shop",
                                        "college_and_university", "coffee shop", "Art Gallery"],
                          title="Confusion Matrix")
    #cm.plot(normailized=True)
    plt.show()
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