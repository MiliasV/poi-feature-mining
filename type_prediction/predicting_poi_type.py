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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from xgboost import plot_importance
from xgboost import XGBClassifier
import xgboost as xgb
import itertools


import pickle


def load_with_pickle(filename):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def save_with_pickle(filename, obj):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'wb') as f:
        pickle.dump(obj, f)


def get_xgb_feat_importances(clf):

    # if isinstance(clf, xgb.XGBModel):
    #     # clf has been created by calling
    #     # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
    # else:
    #     # clf has been created by calling xgb.train.
    #     # Thus, clf is an instance of xgb.Booster.
    #     fscore = clf.get_fscore()

    fscore = clf.get_booster().get_fscore()

    feat_importances = []
    for ft, score in fscore.items():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    print (feat_importances.head())
    return feat_importances


def plot_confusion_matrix(cm,
                          target_names, city,
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

    plt.figure(figsize=(18.61, 9.86)) # 8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("")
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=13, rotation=45)
        plt.yticks(tick_marks, target_names, fontsize=13)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    #figure.set_size_inches(18.61, 9.86)

    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)#\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()

    plt.savefig("/home/bill/Desktop/thesis/conf_" + title  + city + ".pdf", dpi=100)

    # plt.show()


def read_table_as_df(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    return pd.read_sql_query("select * from {tab}".format(tab=table), con=conn)


def read_features(city, scene=True, od_oid=True, od_coco=True,
                  twitter=True, review=True, gf=True, spatial=True):
    dfs = []
    attr = []
    # scene features
    # google-fsq features
    if gf:
        gf_df = read_table_as_df("matched_places_gf_features_" + city)
        gf_df = preprocess_df(gf_df, "gf")
        dfs.append(gf_df)
        attr.append("gf")
    if spatial:
        spatial_df = read_table_as_df("matched_places_spatial_features_" + city)
        spatial_df = preprocess_df(spatial_df, "spatial")
        dfs.append(spatial_df)
        attr.append("spatial")
    if scene:
        sf_df = read_table_as_df("matched_places_agg_scene_features_50_" + city)
        dfs.append(sf_df)
        # print(list(sf_df))
        attr.append("scene")
    # object detection features
    if od_oid:
        od_oid_df = read_table_as_df("matched_places_agg_od_oid_" + city)
        od_oid_df = preprocess_df(od_oid_df, "oid")
        dfs.append(od_oid_df)
        attr.append("oid")
    #coco
    if od_coco:
        od_coco_df = read_table_as_df("matched_places_agg_od_coco_" + city)
        od_coco_df = preprocess_df(od_coco_df, "coco")
        dfs.append(od_coco_df)
        attr.append("coco")
    # twitter features
    if twitter:
        tf_df = read_table_as_df("matched_places_text_features_10_25_" + city)
        tf_df = preprocess_df(tf_df, "tf")
        dfs.append(tf_df)
        attr.append("twitter")
    # review features
    if review:
        # rf_df = read_table_as_df("matched_places_review_features_10_25_" + city)
        rf_df = read_table_as_df("matched_places_experiential_features_" + city)
        rf_df = preprocess_df(rf_df, "rf")
        dfs.append(rf_df)
        attr.append("reviews")

    ##################
    # merge the data #
    ##################
    # dfs to merge gf_df sf_df,
    # tf_df, gf_df, rf_df, od_oid_df, sf_df
    dfs = keep_one_type_col(dfs)
    # merge by reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='placesid'), dfs)
    # drop not needed cols
    df = df.fillna(-1)
    df = df.drop(["placesid"], axis=1)
    return df, attr


def majority_voting(variables):
    mv_classifier = VotingClassifier(estimators=variables, voting="soft")
    return mv_classifier


def naive_bayes(variables):
    nb = GaussianNB()
    return nb


def xgb(params):
    return XGBClassifier(**params)


def gradient_boost(params):
    # params = {'n_estimators': variables["n_estimators"], 'max_depth': variables["max_depth"],
    #           'min_samples_split': variables["min_samples_split"],
    #           'learning_rate': variables["learning_rate"], 'loss': variables["loss"]}
    gb_classifier = GradientBoostingClassifier(**params)
    return gb_classifier


def ada_boost(params):
    # params = {'base_estimator': variables["base_estimator"], 'n_estimators': variables["n_estimators"],'algorithm': variables["algorithm"],
    #           'learning_rate': variables["learning_rate"]}
    ada_classifier = AdaBoostClassifier(**params)
    return ada_classifier


def random_forest(variables):
    rf_classifier = RandomForestClassifier(**variables)
    return rf_classifier


def decision_tree(variables):
    dt_classifier = tree.DecisionTreeClassifier()
    return dt_classifier


def knn(params):
    # k = variables["k"]
    knn_classifier = KNeighborsClassifier(**params)
    return knn_classifier


def lda(params):
    # solv = variables["solver"]
    # nc = variables["n_components"]
    lda_classifier = LinearDiscriminantAnalysis(**params)
    return lda_classifier


def svmc(params):
    clf_svm = svm.SVC(**params)
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
        df = df.drop(["id", "point", "name", "fid", # "Monday", "Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                      "fprice"], axis=1)
        conv_dict = {'morning': 1., 'evening': 2., 'afternoon': 3., 'night': 4., 'None': np.nan}
        for i in ["Monday", "Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            df[i] = df[i].map(conv_dict)

        # , "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]].map(conv_dict)
            # replace(conv_dict, inplace=True)

        # df[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]] = df.apply(conv_dict.get, axis=1)
        #df = df.drop(["id"], axis=1)
        #df = df.drop(["id", "name", "gid", "fid", "Monday", "Tuesday",
        #                    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "fprice"
                         # , "day0open", "day0close"
                         # , "day1open", "day1close"
                         # , "day2open", "day2close"
                         # , "day3open", "day3close"
                         # , "day4open", "day4close"
                         # , "day5open", "day5close"
                         # , "day6open", "day6close"]
                     #], axis=1)
        df = df.fillna(0)
    elif label=="spatial":
        df = df.drop(["point", "name","nightclub_1000", "nightclub_100", "nightclub_5000"], axis=1)
        df = df.fillna(0)

    elif label == "oid":
        # df = df.drop(["placesid", "panosid", "lat", "lng", "year"], axis=1)
        df = df[["placesid", "type", "tree", "treecount", "houseplant", "houseplantcount", "flower", "flowercount",
                 "building", "buildingcount", "skyscraper", "skyscrapercount", "house", "housecount",
                 "conveniencestore", "conveniencestorecount", "office", "officecount", "streetlight", "streetlightcount",
                 "trafficlight", "trafficlightcount", "trafficsign", "trafficsigncount"]]
        # df = df[["point", "type", "tree", "treecount", "houseplant", "houseplantcount", "flower", "flowercount",
        #          "building", "buildingcount", "skyscraper", "skyscrapercount", "house", "housecount",
        #          "conveniencestore", "conveniencestorecount", "office", "officecount", "streetlight", "streetlightcount",
        #          "trafficlight", "trafficlightcount", "trafficsign", "trafficsigncount"]]
        # # convert point column to numeric to merge
        #df["point"] = df["point"].astype("int")
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    if label == "coco":
        df = df[["placesid","type", "trafficlight", "trafficlightcount", "firehydrant", "stopsign", "stopsigncount",
                 "bench", "benchcount", "pottedplant", "pottedplantcount"]]
        # # convert point column to numeric to merge
        #df["point"] = df["point"].astype("int")
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    elif label == "tf":
        df = df.drop(["id", "point", "name", "lat", "lng", "id", "name", "lat", "lng"], axis=1)
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


def visualize_clf_accuracy(score, results, names, num, title, city):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    # notch shape box plot
    bplot2 = axes.boxplot(results,
                          notch=False,  # notch shape
                          vert=True,  # vertical box alignment
                          patch_artist=True,  # fill with color
                          labels=names)  # will be used to label x-ticks
    axes.set_title("")
    # fill with colors
    colors = ['red', 'lightblue', 'lightgreen', 'pink', 'grey', 'yellow', 'black', 'purple', 'brown']
    colors = colors[0:num]

    for patch, cl in zip(bplot2["boxes"], colors):
        patch.set(facecolor=cl)
    # adding horizontal grid lines
    # for ax in axes:
    #axes.yaxis.grid(linewidth=2, which='major', alpha=0.01)
    #axes.yaxis.grid(True)
    axes.yaxis.grid(which="major", color='black', linestyle='-', linewidth=0.4)
    start, end = axes.get_ylim()
    axes.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2), 0.05))
    axes.set_xlabel('Classifiers')
    axes.set_ylabel(score)
    #plt.show()
    plt.savefig("/home/bill/Desktop/thesis/Evaluation_results/" + title  + city + ".pdf")


def keep_one_type_col(dfs):
    for i in range(len(dfs)-1):
        dfs[i] = dfs[i].drop(["type"], axis=1)
    return dfs


if __name__ == '__main__':

    # scoring = "f1_macro"
    # city = "ath"
    #
    # results = load_with_pickle("f_score_all_results")
    # names = load_with_pickle("f_score_all_names")
    # len = load_with_pickle("f_score_all_len_models")
    # title = load_with_pickle("f_score_all_title")
    # visualize_clf_accuracy(scoring, results, names, len, title, city)
    # print(a)
    # db = create_engine('postgresql://postgres:postgres@localhost/pois')

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
    knn_var = {"n_neighbors": 10}
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

    ############################
    # SETTING OTHER PARAMETERS #
    ############################
    scoring = "f1_macro"#"accuracy"
    scoring_2 = "macro"
    # parameters
    red_dim = False
    tt_split = 0.1
    balance_min_class = False
    smote = False
    smote_kind = 'svm'#'svm' # regular
    smote_out_step = 0.5
    cross_val = True
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    #################
    # read the data #
    #################
    # area of Amsterdam equal to Athens
    # lat < 52.38524 AND lat > 52.34757 and lng > 4.84067 and lng < 4.91946
    city = "ams_centre"
    features_num = 15
    df, attr = read_features(city, scene=True, od_oid=True, od_coco=True,
                       twitter=True, review=True, gf=True, spatial=True)

    data_labels = pd.DataFrame(df.type, columns=["type"])

    sw = compute_sample_weight(class_weight='balanced', y=data_labels)
    # print(sw[0:10])
    # print(data_labels)
    xgb_var = {"sample_weight":sw}
    # Ensemble - Majority Voting
    mv_var = [("rf", get_classifier("rf", rf_var)),
              ("lda", get_classifier("lda", lda_var)),
              ("svm", get_classifier("svm", svm_var)),
              ("xgb", get_classifier("xgb", xgb_var))
              #("svm", get_classifier("svm", svm_var))
              ]  ##,
    ######################
    # GETTING THE MODELS #
    ######################
    models = get_clf_models(svm_var, mv_var, rf_var, lda_var, knn_var, gb_var, ada_var,
                            svm=False, ens=False, rf=False, lda=False,
                            knn=False, nb=False, gb=False, ada=False, xgb=True)

    labels = ["restaurant", "bar", "hotel",
     "food_drink_shop", "cafe",
     "college_and_university",
      "art_gallery", "coffee_shop",
     "clothing_store", "nightclub", "gym"
     ]
    title_lab = "set_A"
    rem_labels = ["nightclub"]#["nightclub", "gym"]
    labels = [x for x in labels if x not in rem_labels]
    change_lab = ["None"]#[("bar", "cafe")]#[("bar", "cafe")]
    # change_lab = [("college_and_university", "coll & uni"),
    #               ("food_drink_shop", "food/drink shop")]
    # change_lab = [("clothing_store", "shop_and_service"),
    #               ("food_drink_shop", "food"),
    #               ("coffee_shop", "food"),
    #               ("bar", "nightlife_spot"),
    #               # ("nightclub", "nightlife_spot"),
    #               ("restaurant", "food"),
    #               ("cafe", "food"),
    #               ("hotel", "travel_transport"),
    #               ("art_gallery","arts_and_entertainment"),
    #               ("college_and_university", "college_and_university"),
    #               ("gym", "outdoor_and_recreation")
    #               ]
    for rem_type in rem_labels:
        df = df[df.type != rem_type]
    for type_tuple in change_lab:
        df.loc[df['type'] == type_tuple[0], 'type'] = type_tuple[1]
    labels = [x for x in labels if x not in rem_labels]
    if change_lab[0]!="None":
        labels =list(set([y for (x,y) in change_lab if x in labels]))
    # labels = ["restaurant", "bar", "hotel",
    #  "food/drink shop", "cafe",
    #  "coll & uni",
    #   "art_gallery", "coffee_shop",
    #  "clothing_store", "gym"
    #  ]
    # print(labels)
    print(df.type.value_counts())
    print(df.shape)
    # remove unwanted labels
    #df.loc[df['type'] == "cafe", 'type'] = "bar"
    # df.loc[df['type'] == "nightclub", 'type'] = "bar"

    # df.loc[df['type'] == "cafe", 'type'] = "restaurant"
    # df.loc[df['type'] == "bar", 'type'] = "restaurant"
    #df.loc[df['type'] == "coffee shop", 'type'] = "cafe"
    # df.loc[df['type'] == "nightclub", 'type'] = "restaurant"
    #
    # df.loc[df['type'] == "Art Gallery", 'type'] = "hotel"
    #
    # df.loc[df['type'] == "college_and_university", 'type'] = "hotel"
    #
    # df.loc[df['type'] == "food_drink_shop", 'type'] = "hotel"
    # df.loc[df['type'] == "food_drink_shop", 'type'] = "hotel"



    #df = df[df.type!="gym"]
    # df = df[df.type!="nightclub"]
    # df = df[df.type!="hotel"]
    # df = df[df.type!="food_drink_shop"]
    # df = df[df.type!="college_and_university"]
    # df = df[df.type!="coffee shop"]
    # df = df[df.type!="Art Gallery"]
    # df = df[df.type!="clothing_store"]
    # print(df.type.value_counts())

    # store the df
    #df.to_sql('matched_data_features_ams', db, index=False)
    print(list(df))
    print(df.shape)
    ###############
    # CLASSIFYING #
    ###############

    # Classifying
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
        print("CROSS VAL")
        nsplits = 10
        trainlabel = data_labels
        train = data_features.copy()

        train.loc[:,:] = scaler.fit_transform(data_features)
        for name, clf_model in models:
            print("Classifier RUNNING: ", name)

            # scores = cross_val_score(clf_model, train, trainlabel.values.ravel(),
            #                          cv=StratifiedKFold(n_splits=nsplits, shuffle=True),
            #                          scoring=scoring, verbose=3)
            # train[["topiceng104", "topiceng105", "topiceng106", "topiceng107", "topiceng108"]] = \
            #     train[["topiceng104", "topiceng105", "topiceng106", "topiceng107", "topiceng108"]].apply(pd.to_numeric, errors='coerce', axis=1)
            # train = train.apply(pd.to_numeric, axis=1)
            clf_trained = train_classifier(clf_model, train, trainlabel)
            # print("ok")

            # get_xgb_feat_importances(clf_model)
            feature_important =clf_trained.get_booster().get_score(importance_type='gain')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False).head(features_num)
            print(data.head())
            if city =="ath":
                data.rename(index={'bar_5000': "# Bar (5km)",
                                   "cafe_5000": "# Cafe (5km)",
                                   "food_drink_shop_5000": "# Food/Drink (5km)",
                                   "clothing_store_5000": "# Cl. Store (5km)",
                                   "coffee_shop_5000": "# Coffee Shop (5000m)",
                                   "college_and_university_5000": "# Coll/Uni (5km)",
                                   "gym_5000": "# Gym (5km)",
                                   "hotel_5000": "# Coffee (5km)",
                                   "clothing_store_1000": "# Cl. Store (1km)",
                                   "art_gallery_5000": "# Art Gall. (5km)",
                                   "restaurant_5000": "# Restaurant (5km)",
                                   "day0open": "# Monday Open Time",
                                   'bar_1000': "# Bar (1km)",
                                   "topiceng104_y": "Eng. Reviews Topic 4",
                                   "topiceng105_y": "Eng. Reviews Topic 5",

                                   }, inplace=True)
            else:
                data.rename(index={"Saturday": "Saturday (PopTimes)",

                                   "clothing_store_100": "# Cl. Store (100m)",

                                   'bar_1000': "# Bar (1000m)",
                                   "day5open": "# Saturday Open Time",
                                   "day5close": "# Saturday Close Time",
                                   "day0close": "# Monday Close Time",
                                   "day2open": "# Wednesday Open Time",

                                   "day6close": "# Sunday Close Time",
                                   "topiceng2511_y": "Eng. Reviews Topic 11",
                                   "topiceng2521_y": "Eng. Reviews Topic 21",
                                   "enwordcount_y": "# Words in Eng. Reviews",
                                   "totaltweetcount": "# Tweets",
                                   "enrevcount": "# Eng. Reviews",

                                   "totalwordcount_y": "# Words in all Reviews",

                                   "topiceng104_y": "Eng. Reviews Topic 4",
                                   "topiceng105_y": "Eng. Reviews Topic 5",

                                   }, inplace=True)

            data.plot(kind='barh', grid=True, fontsize=16, legend="Importance Score")
            plt.show()

            # print(a)
            testlabel = trainlabel
            # test_pred = [classes[k] for k in test_pred]

            # test_pred = cross_val_predict(clf_model, train, trainlabel,
            #                               cv=StratifiedKFold(n_splits=nsplits, shuffle=True), verbose=5)
            # precision, recall, fscore, support = score(trainlabel, test_pred, labels=labels, average=scoring_2)

            #clf_model.fit(train, trainlabel)
            # data_dmatrix = xgb.DMatrix(data=train, label=trainlabel)
            # xg_reg = xgb.train(params={}, dtrain=data_dmatrix, num_boost_round=10)
            # plot_importance(xg_reg, max_num_features=10)
            #
            #
            # print(clf_model.get_booster().get_fscore())
            # print(clf_model.get_booster().get_fscore().items())

            # print(zip(clf_model.get_booster().columns, clf_model.get_booster().feature_importances_))
            # clf_model.get_fscore()
            # mapper = {'f{0}'.format(i): v for i, v in enumerate(dtrain.feature_names)}
            # mapped = {mapper[k]: v for k, v in clf_model.get_fscore().items()}
            # plot_importance(mapped, color='red', max_num_features=10)
            # plt.show()
            #print("F1 SCORE: 1st method {m} (+/- {s}".format(m=scores.mean(), s=scores.std() * 2))

            print("CITY: ", city, " Attr.", attr)
            print("CLASSIFIER - ", scoring_2, " :", name)
            print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
            print('precision: {}'.format(precision))
            print('recall: {}'.format(recall))
            print('fscore: {}'.format(fscore))
            #print('fscore MEAN: {}'.format(fscore.mean()))
            print('support: {}'.format(support))
            # #
            # results.append(scores)
            # names.append(name)
    else: # not cross vall
        print("TRAIN/TEST SPLIT")
        for name, clf_model in models:
            res = []
            acc = []
            for i in range(2):
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
                precision, recall, fscore, support = score(testlabel, test_pred, labels=labels, average=scoring_2)
                print("CITY: ", city, " Attr.", attr)
                print("CLASSIFIER - ", scoring_2, " :", name)
                print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
                print('precision: {}'.format(precision))
                print('recall: {}'.format(recall))
                print('fscore: {}'.format(fscore))
            print("FINAL FSCORE:", np.mean(res))
            print("FINAL ACC:", np.mean(acc))
            print(a)
            names.append(name)
            # results.append(res)
   # print(results)
    print(names)
    title = title_lab + "_2_"
    for a in attr:
        title+= a + "_"
    title+="rem_"
    for rem in rem_labels:
        title+= rem + "_"
    # title+="change_"
    # for t in change_lab:
    #     title+= str(t) + "_"
    # save_with_pickle("f_score_results_" + title + city, results)
    # save_with_pickle("f_score_names_" + title + city, names)
    # save_with_pickle("f_score_len_models_" + title + city, len(models))
    # save_with_pickle("f_score_title_" + title + city, title)
    #
    # visualize_clf_accuracy(scoring, results, names, len(models), title, city)

    # cm = confusion_matrix(testlabel.type.tolist(), test_pred, labels=labels)
    # print(test_pred)
    # print(testlabel.type.tolist())
    # print("Confusion matrix:\n%s" % cm)
    # plot_confusion_matrix(cm=cm,
    #                       normalize=True,
    #                       target_names=labels, city=city,
    #                       title=title)
    # #cm.plot(normailized=True)
    #plt.show()

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