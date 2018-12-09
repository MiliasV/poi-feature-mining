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
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus


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
                     color="white" if cm[i, j] > 0.75 else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    #figure.set_size_inches(18.61, 9.86)

    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)#\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()

    plt.savefig("/home/bill/Desktop/thesis/report_images/important_features/conf_" + title  + city + ".pdf", dpi=100)

    plt.show()


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
        tf_df = read_table_as_df("matched_places_text_features_10_" + city)
        tf_df = preprocess_df(tf_df, "tf")
        dfs.append(tf_df)
        attr.append("twitter")
    # review features
    if review:
        # rf_df = read_table_as_df("matched_places_review_features_10_25_" + city + "_2")
        rf_df = read_table_as_df("matched_places_experiential_features_10_" + city)
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
    df = df.fillna(0)
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
        df = df.fillna(0)
    elif label=="spatial":
        df = df.drop(["point", "name","nightclub_1000", "nightclub_100", "nightclub_2000", "nightclub_3000"], axis=1)
        df = df.fillna(0)

    elif label == "oid":
        # df = df.drop(["placesid", "panosid", "lat", "lng", "year"], axis=1)
        df = df[["placesid", "type", "tree", "treecount", "houseplant", "houseplantcount", "flower", "flowercount",
                 "building", "buildingcount", "skyscraper", "skyscrapercount", "house", "housecount",
                 "conveniencestore", "conveniencestorecount", "office", "officecount", "streetlight", "streetlightcount",
                 "trafficlight", "trafficlightcount", "trafficsign", "trafficsigncount"]]
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    if label == "coco":
        df = df[["placesid","type", "trafficlight", "trafficlightcount", "firehydrant", "stopsign", "stopsigncount",
                 "bench", "benchcount", "pottedplant", "pottedplantcount"]]
        # convert true, false to numeric
        df = df * 1
        df = df.fillna(0)

    elif label == "tf":
        df = df.drop(["id", "point", "name", "lat", "lng", "name"], axis=1)
    elif label == "rf":
        df = df.drop(["id", "name", "lat", "lng"], axis=1)

    return df


def get_best_parameters_for_clf(clf_name, traindata, trainlabel, scoring):
    # le = LabelEncoder()
    # le.fit(trainlabel)
    # trainlabel = le.transform(trainlabel)
    train = traindata.copy()
    train.loc[:, :] = scaler.fit_transform(traindata)
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
    elif clf_name == "xgb":
        n_iter = 32
        # A parameter grid for XGBoost
        random_grid = {
            'min_child_weight': [1, 10],
            'gamma': [0.5, 2, 5],
            'subsample': [0.6,  1.0],
            'colsample_bytree': [0.6,  1.0],
            'max_depth': [3, 5]
        }
        clf = XGBClassifier()

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
    axes.yaxis.set_ticks(np.arange(round(0, 2), round(0.70, 2), 0.05))
    axes.set_xlabel('Classifiers')
    axes.set_ylabel("F1-score (macro)")
    #plt.show()
    plt.savefig("/home/bill/Desktop/thesis/" + title  + city + "ex.pdf")


def keep_one_type_col(dfs):
    for i in range(len(dfs)-1):
        dfs[i] = dfs[i].drop(["type"], axis=1)
    return dfs


def plot_dt(clf, names):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("/home/bill/Desktop/dtree_2.pdf")


def vis_boxplot_clfs(scoring, city):
    results = load_with_pickle("f_score_results_set_A_2_gf_spatial_scene_oid_coco_twitter_reviews_rem_nightclub_" + city)
    names = load_with_pickle("f_score_names_set_A_2_gf_spatial_scene_oid_coco_twitter_reviews_rem_nightclub_" + city)
    len = load_with_pickle("f_score_len_models_set_A_2_gf_spatial_scene_oid_coco_twitter_reviews_rem_nightclub_" + city)
    title = load_with_pickle("f_score_title_set_A_2_gf_spatial_scene_oid_coco_twitter_reviews_rem_nightclub_" + city)
    visualize_clf_accuracy(scoring, results, names, len, title, city)


def set_clf_parameters(data_labels):
    # Random Forest - 400
    rf_var = {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2,
          "min_samples_leaf":1, "criterion": "gini", "n_jobs": -1, "oob_score": False, "bootstrap":False,
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
    sw = compute_sample_weight(class_weight='balanced', y=data_labels)
    xgb_var = {"sample_weight": sw, 'subsample': 1, 'min_child_weight': 1, 'max_depth': 5, 'gamma': 2,
               'colsample_bytree': 0.6}
    # Ensemble - Majority Voting
    mv_var = [("rf", get_classifier("rf", rf_var)),
              ("lda", get_classifier("lda", lda_var)),
              ("svm", get_classifier("svm", svm_var)),
              ("xgb", get_classifier("xgb", xgb_var))
              # ("svm", get_classifier("svm", svm_var))
              ]  ##,
    nb_var = {}
    return rf_var, knn_var, lda_var, svm_var, xgb_var, mv_var, gb_var, ada_var, nb_var


def rename_features_for_plot(city, data):
    if city == "ath":
        data.rename(index={
            "day6open": "Sunday (Open Time)",
            "day2close": "Wednesday (Close Time)",
            "Saturday": "Saturday (PopTimes)",
            "Sunday": "Sunday (PopTimes)",
            "day4close": "Friday (Close Time)",
            "day4open": "Friday (Open Time)",
            "day5close": "Saturday (Close Time)",
            "day3close": "Thursday (Close Time)",
            "day3open": "Thursday (Open Time)",
            "day6close": "Sunday (Close Time)",
            "cafe_1000": "# of nearby Cafe (radius = 1km)",
            "clothing_store_100": "# of nearby Cloth. Stores (radius = 100m)",

            "topiceng109_y": "Topic (9th/10): 'Food Place'",
                             # "\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng255_y": "Topic (5th/25): 'Coffee/Drink' ",
                             # "\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng2518_y": "Topic (18th/25): 'Food Place/Service' ",
                              # "\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng2511_y": "Topic (11th/25): 'Hotel/Room/Acropolis' ",
                              # "\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng106_y": "Topic (6th/10): 'Restaurant' ",
                             # "(Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng107_y": "Topic (7th/10): 'Hotel/Room/Breakfast' ",
                             # "\n (Source: Reviews, Lang: Eng., Method: LDA)",

        }, inplace=True)
    else:
        data.rename(index={
            "Saturday": "Saturday (PopTimes)",
            "Sunday": "Sunday (PopTimes)",

            "clothing_store_100": "# of nearby Cloth. Stores (radius = 100m)",
           'bar_1000': "# of nearby Bars (radius = 1km)",
           "day4open": "Friday (Open Time)",
           "day5open": "Saturday (Open Time)",
           "day5close": "Saturday (Close Time)",
           "day0close": "Monday (Close Time)",
           "day2open": "Wednesday (Open Time)",
           "engavgword_y": "# Words in Eng. Reviews (Avg.)",
           "day6close": "Sunday (Close Time)",

           "enwordcount_y": "# of Words in Eng. Reviews (Sum)",
           "totaltweetcount": "# of Tweets (Sum)",
           "enrevcount": "# of Eng. Reviews (Sum)",
           "totalwordcount_y": "# Words in all Reviews (Sum)",

            "topiceng2519_y": "Topic (19th/25): 'Hotel' ",
                              #"\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng107_y": "Topic (7th/10): 'Food Place'",
                            # \n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng1010_y": "Topic (10th/10): 'Store' ",
                              # "\n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng257_y": "Topic (7th/25): 'Beer Place'",
                             # " \n (Source: Reviews, Lang: Eng., Method: LDA)",
            "topiceng101_y": "Topic (1st/10): 'Hotel'"
                             # " \n (Source: Reviews, Lang: Eng., Method: LDA)",

        }, inplace=True)
    return data


def store_fi_to_table(target_label, city, data):
    store_table = "matched_places_" + city + "_fp_" + target_label[0]
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    print(data.head())
    data = rename_features_for_plot(city, data)
    data.reset_index(inplace=True)
    data.to_sql(store_table, db, index=True)


def get_xgb_feature_importances(clf_model, train, trainlabel, target_label):
    clf_trained = train_classifier(clf_model, train, trainlabel)
    get_xgb_feat_importances(clf_model)
    feature_important = clf_trained.get_booster().get_score(importance_type='gain')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"])\
        .sort_values(by="score", ascending=False).head(features_num)
    # print(data.head())
    # store_fi_to_table(target_label, city, data)

    data.plot(kind='barh', grid=True, fontsize=22, legend="Importance Score", color=[plt.cm.Paired([1,1,2,1,2,1,1,1,5,1,2,2,1,2,1,2,8])])
    plt.show()


def write_results_to_file(name, city, attr, testlabel, test_pred, precision, recall, fscore):
    file = open(name, "a")
    file.write("city=  {city} , Set B  attr=   {attr}  acc=  {acc} precision = "
               " {precision}  recall = {recall} fscore = {fscore} \n".format(city=city, attr=attr,
                                                                             acc=accuracy_score(testlabel, test_pred),
                                                                             precision=precision, recall=recall,
                                                                             fscore=fscore))
    file.close()


def print_results(city, attr, scoring_2, name, testlabel, test_pred, precision, recall, fscore,  support):
    print("CITY: ", city, " Attr.", attr)
    print("CLASSIFIER - ", scoring_2, " :", name)
    print("ACCURACY SCORE: ", accuracy_score(testlabel, test_pred))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))



if __name__ == '__main__':
    # scoring = "f1_macro"
    # city = "ams_centre"
    # vis_boxplot_clfs(scoring, city)
    # db = create_engine('postgresql://postgres:postgres@localhost/pois')

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
    feature_set_list = ["all"]  #""scene", "social", "exp", "funct", "spatial", "all"]:
    # # Test only one label!!!!
    target_label = []# ["restaurant`"]

    classes_set = "a"
    city = "ath"
    features_num = 15
    #################
    # read the data #
    #################
    # area of Amsterdam equal to Athens

    for i in feature_set_list:
        if i=="scene":
            df, attr = read_features(city, scene=True, od_oid=True, od_coco=True,
                               twitter=False, review=False, gf=False, spatial=False)
        elif i=="social":
            df, attr = read_features(city, scene=False, od_oid=False, od_coco=False,
                               twitter=True, review=False, gf=False, spatial=False)
        elif i=="exp":
            df, attr = read_features(city, scene=False, od_oid=False, od_coco=False,
                               twitter=False, review=True, gf=False, spatial=False)
        elif i=="funct":
            df, attr = read_features(city, scene=False, od_oid=False, od_coco=False,
                               twitter=False, review=False, gf=True, spatial=False)
        elif i=="spatial":
            df, attr = read_features(city, scene=False, od_oid=False, od_coco=False,
                               twitter=False, review=False, gf=False, spatial=True)
        elif i=="all":
            df, attr = read_features(city, scene=True, od_oid=True, od_coco=True,
                               twitter=True, review=True, gf=True, spatial=True)

        ########################
        # Selecting the labels #
        #######################
        # All labels
        labels = ["restaurant", "bar", "hotel",
                  "food_drink_shop", "cafe",
                  "college_and_university",
                  "art_gallery", "coffee_shop",
                  "clothing_store", "nightclub", "gym"
                  ]
        title_lab = "set_A"

        # Remove labels
        rem_labels =["nightclub"]# ,
                  #  "bar",
                  # "food_drink_shop", "cafe",
                  # "college_and_university",
                  # "art_gallery", "coffee_shop", "nightclub", "gym"]  # ["nightclub", "gym"]
        labels = [x for x in labels if x not in rem_labels]
        for rem_type in rem_labels:
            df = df[df.type != rem_type]

        if target_label:
            #separating the target class from the rest
            df_target = df.loc[df["type"] == target_label[0]]
            df_other = df.loc[df["type"] != target_label[0]]
            # amount of the other classes (minus the target)
            other_labels_count = len(labels) - 1
            target_count = df_target.shape[0]
            # equal amount of samples from each class to have in total equal number of the target class and the rest
            samples_per_class = int(target_count/(other_labels_count+2.3))
            print(samples_per_class)
            # subsampling
            df_other = df_other.groupby('type').apply(lambda x: x.sample(samples_per_class, random_state=42))
            print(df_other.type.value_counts())
            # append one dataframe to the other
            df = df_target.append(df_other)
            print(df.type.value_counts())
            if target_label:
                df.loc[df["type"] != target_label[0], 'type'] = "other"
                print(target_label[0])
                labels = [target_label[0], "other"]
                # df = df.drop(df[df['type'] == 'other'].sample(frac=0.9).index)

        # Change Labels
        change_lab = ["None"]  # [("bar", "cafe")]#[("bar", "cafe")]

        if classes_set=="b":
            change_lab = [("clothing_store", "shop_and_service"),
                          ("food_drink_shop", "food"),
                          ("coffee_shop", "food"),
                          ("bar", "nightlife_spot"),
                          # ("nightclub", "nightlife_spot"),
                          ("restaurant", "food"),
                          ("cafe", "food"),
                          ("hotel", "travel_transport"),
                          ("art_gallery","arts_and_entertainment"),
                          ("college_and_university", "college_and_university"),
                          ("gym", "outdoor_and_recreation")
                          ]

        for type_tuple in change_lab:
            df.loc[df['type'] == type_tuple[0], 'type'] = type_tuple[1]
        # labels = [x for x in labels if x not in rem_labels]
        if change_lab[0] != "None":
            labels = list(set([y for (x, y) in change_lab if x in labels]))

        data_labels = pd.DataFrame(df.type, columns=["type"])
        data_features = df.drop(["type"], axis=1)
        print(df.type.value_counts())
        print(df.shape)

        ##########################
        # SETTING CLF PARAMETERS #
        ##########################
        rf_var, knn_var, lda_var, svm_var, xgb_var, mv_var, gb_var, ada_var, nb_var = set_clf_parameters(data_labels)

        ######################
        # GETTING THE MODELS #
        ######################
        models = get_clf_models(svm_var, mv_var, rf_var, lda_var, knn_var, gb_var, ada_var,
                                svm=False, ens=False, rf=False, lda=False,
                                knn=False, nb=False, gb=False, ada=False, xgb=True)
        ###############
        # CLASSIFYING #
        ###############

        # Classifying
        results = []
        names = []
        print("Training....")

        classes = list(data_labels["type"].unique())
        classes.sort()
        classes = {k: v for k, v in enumerate(classes)}
        print(classes)

        #####################################
        # GRID SEARCHING FOR CLF PARAMETERS #
        #####################################
        # print("Getting Best Parameters....")
        # get_best_parameters_for_clf("xgb", data_features, data_labels, scoring="f1_macro")
        # print(a)
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

            train.loc[:, :] = scaler.fit_transform(data_features)
            for name, clf_model in models:
                print("Classifier RUNNING: ", name)
                testlabel = trainlabel
                #########################################
                # get scores for classifiers comparison #
                #########################################
                # scores = cross_val_score(clf_model, train, trainlabel.values.ravel(),
                #                          cv=StratifiedKFold(n_splits=nsplits, shuffle=True),
                #                          scoring=scoring, verbose=3)
                # results.append(scores)
                # names.append(name)


                ###############################
                # Get XGB Feature Importances #
                ###############################
                get_xgb_feature_importances(clf_model, train, trainlabel, target_label)

                ######################
                # Decision Tree plot #
                ######################
                # for plotting decision tree ########################################
                # clf_trained = train_classifier(clf_model,train, trainlabel)
                # plot_dt(clf_trained, list(train))

                #########################################
                # GET accuracy/precision/recall/f1score #
                #########################################
                test_pred = cross_val_predict(clf_model, train, trainlabel,
                                              cv=StratifiedKFold(n_splits=nsplits, shuffle=True), verbose=5)
                precision, recall, fscore, support = score(trainlabel, test_pred, labels=labels, average=scoring_2)
                # write_results_to_file("/home/bill/Desktop/thesis/scores_" + city + ".txt", city, attr, trainlabel,
                #                       test_pred, precision, recall, fscore)
                print("TYPE: ", target_label)
                print_results(city, attr, scoring_2, name, trainlabel, test_pred, precision, recall, fscore, support)

                print(a)

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
                    print_results(city, attr, scoring_2, name, trainlabel, test_pred, precision, recall, fscore,
                                  support)
                # names.append(name)
                # results.append(res)

        title = title_lab + "_"
        for a in attr:
            title+= a + "_"
        title+="rem_"
        for rem in rem_labels:
            title+= rem + "_"
        # Visualize Classifiers accuracy
        # visualize_clf_accuracy(scoring, results, names, len(models), title, city)
        labels = sorted(labels)
        cm = confusion_matrix(testlabel.type.tolist(), test_pred, labels=labels)

        print(test_pred)
        print(testlabel.type.tolist())
        print("Confusion matrix:\n%s" % cm)
        # Visualize Conf. Matrix
        # plot_confusion_matrix(cm=cm,
        #                       normalize=True,
        #                       target_names=labels, city=city,
        #                       title=title)
        #cm.plot(normailized=True)
        #plt.show()

        print("Done...")