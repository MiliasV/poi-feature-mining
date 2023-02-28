from sklearn import linear_model
from sklearn import datasets  ## imports datasets from scikit-learn
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def create_linear_reg_model(data):
    clf = linear_model.LinearRegression()
    clf.fit([[getattr(t, 'x%d' % i) for i in range(1, 8)] for t in texts],[t.y for t in texts])
    return clf


def get_trained_model(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    df = pd.read_sql_query("select * from {tab} where match='1' LIMIT 1000".format(tab=table), con=conn)
    df2 = pd.read_sql_query("select * from {tab}  where match='0' LIMIT 1000".format( tab=table), con=conn)
    df = df.append(df2)
    # remove id column
    df = df.drop(["id"], axis=1)
    df = df.fillna(value=-1)
    #print(X["match"].value_counts())
    train, test = train_test_split(df, test_size=0.2)
    ytrain = pd.DataFrame(train.match, columns=["match"])
    ytest = pd.DataFrame(test.match, columns=["match"])
    train = train.drop(["match"], axis=1)
    test = test.drop(["match"], axis=1)
    # random forest
    clf = RandomForestClassifier()
    clf.fit(train, ytrain)
    print(clf.score(test, ytest))
    return clf


if __name__ == '__main__':
    clf = get_trained_model("similarities_ams_table")
    #print(clf.feature_importances_)
    # print(clf.score(test, ytest))
    # print(clf.n_classes_)
    # lm = linear_model.LinearRegression()
    # model = lm.fit(X, y)
    # predictions = lm.predict(X)
    # print(predictions)[0:5]
    # Put the target (housing value -- MEDV) in another DataFrame
    #target = pd.DataFrame(data.target, columns=["MEDV"])
    # clf = create_linear_reg_model("")
    # print(clf.coef)