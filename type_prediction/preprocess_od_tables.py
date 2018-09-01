import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# Importing Gensim
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def read_table_as_df(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    return pd.read_sql_query("select * from {tab}".format(tab=table), con=conn)


def aggregation_functions():
    aggregation_functions_oid = {'point': 'first', 'panosid': 'first', 'type':'first',
                             'person':'any', 'personcount': 'sum', 'tree':'any', 'treecount': 'sum',
                             'clothing': 'any', 'clothingcount': 'sum',
                             'man': 'any', 'mancount': 'sum',
                             'woman': 'any', 'womancount': 'sum',
                             'houseplant': 'any', 'houseplantcount': 'sum',
                             'flower': 'any', 'flowercount': 'sum',
                             'building': 'any', 'buildingcount': 'sum',
                             'house': 'any', 'housecount': 'sum',
                             'conveniencestore': 'any', 'conveniencestorecount': 'sum',
                             'office': 'any', 'officecount': 'sum',
                             'streetlight': 'any', 'streetlightcount': 'sum',
                             'trafficlight': 'any', 'trafficlightcount': 'sum',
                             'trafficsign': 'any', 'trafficsigncount': 'sum',
                             'tent': 'any', 'tentcount': 'sum',
                             'vehicle': 'any', 'vehiclecount': 'sum',
                             'landvehicle': 'any', 'landvehiclecount': 'sum',
                             'car': 'any', 'carcount': 'sum',
                             'bike': 'any', 'bikecount': 'sum',
                             'boat': 'any', 'boatcount': 'sum',
                             'lat': 'first', 'lng':'first',
                             'year': 'first'
                             }
    aggregation_functions_sf = {'point': 'first', 'scene1': 'sum'}

    return aggregation_functions_oid


def aggregate_and_store_od_oid(table, df, db, f):
    # aggregate matched_od_oid_ams
    df_new = df.groupby(df['placesid']).aggregate(f)
    df_new.to_sql(table, db)


def aggregate_and_store_scene_features(table, scenes):
    df = read_table_as_df(table)
    df = df[['point', 'typeofenvironment', 'scene1', 'scene2', 'scene3', 'scene4']]
    # get all the values from those cols
    #cols = pd.unique(df[['scene1', 'scene2', 'scene3', 'scene4']].values.ravel('K'))
    dfs = []
    # create dummy dataframes for each column
    for s in scenes:
        dfs.append(get_dumies_for_scene(s))
    df = dfs[0]
    # add them
    for i in range(len(scenes)-1):
        df = df.set_index('point').add(dfs[i+1].set_index('point'), fill_value=0).reset_index()
    # convert column point to numeric
    df["point"] = pd.to_numeric(df["point"])
    df = df.sort_values(by=["point"])
    df.to_sql('matched_agg_scene_features_ams', db, index=False)


def get_dumies_for_scene(s):
    df_scene = df[["point", s]]
    df_scene.columns = ["point", "scene"]
    df_scene = pd.get_dummies(df_scene, columns=["scene"], prefix='', prefix_sep='')
    df_scene = df_scene.groupby(df_scene['point'], as_index=False).aggregate('sum')
    return df_scene

# preprocess for object detection tables
# Perform aggregations - One row per place


if __name__ == '__main__':
    # combine and aggregate:
    # (1) matched_od_oid_ams
    # (2) matched_scene_features_ams
    df = read_table_as_df("matched_od_oid_ams")
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    aggregate_and_store_od_oid("matched_agg_od_oid_ams", read_table_as_df("matched_od_oid_ams"),
                               db, aggregation_functions())

    # aggregate matched_scene_features_ams (+ one hot encoding)
    scenes = ["scene1", "scene2", "scene3", "scene4"]
    aggregate_and_store_scene_features("matched_scene_features_ams", scenes)
    #print(df[df["point"]=='5']["scene_parking_lot"])


