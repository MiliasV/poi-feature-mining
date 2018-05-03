import sys
sys.path.append("..")
import google.google_config as google_config
from googleplaces import GooglePlaces, types, lang
import sqlite3
import osm_pois.get_osm_data as get_osm_data
import pprint
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker, mapper, Session
from sqlalchemy.ext.automap import automap_base
import simplejson as json
import populartimes
from geoalchemy2 import Geometry


##########################################################
# Script for finding and storing POIs from Google Places #
##########################################################

def setup():
    API_KEY = google_config.api_key
    google_places = GooglePlaces(API_KEY)
    return  google_places, API_KEY


class ALTable(object):
    pass


def create_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("rscount", Numeric),
              Column("phcount", Numeric),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def extract_data_from_json(google_json):
    name = google_json["name"]
    id = google_json["place_id"]
    reviews_count = 0
    photos_count = 0
    if "reviews" in google_json:
        reviews_count = len(google_json["reviews"])

    if "photos" in google_json:
        photos_count = len(google_json["photos"])

    # at least 5 types to be inserted
    while len(google_json["types"])<5:
        google_json["types"].append(None)

    lat = google_json["geometry"]["location"]["lat"]
    lng = google_json["geometry"]["location"]["lng"]
    return id, name, google_json["types"], reviews_count, photos_count, lat, lng


#def get_poi_within_radius(table, radius):


def insert_data(google_json, api_key, ogc_fid):
    Base = automap_base()
    # Connect to the database
    #db = create_engine("sqlite:///../../../databases/google_places.db")
    db = create_engine('postgresql://postgres:postgres@localhost/pois')

    # create object to manage table definitions
    metadata = MetaData(db)
    # create table if it doesn't exist
    create_table(db, "google_ams_centroids_40", metadata)
    # reflect the tables
    Base.prepare(db, reflect=True)
    GTable = Base.classes.google_ams_centroids_40
    #data_table = Table('google_ams_centroids_40', metadata, autoload=True)
    # create a Session
    session = Session(db)
    gid, name, types, reviews_count, photos_count, lat, lng = extract_data_from_json(google_json)
    json_string = json.dumps(google_json)
    pop_times = get_pop_times_from_id(api_key, gid)
    geo = 'POINT({} {})'.format(lng, lat)
    try:
        session.add(
            GTable(id=gid, name=name, type1=types[0], type2= types[1], type3= types[2],
                   type4= types[3], type5 = types[4], rscount=reviews_count,
                   phcount=photos_count, lat=lat, lng=lng, poptimes=pop_times, geom=geo,
                   json=json_string))
        session.commit()
        print("~~ ", google_json["name"], " INSERTED!")

    except Exception as err:
        session.rollback()
        print("# NOT INSERTED: ", err)


def get_places_by_ll(google_places, ll, rad):
    return google_places.nearby_search(lat_lng=ll, radius=rad)


def get_pop_times_from_id(api_key, place_id):
    pop_times = populartimes.get_id(api_key, place_id)
    if "populartimes" in pop_times:
        pop_times_string = json.dumps(pop_times["populartimes"])
    else:
        pop_times_string = None
    return pop_times_string


def get_last_id_from_logfile(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()
        f.close()
        return int(lines[-1])


if __name__ == "__main__":
    google_places, api_key = setup()
    city = "ams"
    rad = 20
    count = 0
    # databases
    center_db = "/home/bill/Desktop/thesis/maps/" + city + "_center_centroids_40.sqlite"
    logfile = "/home/bill/Desktop/thesis/logfiles/" + city + "_point_iterations.txt"
    DB = center_db
    # connect to osm db
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    last_searched_id = get_last_id_from_logfile(logfile)
    print(last_searched_id)
    c.execute("SELECT ogc_fid, lat, Lon "
              "FROM ams_center_centroids_40 "
              "WHERE id>={last_id}-1".format(last_id=last_searched_id))
    #################
    # FOR EACH POINT#
    #################
    for ogc_fid, point_lat, point_lng in c:
        print("POINT: ", ogc_fid, point_lat, point_lng)
        # keep the id of the last searched point
        with open(logfile, "w") as text_file:
            print(f"Last searched point \n{ogc_fid}", file=text_file)
            text_file.close()
        ll = {"lat": str(point_lat), "lng": str(point_lng)}
        query_results = google_places.nearby_search(lat_lng=ll, radius=rad)
        #pp = pprint.PrettyPrinter(indent=4)
        for place in query_results.places:
             place.get_details()
             google_json = place.details
        #     # if any(x in google_not_wanted_types for x in google_json["types"]):
        #     #     continue
        #     # else:
             insert_data(google_json, api_key, ogc_fid)
