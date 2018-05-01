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


class ALTable(object):
    pass


def create_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("ID", String, primary_key=True, nullable=False),
              Column("Name", String),
              Column("Type1", String),
              Column("Type2", String),
              Column("Type3", String),
              Column("Type4", String),
              Column("Type5", String),
              Column("Reviews_Count", Numeric),
              Column("Photos_Count", Numeric),
              Column("Lat", Numeric),
              Column("Lng", Numeric),
              Column("PopTimes", String),
              Column("Json_String", String))
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


def insert_data(google_json, api_key):
    Base = automap_base()
    # Connect to the database
    db = create_engine("sqlite:///google_places.db")
    # create object to manage table definitions
    metadata = MetaData(db)
    # create table if it doesn't exist
    create_table(db, "google_places", metadata)
    # reflect the tables
    Base.prepare(db, reflect=True)
    GTable = Base.classes.google_places
    data_table = Table('google_places', metadata, autoload=True)
    # create a Session
    session = Session(db)
    gid, name, types, reviews_count, photos_count, lat, lng = extract_data_from_json(google_json)
    json_string = json.dumps(google_json)
    pop_times = get_pop_times_from_id(api_key, gid)
    try:
        session.add(
            GTable(ID=gid, Name=name, Type1= types[0], Type2= types[1], Type3= types[2],
                   Type4= types[3], Type5 = types[4], Reviews_Count=reviews_count,
                   Photos_Count=photos_count, Lat=lat, Lng=lng, PopTimes=pop_times,
                   Json_String=json_string))
        session.commit()
        print("~~ ", google_json["name"], " INSERTED!")

    except Exception as err:
        session.rollback()
        print("# NOT INSERTED: ", err)


def setup():
    API_KEY = google_config.api_key
    google_places = GooglePlaces(API_KEY)
    return  google_places, API_KEY


def get_places_by_ll(google_places, ll, rad):
    return google_places.nearby_search(lat_lng=ll, radius=rad)


def get_pop_times_from_id(api_key, place_id):
    pop_times = populartimes.get_id(api_key, place_id)
    if "populartimes" in pop_times:
        pop_times_string = json.dumps(pop_times)
    else:
        pop_times_string = None
    return pop_times_string


if __name__ == "__main__":
    google_places, api_key = setup()
    # databases
    ams_db ='../../../databases/ams_bb.db'
    ath_db = '../../../databases/ath_bb.db'
    # select osm db
    DB = ams_db
    # connect to osm db
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # select which pois'types we investigate
    # select which pois'types we investigate
    pois_ids = get_osm_data.get_ids_by_category(c, "'amenity'", "('cinema', 'nightclub', 'theatre', "
                                                              "'bar', 'cafe', 'fast_food', 'pub', 'restaurant'"
                                                              "'pharmacy', 'hospital', 'university','school'"
                                                              "'cemetery', 'police', 'church', 'archaeologogical_site')")
    # choose radius (how far from the osm ll)
    rad = 20
    count = 0
    ###############
    # FOR EACH POI#
    ###############
    for poi in pois_ids:
        poi_id = poi[0]
        lat_long = get_osm_data.get_lat_long_from_id(c,poi_id)
        ll = {"lat": str(lat_long[0][1]), "lng": str(lat_long[0][2])}
        query_results = google_places.nearby_search(lat_lng=ll, radius=rad)
        pp = pprint.PrettyPrinter(indent=4)

        for place in query_results.places:
            place.get_details()
            google_json = place.details
            if any(x in google_not_wanted_types for x in google_json["types"]):
                continue
            else:
                insert_data(google_json, api_key)
                #pp.pprint(place.details)