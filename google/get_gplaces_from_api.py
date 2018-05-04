import sys
sys.path.append("..")
import google.google_config as google_config
from googleplaces import GooglePlaces, types, lang
import sqlite3
import pprint
import simplejson as json
import populartimes
import create_pois_table


##########################################################
# Script for finding and storing POIs from Google Places #
##########################################################

def setup():
    API_KEY = google_config.api_key
    google_places = GooglePlaces(API_KEY)
    return  google_places, API_KEY


def extract_data_from_json(google_json):
    name = google_json["name"]
    id = google_json["place_id"]
    pop_times = google_json["poptimes"]
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
    return id, name, google_json["types"], reviews_count, photos_count, pop_times, lat, lng


def insert_data(session, GTable, search_lat, search_lng, google_json, ogc_fid,
                count_places, count_duplicates):
    gid, name, types, reviews_count, photos_count, pop_times, lat, lng = extract_data_from_json(google_json)
    json_string = json.dumps(google_json)
    geo = 'POINT({} {})'.format(lng, lat)
    try:
        session.add(
            GTable(id=gid, originalpointindex=ogc_fid, name=name, type1=types[0], type2= types[1], type3= types[2],
                   type4= types[3], type5 = types[4], rscount=reviews_count,
                   phcount=photos_count, lat=lat, lng=lng, searchlat=search_lat,
                   searchlng=search_lng, poptimes=pop_times, geom=geo,
                   json=json_string))
        session.commit()
        count_places += 1
        print("~~ ", google_json["name"], " INSERTED!")

    except Exception as err:
        session.rollback()
        count_duplicates += 1
        print("# NOT INSERTED: ", err)
    return count_places, count_duplicates


def insert_count_data(session, CTable, poi_number, count_places, count_dupl):
    try:
        session.add(CTable(id=poi_number, countplaces=count_places, countduplicates=count_dupl))
        session.commit()
    except Exception as err:
        session.rollback()
        print("# [COUNT] NOT INSERTED: ", err)


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


def add_pop_times_in_places(api_key, query_results):
    # for each place gotten from google
    for place in query_results.places:
        place.get_details()
        gid = place.details["place_id"]
        place.details["poptimes"] = get_pop_times_from_id(api_key, gid)
    return query_results.places


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
    c.execute("SELECT ogc_fid, lat, Lon "
              "FROM ams_center_centroids_40 "
              "WHERE id>={last_id}-1".format(last_id=last_searched_id))
    # setup table and session
    # define which table
    session, GTable, CTable = create_pois_table.setup_db("google_ams_center_40",
                                                         "google_ams_center_40_count")
    #define types we don't care about
    google_not_wanted_types = ["route"]
    # For each point --> search nearby in google
    for ogc_fid, point_lat, point_lng in c:
        count_places = 0
        count_duplicates = 0
        print("POINT: ", ogc_fid, point_lat, point_lng)
        # keep the id of the last searched point
        with open(logfile, "w") as text_file:
            print(f"Last searched point \n{ogc_fid}", file=text_file)
            text_file.close()
        ll = {"lat": str(point_lat), "lng": str(point_lng)}
        query_results = google_places.nearby_search(lat_lng=ll, radius=rad)
        query_results.places = add_pop_times_in_places(api_key, query_results)
        # for each place gotten from google
        for place in query_results.places:
            google_json = place.details
            gid = google_json["place_id"]
            if not any(x in google_not_wanted_types for x in google_json["types"]):
                count_places, count_duplicates = insert_data(session, GTable, point_lat,
                                                             point_lng, google_json, ogc_fid,
                                                             count_places, count_duplicates)
        insert_count_data(session, CTable, ogc_fid, count_places, count_duplicates)
